#!/usr/bin/env python3
"""Paired Windows CPU latency/throughput acceptance. This command runs benchmarks.

One invocation is one performance round for each of the three fixed raw samples.
The persistent task ledger defaults to three rounds, including failed runs.
Explicit recorded user authorizations can add two initial rounds, then one per followup.
Correctness preflight, loading, cold profiles, allocation probes and serialization
are separate from the five official warmed samples. No adaptive repetitions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid

import cpu_parity as gate
import cpu_variants as full_gate
import windows_cpu_load as cpu_load

BUDGETS = (1, 2, 4, 8)
FAMILIES = ("rec", "small_rec", "medium_rec")
WORKLOADS = ("sample", "dense-multilingual", "text-long")
SOURCE_HASHES = {"sample": "f437d3231d0661a0497f0ad71ff22d4d3c1b66d3dd81bdcee33b058b4828a86e"}
WORKLOAD_HASHES = {**SOURCE_HASHES,
                  "dense-multilingual": "0f5e48855b450f90419163680083dbfc525bdb3ae0527b3e26d7b9ba8f19399a",
                  "text-long": "7b3731d14ac5f87a518f7c86c8c44862677c35945cf3dca1be7f2f23d0d2d02e"}
PRE_RESERVATION_PHASES = ("before-preflight-and-round-reservation", "after-preflight-before-round-reservation")
IDLE_QUIET_GAP_SECONDS = 30
RUST_RETAINED_FIELDS = ("activation_bytes", "workspace_bytes", "kernel_scratch_bytes", "nonkernel_workspace_bytes")
CSHARP_OBSERVED_FIELDS = ("activation_bytes", "workspace_bytes")
RETAINED_MEMORY_POLICY = {
    "version": 2,
    "id": "rust-bounded-reuse-reference-observation-v2",
    "comparison": "after_samples and repeated_shape must not exceed the warmed value for any listed metric",
    "rust": {"required_for_acceptance": True, "metrics": list(RUST_RETAINED_FIELDS),
             "scope": "retained request capacities including kernel buffers"},
    "csharp": {"required_for_acceptance": False, "metrics": list(CSHARP_OBSERVED_FIELDS),
               "scope": "observed payloads and estimated collection slots; opaque collection backing capacity excluded",
               "exact_total_bound_established": False,
               "contract": "The pinned reference is immutable. Report all phases, growth deltas and numeric-pool diagnostics; observed stability does not prove a total C# memory bound."},
}


class IdleGate:
    """Wait only before reserving a round; a started benchmark never retries."""
    def __init__(self, report: dict, report_path: Path, verify_frozen_inputs, wait_for_idle: bool = False):
        self.report = report
        self.report_path = report_path
        self.verify_frozen_inputs = verify_frozen_inputs
        self.wait_for_idle = wait_for_idle

    def check(self, phase: str) -> None:
        while True:
            self.verify_frozen_inputs()
            snapshot = {"phase": phase, **cpu_load.sample_load(interval_seconds=1)}
            self.verify_frozen_inputs()
            can_wait = self.wait_for_idle and phase in PRE_RESERVATION_PHASES and "round_id" not in self.report
            idle = snapshot.get("idle") is True
            snapshot["pre_reservation_wait_allowed"] = can_wait
            self.report["host_load_checks"].append(snapshot)
            self.report["host_load_state"] = {"phase": phase,
                "status": "idle" if idle else "waiting_for_idle" if can_wait else "contended",
                "round_reserved": "round_id" in self.report}
            gate.write_json(self.report_path, self.report)
            if idle:
                return
            if not can_wait:
                gate.fail("CPU contention prevents performance acceptance: " + "; ".join(snapshot.get("reasons", [])))
            print(f"Waiting for idle CPU before round reservation; next load check in {IDLE_QUIET_GAP_SECONDS} seconds.", flush=True)
            time.sleep(IDLE_QUIET_GAP_SECONDS)

    def preflight_and_reserve(self, preflight_once, reserve_once):
        self.check(PRE_RESERVATION_PHASES[0])
        preflight = preflight_once()
        self.check(PRE_RESERVATION_PHASES[1])
        reserve_once()
        return preflight


def verify_frozen_corpus(corpus: Path, manifest_sha256: str) -> None:
    # Check the original manifest before trusting its per-image hashes. A raw
    # image and its manifest entry must not be replaced consistently while idle.
    if gate.sha256(corpus / "corpus.json") != manifest_sha256:
        gate.fail("Frozen acceptance corpus manifest changed")
    gate.validate_corpus(corpus)


def runtime_sources(snapshot: dict) -> dict:
    return {name: digest for name, digest in snapshot["files"].items()
            if name.startswith(("src/", ".cargo/")) or name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "README.md", "build.rs")}


def compiler_build_configuration() -> dict:
    """Identify the actual queried compiler and release inputs, without building."""
    flags = gate.build_flags()
    compiler = gate.run([gate.require_tool("rustc"), "--version", "--verbose"], capture=True)
    if flags != gate.build_flags():
        gate.fail("Rust build configuration changed while identifying the compiler")
    return {"rustc": compiler, "build_flags": flags}


def verify_compiler_build_configuration(environment: dict) -> dict:
    if (not isinstance(environment, dict) or not isinstance(environment.get("rustc"), str)
            or not environment["rustc"].strip() or not isinstance(environment.get("build_flags"), dict)
            or not environment["build_flags"]):
        gate.fail("The full correctness baseline lacks compiler/build configuration evidence")
    configuration = compiler_build_configuration()
    if configuration != {name: environment[name] for name in ("rustc", "build_flags")}:
        gate.fail("Rust compiler/build configuration changed; run a fresh full correctness gate before acceptance")
    return configuration


def validate_correctness_prerequisite(bundle: Path, models: Path, lock: dict, sources: dict) -> dict:
    """Recheck retained full evidence without builds, inference or ledger writes."""
    metadata = full_gate.read_identity(bundle, lock)
    provenance = metadata["rust_build"]
    if (provenance.get("provenance") != "built_by_driver"
            or runtime_sources(provenance["sources"]) != runtime_sources(sources)):
        gate.fail("Run a fresh full correctness gate for the unchanged native Auto library before performance acceptance")
    configuration = verify_compiler_build_configuration(metadata.get("benchmark_environment"))
    # The acceptance helper may change independently of the library. Bind the
    # old capture to its own frozen producer, not the mutable target executable
    # or the newly compiled acceptance helper.
    executable = bundle / "binaries/rust/cpu_parity.exe"
    if gate.sha256(executable) != provenance.get("executable_sha256"):
        gate.fail("Frozen full-correctness Rust producer changed")
    gate.verify_source_archive(provenance["sources"], bundle / "sources/rust")
    exporter_archive = bundle / "sources/exporter"
    archived_exporter_available = exporter_archive.is_dir()
    if archived_exporter_available:
        # Early retained C# oracles have a frozen runtime and source identity,
        # but no source archive. Match the maintained replay contract: verify
        # any existing archive without fabricating absent historical bytes.
        gate.verify_source_archive(metadata["exporter_build"]["sources"], exporter_archive)
    full_gate.verify_inputs(models, bundle / "corpus", lock, metadata["corpus_sha256"])
    verification = full_gate.verify_original(bundle, lock["tolerances"])
    if verification["passed"] is not True:
        gate.fail("Retained full-correctness imports, tensors or public results no longer pass")
    return {"schema_version": 1, "passed": True, "run_sha256": gate.sha256(bundle / "run.json"),
            "report_sha256": gate.sha256(bundle / "report.json"), "rust_build": provenance,
            "exporter_build": metadata["exporter_build"],
            "capture_sha256": {name: gate.sha256(bundle / name / "capture.json") for name in
                               ("csharp-reference", "csharp-auto", "rust-reference", "rust-auto")},
            "archived_exporter_sources_verified": archived_exporter_available,
            "runtime_sources": runtime_sources(provenance["sources"]),
            "compiler_build_configuration": configuration,
            "verification": verification,
            "validator_sha256": gate.sha256(Path(full_gate.__file__)),
            "comparison_helper_sha256": gate.sha256(Path(gate.__file__))}


def prepare_corpus(source: Path, output: Path) -> dict:
    gate.validate_corpus(source)
    records = {image["id"]: image for image in gate.read_json(source / "corpus.json")["images"]}
    raw = {}
    for name, expected in SOURCE_HASHES.items():
        record = records[name]
        if record["sha256"] != expected or (record["width"], record["height"]) != (320, 320):
            gate.fail(f"The reviewed {name} acceptance source changed")
        raw[name] = gate.checked_child(source, record["file"]).read_bytes()
    raw["dense-multilingual"] = raw["sample"] + raw["sample"]
    # A fixed rectangle contains natural Chinese/Latin glyphs, with no resampling.
    # Four adjacent copies make each line wider than those on the source page.
    width, height = 928, 128
    long = bytearray([255]) * (width * height * 3)
    for top in (12, 48, 84):
        for row, source_y in enumerate(range(250, 270)):
            glyphs = raw["sample"][(source_y * 320 + 15) * 3:(source_y * 320 + 240) * 3]
            start = ((top + row) * width + 14) * 3
            long[start:start + 900 * 3] = glyphs * 4
    raw["text-long"] = bytes(long)
    output.mkdir(parents=True)
    images = []
    for name in WORKLOADS:
        path = output / (name + ".bgr")
        path.write_bytes(raw[name])
        width, height = {"sample": (320, 320), "dense-multilingual": (320, 640), "text-long": (928, 128)}[name]
        if gate.sha256(path) != WORKLOAD_HASHES[name]:
            gate.fail(f"The reviewed {name} acceptance derivation changed")
        images.append({"id": name, "width": width, "height": height, "file": path.name, "sha256": gate.sha256(path)})
    manifest = {"schema_version": 1, "pixel_format": "bgr24", "images": images,
                "dense_derivation": "Two vertical copies of the pinned multilingual raw sample, without decoding or fonts",
                "long_derivation": {"rectangle_xyxy": [15, 250, 240, 270], "copies_per_line": 4,
                                    "line_left": 14, "line_tops": [12, 48, 84], "background_bgr": [255, 255, 255]},
                "source_hashes": SOURCE_HASHES}
    gate.write_json(output / "corpus.json", manifest)
    gate.validate_corpus(output)
    return manifest


@contextmanager
def locked_ledger(ledger: Path):
    """One Windows lock protects both authorizations and round reservations."""
    import msvcrt
    ledger.parent.mkdir(parents=True, exist_ok=True)
    # Lock a stable file, rather than the data inode that atomic replacement
    # changes. Never delete this sidecar: concurrent writers must lock the same
    # byte even while a completed ledger replaces its predecessor.
    lock_path = ledger.with_name(ledger.name + ".lock")
    with lock_path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            gate.fail("Another acceptance runner is updating the task's performance ledger")
        try:
            existed = ledger.exists()
            original = ledger.read_bytes() if existed else b""
            if existed and not original.strip():
                gate.fail("Existing performance ledger is empty; restore its reservation history instead of resetting rounds")
            state = json.loads(original) if existed else {"schema_version": 1, "tasks": {}}
            if state.get("schema_version") != 1 or not isinstance(state.get("tasks"), dict):
                gate.fail("Unsupported performance ledger")
            before = json.dumps(state, sort_keys=True)
            yield state, original
            if not existed or json.dumps(state, sort_keys=True) != before:
                if ledger.exists() != existed or (existed and ledger.read_bytes() != original):
                    gate.fail("Performance ledger changed outside its lock; refusing a lost update")
                # A failed write/flush/replace must leave every earlier
                # reservation intact. Persist a complete exclusive temporary
                # file before publishing it with one same-directory replace.
                temporary = ledger.with_name(f".{ledger.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("xb") as pending:
                        pending.write(json.dumps(state, indent=2).encode("utf-8"))
                        pending.flush()
                        os.fsync(pending.fileno())
                    os.replace(temporary, ledger)
                except BaseException:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        # Preserve the commit failure if cleanup also fails.
                        pass
                    raise
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _initial_round_policy(state: dict, task: str) -> dict:
    extensions = state.get("round_cap_extensions", {})
    if not isinstance(extensions, dict):
        gate.fail("Malformed task-specific performance-round authorizations")
    extension = extensions.get(task)
    if extension is None:
        return {"base_round_cap": 3, "additional_rounds": 0, "maximum_rounds": 3, "authorization": None}
    if not isinstance(extension, dict):
        gate.fail("Malformed task-specific performance-round authorization")
    authorization = extension.get("authorization", {})
    try:
        timestamp = datetime.datetime.fromisoformat(authorization["recorded_at_utc"].replace("Z", "+00:00"))
        uuid.UUID(authorization["id"])
        valid = (type(extension.get("base_round_cap")) is int and extension["base_round_cap"] == 3
                 and type(extension.get("additional_rounds")) is int and extension["additional_rounds"] == 2
                 and type(extension.get("maximum_rounds")) is int and extension["maximum_rounds"] == 5
                 and authorization.get("task_id") == task and authorization.get("source") == "explicit_user_instruction"
                 and isinstance(authorization.get("text"), str) and bool(authorization["text"].strip())
                 and timestamp.utcoffset() == datetime.timedelta(0)
                 and isinstance(authorization.get("prior_ledger_sha256"), str) and len(authorization["prior_ledger_sha256"]) == 64
                 and isinstance(authorization.get("prior_ledger_backup"), str))
    except (KeyError, TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        gate.fail("Malformed task-specific performance-round authorization")
    backup = Path(authorization["prior_ledger_backup"])
    if not backup.is_file() or gate.sha256(backup) != authorization["prior_ledger_sha256"]:
        gate.fail("The pre-authorization ledger backup is missing or changed")
    original_task = gate.read_json(backup).get("tasks", {}).get(task)
    current_task = state["tasks"].get(task)
    if (not isinstance(original_task, dict) or not isinstance(current_task, dict)
            or any(current_task.get(digest, [])[:len(rows)] != rows for digest, rows in original_task.items())):
        gate.fail("Original task reservations changed after the user-authorized extension")
    return extension


def _ledger_prefix(prior, current) -> bool:
    """Existing reservation/auth records are immutable; new dict keys/list rows may append."""
    if isinstance(prior, dict):
        return isinstance(current, dict) and all(key in current and _ledger_prefix(value, current[key])
                                                 for key, value in prior.items())
    if isinstance(prior, list):
        return isinstance(current, list) and current[:len(prior)] == prior
    return type(prior) is type(current) and prior == current


def _canonical_uuid(value) -> bool:
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _authorization_ids(state: dict) -> dict:
    ids = {}
    initial = state.get("round_cap_extensions", {})
    followups = state.get("round_cap_followups", {})
    if not isinstance(initial, dict) or not isinstance(followups, dict):
        gate.fail("Malformed task-specific performance-round authorization chain")
    for task, extension in initial.items():
        if not isinstance(extension, dict):
            gate.fail("Malformed task-specific performance-round authorization")
        authorization = extension.get("authorization", {})
        id_ = authorization.get("id") if isinstance(authorization, dict) else None
        if not _canonical_uuid(id_) or id_ in ids:
            gate.fail("Malformed or reused performance-round authorization ID")
        ids[id_] = task
    for task, entries in followups.items():
        if not isinstance(entries, list):
            gate.fail("Malformed task-specific performance-round followup list")
        for entry in entries:
            authorization = entry.get("authorization", {}) if isinstance(entry, dict) else {}
            id_ = authorization.get("id") if isinstance(authorization, dict) else None
            if not _canonical_uuid(id_) or id_ in ids:
                gate.fail("Malformed or reused performance-round authorization ID")
            ids[id_] = task
    return ids


def round_policy(state: dict, task: str) -> dict:
    """Resolve immutable initial authorization plus hash-bound append-only +1 records."""
    _authorization_ids(state)
    initial = _initial_round_policy(state, task)
    entries = state.get("round_cap_followups", {}).get(task, [])
    if not entries:
        return initial  # Preserve the exact historical three/five-round policy schema.
    if initial["authorization"] is None:
        gate.fail("A followup requires an existing initial two-round authorization")
    policy = dict(initial)
    chain = [initial["authorization"]["id"]]
    for index, entry in enumerate(entries):
        auth = entry.get("authorization", {})
        try:
            timestamp = datetime.datetime.fromisoformat(auth["recorded_at_utc"].replace("Z", "+00:00"))
            valid = (set(entry) == {"additional_rounds", "previous_maximum_rounds", "maximum_rounds", "authorization"}
                and type(entry["additional_rounds"]) is int and entry["additional_rounds"] == 1
                and type(entry["previous_maximum_rounds"]) is int and entry["previous_maximum_rounds"] == policy["maximum_rounds"]
                and type(entry["maximum_rounds"]) is int and entry["maximum_rounds"] == policy["maximum_rounds"] + 1
                and auth["source"] == "explicit_user_instruction" and auth["task_id"] == task
                and auth["prior_authorization_id"] == chain[-1]
                and isinstance(auth["text"], str) and bool(auth["text"].strip())
                and timestamp.utcoffset() == datetime.timedelta(0)
                and isinstance(auth["prior_ledger_sha256"], str) and len(auth["prior_ledger_sha256"]) == 64
                and isinstance(auth["prior_ledger_backup"], str))
        except (KeyError, TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            gate.fail("Malformed or stale performance-round followup authorization")
        backup = Path(auth["prior_ledger_backup"])
        if not backup.is_file() or gate.sha256(backup) != auth["prior_ledger_sha256"]:
            gate.fail("The followup pre-authorization ledger backup is missing or changed")
        prior = gate.read_json(backup)
        if (not _ledger_prefix(prior, state)
                or prior.get("round_cap_followups", {}).get(task, []) != entries[:index]
                or prior.get("round_cap_extensions", {}).get(task) != state["round_cap_extensions"][task]):
            gate.fail("Original full-ledger reservations or authorization chain changed after followup")
        previous_policy = round_policy(prior, task)
        if (previous_policy["maximum_rounds"] != policy["maximum_rounds"]
                or previous_policy["authorization"]["id"] != chain[-1]):
            gate.fail("Followup backup does not identify the prior latest authorization")
        chain.append(auth["id"])
        policy = {"base_round_cap": 3, "additional_rounds": entry["maximum_rounds"] - 3,
                  "maximum_rounds": entry["maximum_rounds"], "authorization": dict(auth),
                  "authorization_chain": list(chain), "followup_authorizations": entries[:index + 1]}
    return policy


def record_followup_round_extension(ledger: Path, task: str, additional_rounds: int,
                                    authorization_id: str, prior_authorization_id: str,
                                    authorization_text: str) -> dict:
    """Record ONE explicit +1 instruction only; this function never reserves or runs a round."""
    if (type(additional_rounds) is not int or additional_rounds != 1 or not task
            or not _canonical_uuid(authorization_id) or not _canonical_uuid(prior_authorization_id)
            or authorization_id == prior_authorization_id
            or not isinstance(authorization_text, str) or not authorization_text.strip()):
        gate.fail("An existing task, distinct new/prior UUIDs, explicit instruction and exactly one further round are required")
    with locked_ledger(ledger) as (state, original):
        if task not in state["tasks"]:
            gate.fail("Record a followup against the existing task ID; never reset reservations")
        policy = round_policy(state, task)
        if policy["authorization"] is None:
            gate.fail("A followup requires an existing initial two-round authorization")
        ids = _authorization_ids(state)
        entries = state.get("round_cap_followups", {}).get(task, [])
        if authorization_id in ids:
            latest = entries[-1]["authorization"] if entries else None
            if (latest is not None and latest["id"] == authorization_id
                    and latest["prior_authorization_id"] == prior_authorization_id
                    and latest["text"] == authorization_text.strip()):
                return policy  # Idempotent retry, including after a consumed round.
            gate.fail("Conflicting or replayed performance-round authorization ID")
        if policy["authorization"]["id"] != prior_authorization_id:
            gate.fail("Followup must reference the latest prior authorization ID")
        prior_sha = hashlib.sha256(original).hexdigest()
        backup = ledger.with_name(f"{ledger.name}.before-followup-{authorization_id}-{prior_sha[:16]}.json")
        if backup.exists():
            if backup.read_bytes() != original:
                gate.fail("Existing followup pre-authorization backup differs")
        else:
            with backup.open("xb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
        if backup.read_bytes() != original:
            gate.fail("Followup pre-authorization ledger backup verification failed")
        entry = {"additional_rounds": 1, "previous_maximum_rounds": policy["maximum_rounds"],
            "maximum_rounds": policy["maximum_rounds"] + 1,
            "authorization": {"id": authorization_id, "source": "explicit_user_instruction", "task_id": task,
                "text": authorization_text.strip(), "prior_authorization_id": prior_authorization_id,
                "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "prior_ledger_sha256": prior_sha, "prior_ledger_backup": str(backup.resolve())}}
        state.setdefault("round_cap_followups", {}).setdefault(task, []).append(entry)
        return round_policy(state, task)


def read_round_policy(ledger: Path, task: str) -> dict:
    with locked_ledger(ledger) as (state, _):
        return round_policy(state, task)


def record_round_extension(ledger: Path, task: str, additional_rounds: int, authorization_text: str) -> dict:
    if type(additional_rounds) is not int or additional_rounds != 2 or not task or not isinstance(authorization_text, str) or not authorization_text.strip():
        gate.fail("A task ID, explicit user authorization, and exactly two additional rounds are required")
    with locked_ledger(ledger) as (state, original):
        if task not in state["tasks"]:
            gate.fail("Record the extension against the existing task ID; do not create a new task to reset rounds")
        previous = round_policy(state, task)
        if previous["additional_rounds"]:
            if previous["authorization"]["text"] != authorization_text.strip():
                gate.fail("This task already has a recorded extension; another authorization cannot stack or replace it")
            return previous
        prior_sha = hashlib.sha256(original).hexdigest()
        backup = ledger.with_name(f"{ledger.name}.before-user-extension-{prior_sha[:16]}.json")
        if backup.exists():
            if backup.read_bytes() != original:
                gate.fail("Existing pre-authorization ledger backup differs")
        else:
            with backup.open("xb") as stream:
                stream.write(original)
        if backup.read_bytes() != original:
            gate.fail("Pre-authorization ledger backup verification failed")
        extension = {"base_round_cap": 3, "additional_rounds": 2, "maximum_rounds": 5,
            "authorization": {"id": str(uuid.uuid4()), "source": "explicit_user_instruction", "task_id": task,
                "text": authorization_text.strip(), "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "prior_ledger_sha256": prior_sha, "prior_ledger_backup": str(backup.resolve())}}
        state.setdefault("round_cap_extensions", {})[task] = extension
        return extension


def reserve_round(ledger: Path, task: str, hashes: list[str], round_id: str, output: Path, expected_policy: dict | None = None) -> dict:
    """Reservations retain all history and consume the task's recorded cap."""
    with locked_ledger(ledger) as (state, _):
        policy = round_policy(state, task)
        if expected_policy is not None and policy != expected_policy:
            gate.fail("Performance-round authorization changed after the immutable plan")
        samples = state["tasks"].setdefault(task, {})
        if any(len(samples.get(digest, [])) >= policy["maximum_rounds"] for digest in hashes):
            if policy["maximum_rounds"] == 3:
                gate.fail("This task has already used three performance rounds for an acceptance sample")
            count = "five" if policy["maximum_rounds"] == 5 else "six" if policy["maximum_rounds"] == 6 else str(policy["maximum_rounds"])
            gate.fail(f"This task has already used its {count} authorized performance rounds for an acceptance sample")
        for digest in hashes:
            row = {"round_id": round_id, "output": str(output)}
            if policy["authorization"] is not None:
                row.update({"maximum_rounds": policy["maximum_rounds"], "authorization_id": policy["authorization"]["id"]})
            samples.setdefault(digest, []).append(row)
        return policy


JIT_CONTROL_SUFFIXES = ("TIEREDCOMPILATION", "TIEREDPGO", "TC_QUICKJIT", "TC_QUICKJITFORLOOPS", "TC_ONSTACKREPLACEMENT", "READYTORUN")


def jit_environment(environment: dict) -> dict:
    """Record inheritance, including explicit unset values; never tune it."""
    result = {prefix + suffix: None for prefix in ("DOTNET_", "COMPLUS_") for suffix in JIT_CONTROL_SUFFIXES}
    for name, value in environment.items():
        upper = name.upper()
        if upper.startswith(("DOTNET_", "COMPLUS_")):
            suffix = upper.split("_", 1)[1]
            if suffix.startswith(("TIERED", "PGO", "TC_", "QUICKJIT", "READYTORUN", "JIT")):
                result[upper] = value
    return dict(sorted(result.items()))


def validate_runtime_identity(capture: dict) -> None:
    identity = capture.get("runtime_identity", {})
    if not isinstance(identity, dict):
        gate.fail("Missing or invalid C# runtime/JIT provenance")
    variables = identity.get("jit_environment")
    if (not all(isinstance(identity.get(key), str) and identity[key] for key in
                ("framework_description", "environment_version", "process_architecture", "os_description", "contract"))
            or identity["process_architecture"].lower() != "x64" or not isinstance(variables, dict)
            or any(not isinstance(name, str) or (value is not None and not isinstance(value, str)) for name, value in variables.items())
            or jit_environment(variables) != variables):
        gate.fail("Missing or invalid C# runtime/JIT provenance")


POOL_TYPES = {"float": 4, "int": 4, "uint": 4, "byte": 1, "short": 2, "char": 2,
              "Vector256<float>": 32, "Vector512<float>": 64, "CtcRun": 16}
POOL_SCHEMAS = {"BufferRented": ["bufferId", "bufferSize", "poolId", "bucketId"],
                "BufferReturned": ["bufferId", "bufferSize", "poolId"]}


def validate_pool_probe(probe: dict, self_test: bool) -> None:
    fields = ("peak_observed_unreturned_bytes", "rented_bytes", "rent_count", "return_count", "unreturned_rental_bytes",
              "unmatched_return_count", "unmatched_return_bytes")
    groups = probe.get("unreturned_by_type", [])
    pools = probe.get("pools", [])
    if (probe.get("supported") is not True or probe.get("self_test") is not self_test
            or probe.get("event_source") != "System.Buffers.ArrayPoolEventSource" or not probe.get("contract")
            or any(type(probe.get(field)) is not int or probe[field] < 0 for field in fields)
            or probe["rent_count"] <= 0 or probe["return_count"] <= 0 or probe["return_count"] > probe["rent_count"]
            or not 0 <= probe["unreturned_rental_bytes"] <= probe["peak_observed_unreturned_bytes"] <= probe["rented_bytes"]
            or probe["peak_observed_unreturned_bytes"] <= 0 or probe.get("schemas") != POOL_SCHEMAS
            or not set(probe.get("numeric_types_observed", [])) or not set(probe["numeric_types_observed"]) <= set(POOL_TYPES)
            or len(pools) != len(POOL_TYPES) or {p.get("name"): p.get("element_bytes") for p in pools} != POOL_TYPES
            or len({p.get("pool_id") for p in pools}) != len(POOL_TYPES)
            or any(type(p.get("pool_id")) is not int for p in pools)
            or any(g.get("type") not in POOL_TYPES or type(g.get("count")) is not int or g["count"] <= 0
                   or type(g.get("bytes")) is not int or g["bytes"] <= 0 or type(g.get("all_meet_discard_capacity")) is not bool for g in groups)
            or sum(g["bytes"] for g in groups) != probe["unreturned_rental_bytes"]
            or sum(g["count"] for g in groups) != probe["rent_count"] - probe["return_count"]):
        gate.fail("Missing or invalid separate numeric ArrayPool scratch probe")
    boundary = probe.get("discard_boundary_self_test", {}) or {}
    measurements = boundary.get("measurements", {})
    if self_test and (probe["rent_count"] != 9 or probe["return_count"] != 9 or probe["unreturned_rental_bytes"] != 0
                      or probe["unmatched_return_count"] != 0 or set(probe.get("numeric_types_observed", [])) != set(POOL_TYPES)
                      or boundary.get("supported") is not True or boundary.get("expected_no_return_event") is not True
                      or boundary.get("requested_elements") != 8193 or type(boundary.get("actual_elements")) is not int
                      or boundary["actual_elements"] < 16384 or boundary.get("unreturned_capacity_bytes") != boundary["actual_elements"] * 4
                      or measurements.get("rent_count") != 1 or measurements.get("return_count") != 0
                      or measurements.get("unreturned_rental_bytes") != boundary["unreturned_capacity_bytes"]):
        gate.fail("The runtime ArrayPool probe did not pass its payload/identity self-test")


PHASE_RESHAPE_CONTRACT = "summed-full-ocr-session-reshape-calls-including-noops-v1"
PHASE_INITIALIZATION_CONTRACT = "diagnostic-pools-precreated-outside-run-including-default-session-bindings-v1"
PHASE_OWNED_CONTRACT = "pinned-api-intrinsically-owned-no-post-run-copy-v1"
PHASE_STAGES = ("det_preprocess", "det_graph", "det_postprocess", "crop", "cls_preprocess", "cls_graph",
                "rec_preprocess", "rec_graph", "rec_postprocess", "lines_wall", "crop_setup", "rec_acquire",
                "cls_acquire", "rec_decode", "rec_release", "cls_release", "rec_cache_get", "rec_rent",
                "rec_pool", "rec_reshape", "det_unclip")


def validate_rust_phase_profiles(profiles: list, workers: int) -> None:
    if (not isinstance(profiles, list) or len(profiles) != workers
            or any(not isinstance(profile, dict)
                   or any(type(profile.get(field)) is not int or profile[field] < 0
                          for field in ("binding_ns", "reshape_ns")) for profile in profiles)):
        gate.fail("Rust cold profiles must separate nonnegative binding and reshape work per request")


def phase_diagnostic_arguments(ledger: Path, task: str, report: dict, output: Path,
                               planned_policy: dict, frozen_plan_sha256: str) -> list[str]:
    """No diagnostic model/timing process can start before the same authorized reservation."""
    if (report.get("complete") is not False or report.get("error") or not _canonical_uuid(report.get("round_id"))
            or report.get("frozen_plan_sha256") != frozen_plan_sha256
            or gate.sha256(output / "plan.json") != frozen_plan_sha256):
        gate.fail("Phase diagnostics require the active reserved round and its unchanged frozen plan")
    with locked_ledger(ledger) as (state, _):
        current = round_policy(state, task)
        if current != planned_policy:
            gate.fail("Phase diagnostic authorization changed after the immutable plan")
        for digest in WORKLOAD_HASHES.values():
            rows = [row for row in state["tasks"].get(task, {}).get(digest, [])
                    if row.get("round_id") == report["round_id"]]
            if (len(rows) != 1 or Path(rows[0].get("output", "")).resolve() != output.resolve()
                    or (current["authorization"] is not None
                        and (rows[0].get("authorization_id") != current["authorization"]["id"]
                             or rows[0].get("maximum_rounds") != current["maximum_rounds"]))):
                gate.fail("Phase diagnostic workload is missing its exact authorized-round reservation")
    return ["--phase-only", "--phase-round-id", report["round_id"], "--phase-plan-sha256", frozen_plan_sha256]


def attach_phase_capture(reference: dict, diagnostic: dict, mode: str, budget: int,
                         round_id: str, frozen_plan_sha256: str) -> None:
    workers = budget if mode == "throughput" else 1
    expected = {f"{family}/{image}" for family in FAMILIES for image in WORKLOADS}
    if not isinstance(diagnostic, dict):
        gate.fail("Malformed separate phase capture")
    cases = diagnostic.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        gate.fail("Malformed separate phase cases")
    if (diagnostic.get("schema_version") != 1 or diagnostic.get("complete") is not True
            or diagnostic.get("engine") != "csharp" or diagnostic.get("phase_only") is not True
            or diagnostic.get("official_samples") is not False or diagnostic.get("mode") != mode
            or diagnostic.get("budget") != budget or diagnostic.get("workers") != workers
            or diagnostic.get("capacity_per_request") != (1 if mode == "throughput" else budget)
            or diagnostic.get("phase_round_id") != round_id or diagnostic.get("phase_plan_sha256") != frozen_plan_sha256
            or diagnostic.get("reshape_contract") != PHASE_RESHAPE_CONTRACT
            or diagnostic.get("initialization_contract") != PHASE_INITIALIZATION_CONTRACT
            or len(cases) != 9 or {case.get("id") for case in cases} != expected or set(reference) != expected):
        gate.fail("Separate phase process changed or omitted its round, plan, CPU partition, workload, or accounting contract")
    validate_runtime_identity(diagnostic)
    def number(value, positive=False):
        try:
            return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
        except OverflowError:
            return False
    exact = {name: 0.0 for name in ("tensor_atol", "tensor_rtol", "score_atol", "coordinate_atol")}
    validated = {}
    for case in cases:
        source = reference[case["id"]]
        if gate.compare_result(source["result"], case.get("result"), exact):
            gate.fail("Separate phase process changed public outputs")
        phase = case.get("phase", {})
        if not isinstance(phase, dict):
            gate.fail("Malformed full-OCR phase measurements")
        lines = len(source["result"]["lines"])
        stages = phase.get("reshape_stages", [])
        if not isinstance(stages, list) or any(not isinstance(stage, dict) for stage in stages):
            gate.fail("Malformed full-OCR reshape stages")
        expected_calls = {"det": workers, "cls": workers * lines, "rec": workers * lines}
        if (case.get("line_count") != lines or any(not number(case.get(field)) for field in ("model_load_ns", "compile_requests_ns"))
                or not isinstance(phase, dict) or not number(phase.get("first_run_wall_ns"), True)
                or phase.get("includes_reshape_and_instrumentation") is not True
                or phase.get("cold_contract") != "first-run-of-this-input-after-explicit-diagnostic-session-creation; earlier-shape-caches-may-exist"
                or not number(phase.get("diagnostic_session_creation_ns"))
                or type(phase.get("diagnostic_session_creation_calls")) is not int or phase["diagnostic_session_creation_calls"] != budget * 3
                or type(phase.get("reshape_ns")) is not int or phase["reshape_ns"] < 0
                or phase.get("reshape_time_kind") != "work-sum; may-overlap-across-workers; never-subtract-from-wall-time"
                or not isinstance(stages, list) or len(stages) != 3 or {stage.get("stage") for stage in stages} != set(expected_calls)
                or any(type(stage.get("calls")) is not int or stage["calls"] != expected_calls[stage["stage"]]
                       or type(stage.get("failed_calls")) is not int or stage["failed_calls"] != 0
                       or type(stage.get("nanoseconds")) is not int or stage["nanoseconds"] < 0 for stage in stages)
                or sum(stage["nanoseconds"] for stage in stages) != phase["reshape_ns"]
                or type(phase.get("post_run_owned_copy_ns")) is not int or phase["post_run_owned_copy_ns"] != 0
                or phase.get("ownership_contract") != PHASE_OWNED_CONTRACT
                or not isinstance(phase.get("ownership_explanation"), str) or not phase["ownership_explanation"]
                or not number(phase.get("post_run_text_publication_ns"))
                or phase.get("text_publication_contract") != "actual-PaddleOcrResult.Text-string.Join-once-per-worker-after-Run; separate-diagnostic-work-sum"
                or phase.get("official_timing_changed") is not False):
            gate.fail("Missing or invalid full-OCR reshape/ownership phase measurements")
        pipeline = phase.get("pinned_pipeline_stages", [])
        if (not isinstance(pipeline, list) or len(pipeline) != len(PHASE_STAGES)
                or any(not isinstance(stage, dict) for stage in pipeline)
                or [stage.get("stage") for stage in pipeline] != list(PHASE_STAGES)
                or any(not number(stage.get("nanoseconds")) or type(stage.get("calls")) is not int or stage["calls"] < 0 for stage in pipeline)):
            gate.fail("Separate phase process omitted the pinned pipeline stage evidence")
        validated[case["id"]] = {"diagnostic_model_load_ns": case["model_load_ns"],
            "diagnostic_compile_requests_ns": case["compile_requests_ns"], **phase}
    # Validate the entire capture before attaching any partial evidence.
    for id_, phase in validated.items():
        reference[id_]["phase_diagnostic"] = phase


def validate_capture(capture: dict, engine: str, mode: str, budget: int, samples: int, iterations: int, warmups: int = 2) -> dict:
    workers = budget if mode == "throughput" else 1
    expected = {f"{family}/{image}" for family in FAMILIES for image in WORKLOADS}
    cases = capture.get("cases", [])
    if (capture.get("schema_version") != 1 or capture.get("complete") is not True or capture.get("engine") != engine
            or capture.get("mode") != mode or capture.get("budget") != budget or capture.get("workers") != workers
            or capture.get("capacity_per_request") != (1 if mode == "throughput" else budget)
            or capture.get("owned_outputs") is not True or len(cases) != 9 or {case.get("id") for case in cases} != expected):
        gate.fail("Acceptance producer omitted cases or did not honor the owned-output CPU partition")
    if capture.get("warmups") != warmups or (mode != "validate" and (capture.get("sample_count") != samples or capture.get("iterations") != iterations)):
        gate.fail("Acceptance producer changed the fixed measurement schedule")
    if engine == "csharp":
        validate_runtime_identity(capture)
    if engine == "csharp" and mode == "validate":
        if capture.get("scratch_observer_enabled") is not True or capture.get("scratch_only") is not False:
            gate.fail("C# validation must qualify its separate scratch observer")
        validate_pool_probe(capture.get("array_pool_probe_support", {}), True)
    elif engine == "csharp" and (capture.get("scratch_observer_enabled") is not False
                                 or capture.get("scratch_only") is not False or capture.get("array_pool_probe_support") is not None):
        gate.fail("Official C# timing/allocation processes must never enable the scratch observer")
    for case in cases:
        if any(type(case.get(field)) not in (int, float) or not math.isfinite(case[field]) or case[field] < 0 for field in ("model_load_ns", "compile_requests_ns")):
            gate.fail("Missing loading/compilation phase evidence")
        shared = case.get("shared", {})
        if (any(type(shared.get(field)) is not int or shared[field] < 0 for field in ("normalized_weight_bytes", "packed_weight_bytes"))
                or shared["normalized_weight_bytes"] == 0 or "cold" not in case):
            gate.fail("Missing shared weights or cold-inference evidence")
        if mode == "validate":
            if case["cold"] is not None:
                gate.fail("Validation outputs must not pretend to measure cold inference")
        else:
            cold = case["cold"]
            profile = cold.get("profiles" if engine == "rust" else "profile") if isinstance(cold, dict) else None
            if (not isinstance(cold, dict) or type(cold.get("wall_ns")) not in (int, float)
                    or not math.isfinite(cold["wall_ns"]) or cold["wall_ns"] <= 0
                    or cold.get("includes_binding_and_reshape") is not True or cold.get("instrumented") is not True
                    or not isinstance(profile, list) or not profile or any(p is None for p in profile)):
                gate.fail("Missing instrumented cold/reshape stage evidence")
            if engine == "rust":
                validate_rust_phase_profiles(profile, workers)
        family, image = case["id"].split("/")
        gate.validate_case_payload({}, {"id": f"ocr/{family}/{image}", "result": case.get("result")})
        text = case["result"]["text"]
        lines = case["result"]["lines"]
        if (case.get("line_count") != len(lines) or text != "\n".join(line["text"] for line in lines)
                or case.get("max_recognition_tensor_width") != max((line["recognition_tensor_width"] for line in lines), default=0)):
            gate.fail("Acceptance summary differs from its actual public outputs")
        if not case.get("line_count"):
            gate.fail(f"Acceptance workload has no detected text: {case['id']}")
        if image == "dense-multilingual" and (case["line_count"] < 32 or not any(ord(c) > 127 for c in text) or not any(c.isascii() and c.isalpha() for c in text)):
            gate.fail(f"Dense workload no longer exercises at least 32 multilingual lines: {case['id']}")
        if image == "text-long" and (case["line_count"] < 3 or case.get("max_recognition_tensor_width", 0) < 1280):
            gate.fail(f"Long-line workload did not retain three lines and reach a recognition tensor width of 1280: {case['id']}")
        if mode != "validate":
            timings = case.get("samples", [])
            if len(timings) != samples or any(type(sample.get("wall_ns")) not in (int, float) or not math.isfinite(sample["wall_ns"]) or sample["wall_ns"] <= 0 or sample.get("requests") != workers * iterations for sample in timings):
                gate.fail("Missing or malformed official warmed timing samples")
            memory = case.get("memory", {})
            for phase in ("warmed", "after_samples", "repeated_shape"):
                if any(type(memory.get(phase, {}).get(field)) is not int or memory[phase][field] < 0 for field in ("activation_bytes", "workspace_bytes")):
                    gate.fail("Missing retained-memory evidence")
                state = memory[phase]
                if engine == "rust":
                    if (any(type(state.get(field)) is not int or state[field] < 0 for field in ("kernel_scratch_bytes", "nonkernel_workspace_bytes"))
                            or state["kernel_scratch_bytes"] + state["nonkernel_workspace_bytes"] != state["workspace_bytes"]):
                        gate.fail("Rust kernel scratch must partition retained workspace exactly")
                elif (state.get("kernel_scratch_bytes") is not None or state.get("nonkernel_workspace_bytes") is not None
                      or not state.get("kernel_scratch_contract") or not state.get("workspace_contract")
                      or state.get("workspace_measurement") != "observed_payloads_plus_collection_count_estimate"):
                    gate.fail("Unmeasured C# temporary kernel scratch must be explicit null with its accounting contract")
                if engine == "csharp" and (any(type(state.get(field)) is not int or state[field] < 0 for field in ("activation_usable_bytes", "activation_guard_bytes"))
                        or state["activation_usable_bytes"] + state["activation_guard_bytes"] != state["activation_bytes"]
                        or state["activation_guard_bytes"] % 256 != 0 or not state.get("activation_contract")):
                    gate.fail("C# native activation allocation must include its usable/guard byte partition")
            if (type(memory.get("peak_process_working_set_bytes")) is not int or memory["peak_process_working_set_bytes"] <= 0
                    or type(memory.get("allocation_probe", {}).get("allocated_bytes")) is not int or memory["allocation_probe"]["allocated_bytes"] < 0):
                gate.fail("Missing process-peak/allocation evidence")
        elif engine == "csharp":
            validate_pool_probe(case.get("numeric_pool_scratch_qualification", {}), False)
    return {case["id"]: case for case in cases}


def attach_pool_probe(reference: dict, diagnostic: dict, mode: str, budget: int) -> None:
    workers = budget if mode == "throughput" else 1
    cases = diagnostic.get("cases", [])
    if (diagnostic.get("schema_version") != 1 or diagnostic.get("complete") is not True
            or diagnostic.get("engine") != "csharp" or diagnostic.get("mode") != mode or diagnostic.get("budget") != budget
            or diagnostic.get("workers") != workers or diagnostic.get("capacity_per_request") != (1 if mode == "throughput" else budget)
            or diagnostic.get("scratch_only") is not True or diagnostic.get("scratch_observer_enabled") is not True
            or diagnostic.get("owned_outputs") is not True or len(cases) != 9 or {case.get("id") for case in cases} != set(reference)):
        gate.fail("Separate C# numeric scratch process changed its workload or CPU partition")
    validate_runtime_identity(diagnostic)
    validate_pool_probe(diagnostic.get("array_pool_probe_support", {}), True)
    exact = {"tensor_atol": 0., "tensor_rtol": 0., "score_atol": 0., "coordinate_atol": 0.}
    for case in cases:
        if gate.compare_result(reference[case["id"]]["result"], case.get("result"), exact):
            gate.fail("Separate C# numeric scratch process changed public results")
        probe = case.get("numeric_pool_scratch_probe", {})
        validate_pool_probe(probe, False)
        reference[case["id"]]["memory"]["numeric_pool_scratch_probe"] = probe
        reference[case["id"]]["memory"]["numeric_pool_probe_support"] = diagnostic["array_pool_probe_support"]


def compare_pair(reference: dict, rust: dict, tolerances: dict, mode: str) -> dict:
    rows = []
    if set(reference) != set(rust) or len(reference) != 9:
        gate.fail("Acceptance pair must contain the same nine fixed workloads")
    for id_ in sorted(reference):
        original, rewritten = reference[id_], rust[id_]
        errors = gate.compare_result(original["result"], rewritten["result"], tolerances)
        row = {"id": id_, "result_errors": errors, "passed": not errors}
        if "phase_diagnostic" in original:
            row["phase_accounting"] = {"reference_diagnostic": original["phase_diagnostic"],
                "rust_cold_profiles": rewritten["cold"]["profiles"],
                "contract": "Diagnostic call-work sums overlap wall phases; official owned latency/throughput samples are unchanged"}
        if mode != "validate":
            original_ns = statistics.median(sample["wall_ns"] for sample in original["samples"])
            rust_ns = statistics.median(sample["wall_ns"] for sample in rewritten["samples"])
            latency_ratio = rust_ns / original_ns
            throughput_ratio = (rewritten["samples"][0]["requests"] / rust_ns) / (original["samples"][0]["requests"] / original_ns)
            bound = latency_ratio <= 1.10 if mode == "latency" else throughput_ratio >= 0.90
            memory = []
            for engine, case in (("csharp", original), ("rust", rewritten)):
                state = case["memory"]
                if engine == "csharp":
                    validate_pool_probe(state.get("numeric_pool_scratch_probe", {}), False)
                fields = RUST_RETAINED_FIELDS if engine == "rust" else CSHARP_OBSERVED_FIELDS
                stable = all(state[phase][field] <= state["warmed"][field] for phase in ("after_samples", "repeated_shape") for field in fields)
                policy = RETAINED_MEMORY_POLICY[engine]
                deltas = {phase: {field: state[phase][field] - state["warmed"][field] for field in fields}
                          for phase in ("after_samples", "repeated_shape")}
                memory.append({**state, "engine": engine, "bounded_reuse": stable, "observed_stable": stable,
                               "growth_observed": not stable, "deltas_from_warmed": deltas,
                               "required_for_acceptance": policy["required_for_acceptance"],
                               "bounded_reuse_scope": policy["scope"],
                               **({"exact_total_bound_established": False} if engine == "csharp" else {})})
            row.update({"reference_median_ns": original_ns, "rust_median_ns": rust_ns, "rust_latency_ratio": latency_ratio,
                        "rust_throughput_ratio": throughput_ratio, "memory": memory,
                        "passed": row["passed"] and bound and all(item["bounded_reuse"] for item in memory if item["required_for_acceptance"])})
        rows.append(row)
    return {"passed": all(row["passed"] for row in rows), "retained_memory_policy": RETAINED_MEMORY_POLICY, "cases": rows}


def environment(mode: str, budget: int) -> dict:
    result = gate.reference_environment("auto", 1 if mode == "throughput" else budget)
    # Formal acceptance uses every normally available reference ISA, including
    # AVX-512 on capable hardware; the earlier AVX2 diagnostic oracle is separate.
    result.pop("DOTNET_EnableAVX512F", None)
    result.pop("DOTNET_EnableAVX512BW", None)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--correctness-bundle", type=Path, help="Passing complete 149-case bundle for the current library sources")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-id", default=os.environ.get("CODEX_THREAD_ID"), help="Stable task identity; never change it to reset the three-round cap")
    parser.add_argument("--ledger", type=Path, default=gate.ROOT / "target/cpu-performance-ledger.json")
    authorization_mode = parser.add_mutually_exclusive_group()
    authorization_mode.add_argument("--record-round-extension", action="store_true", help="Record the initial explicit two-round authorization without inference")
    authorization_mode.add_argument("--record-followup-round-extension", action="store_true", help="Record ONE explicit further round without inference or reservation")
    parser.add_argument("--additional-rounds", type=int, choices=(1, 2), help="Initial authorization requires 2; explicit followup requires 1")
    parser.add_argument("--authorization-id", help="Distinct UUID for the explicit new followup instruction")
    parser.add_argument("--prior-authorization-id", help="Latest prior authorization UUID; stale chains fail")
    parser.add_argument("--round-authorization", help="User instruction or explicit authorization context, retained in the task ledger")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--wait-for-idle", action="store_true", help="Wait with 30-second quiet gaps only before round reservation; post-reservation contention still fails immediately")
    args = parser.parse_args(argv)
    if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error("CPU acceptance requires Windows x64")
    if not args.task_id or args.warmups < 2 or args.samples < 5 or args.iterations < 1:
        parser.error("a stable task-id, >=2 warmups, >=5 samples and positive iterations are required")
    if args.record_round_extension:
        if args.additional_rounds != 2 or not args.round_authorization or args.authorization_id is not None or args.prior_authorization_id is not None:
            parser.error("--record-round-extension requires only --additional-rounds 2 and --round-authorization")
        policy = record_round_extension(args.ledger.resolve(), args.task_id, args.additional_rounds, args.round_authorization)
        print(json.dumps({"task_id": args.task_id, "round_policy": policy, "performance_round_reserved": False}, indent=2))
        return 0
    if args.record_followup_round_extension:
        if args.additional_rounds != 1 or not args.round_authorization or not args.authorization_id or not args.prior_authorization_id:
            parser.error("--record-followup-round-extension requires --additional-rounds 1, --authorization-id, --prior-authorization-id and --round-authorization")
        policy = record_followup_round_extension(args.ledger.resolve(), args.task_id, 1,
            args.authorization_id, args.prior_authorization_id, args.round_authorization)
        print(json.dumps({"task_id": args.task_id, "round_policy": policy, "performance_round_reserved": False}, indent=2))
        return 0
    if any(value is not None for value in (args.additional_rounds, args.round_authorization, args.authorization_id, args.prior_authorization_id)):
        parser.error("Record an explicit authorization separately before benchmarking")
    if any(value is None for value in (args.reference_dir, args.models_dir, args.correctness_bundle, args.output)):
        parser.error("Benchmarking requires --reference-dir, --models-dir, --correctness-bundle, and --output")
    planned_round_policy = read_round_policy(args.ledger.resolve(), args.task_id)
    out, models, source = args.output.resolve(), args.models_dir.resolve(), args.correctness_bundle.resolve()
    if out.exists():
        gate.fail("Acceptance output must be a fresh directory")
    lock = gate.read_json(gate.LOCK)
    gate.validate_inputs(args.reference_dir.resolve(), models, lock)
    sources = gate.rust_sources()
    correctness_prerequisite = validate_correctness_prerequisite(source, models, lock, sources)
    out.mkdir(parents=True)
    corpus = prepare_corpus(source / "corpus", out / "corpus")
    corpus_manifest_sha256 = gate.sha256(out / "corpus" / "corpus.json")
    dotnet, cargo = gate.require_tool("dotnet"), gate.require_tool("cargo")
    exporter_sources = gate.exporter_sources()
    gate.run([dotnet, "build", str(gate.ROOT / "tools/reference-exporter/ReferenceExporter.csproj"), "-c", "Release", f"-p:ReferenceRoot={args.reference_dir.resolve()}", "--nologo"])
    gate.run([cargo, "build", "--release", "--locked", "--example", "cpu_parity", "--features", "profiling"])
    if gate.rust_sources() != sources or gate.exporter_sources() != exporter_sources:
        gate.fail("Sources changed during acceptance compilation")
    configuration = correctness_prerequisite["compiler_build_configuration"]
    if compiler_build_configuration() != configuration:
        gate.fail("Rust compiler/build configuration changed during acceptance compilation")
    native_root = gate.ROOT / "tools/reference-exporter/bin/Release/net10.0"
    shutil.copytree(native_root, out / "binaries/csharp")
    executable = out / "binaries/rust/cpu_parity.exe"
    executable.parent.mkdir(parents=True)
    shutil.copy2(gate.ROOT / "target/release/examples/cpu_parity.exe", executable)
    gate.archive_sources(sources, out / "sources/rust")
    gate.archive_sources(exporter_sources, out / "sources/exporter")
    gate.archive_sources(gate.source_snapshot(gate.ROOT, [Path(__file__).resolve(), Path(gate.__file__).resolve(),
        Path(full_gate.__file__).resolve(), Path(cpu_load.__file__).resolve(), gate.LOCK]), out / "sources/driver")
    exporter = out / "binaries/csharp/Sdcb.SimdPaddleOCR.Tests.dll"
    frozen_jit_environment = jit_environment(environment("latency", 1))
    fingerprints = {"rust_sources": sources, "exporter_sources": exporter_sources,
                    "corpus_manifest_sha256": corpus_manifest_sha256,
                    "rust_executable_sha256": gate.sha256(executable),
                    "csharp_runtime_files": {path.name: gate.sha256(path) for path in exporter.parent.iterdir() if path.is_file()},
                    "rustc": configuration["rustc"],
                    "dotnet": gate.run([dotnet, "--info"], capture=True), "build_flags": configuration["build_flags"],
                    "driver_sha256": gate.sha256(Path(__file__)), "comparator_sha256": gate.sha256(Path(gate.__file__)),
                    "full_gate_helper_sha256": gate.sha256(Path(full_gate.__file__)),
                    "cpu_load_helper_sha256": gate.sha256(Path(cpu_load.__file__)),
                    "dotnet_jit_environment": frozen_jit_environment,
                    "dotnet_runtime_config": gate.read_json(exporter.parent / "Sdcb.SimdPaddleOCR.Tests.runtimeconfig.json"),
                    "host_name": platform.node(), "cpu": os.environ.get("PROCESSOR_IDENTIFIER", ""), "os": platform.platform()}
    plan = {"schema_version": 1, "budgets": BUDGETS, "workloads": corpus, "warmups": args.warmups, "samples": args.samples, "iterations": args.iterations,
            "task_id": args.task_id, "ledger": str(args.ledger.resolve()), "round_policy": planned_round_policy,
            "retained_memory_policy": RETAINED_MEMORY_POLICY,
            "jit_provenance_contract": "Recorded CLR runtime configuration and inherited JIT controls; null means unset, not disabled. Fixed warmups do not establish settled JIT tiers or PGO.",
            "host_load_policy": {"interval_seconds": 1, **cpu_load.POLICY},
            "pre_reservation_idle_wait": {"enabled": args.wait_for_idle, "quiet_gap_seconds": IDLE_QUIET_GAP_SECONDS,
                                          "phases": list(PRE_RESERVATION_PHASES), "post_reservation": "fail immediately; no benchmark repeats"},
            "owned_outputs": True, "latency_limit": 1.10, "throughput_floor": .90, "cpu_partition": "latency: one request with N compute workers; throughput: N persistent serial requests, aggregate capacity N",
            "diagnostic_schedule": "allocation-volume batch has no listener; after each paired timing capture, a separate C# process runs one warmed numeric ArrayPool batch per workload with EventListener, exact outputs checked; fixed stack tiles excluded; preflight validates event payloads/identities",
            "phase_diagnostic_schedule": "After the paired timing and scratch diagnostic, one separate C# full-OCR phase process per partition measures actual session reshapes and lazy Text publication. Same reserved round/plan; no additional warmups, official samples, clones, or performance repeats.",
            "phase_contracts": {"reshape": PHASE_RESHAPE_CONTRACT, "initialization": PHASE_INITIALIZATION_CONTRACT,
                                "owned_copy": PHASE_OWNED_CONTRACT},
            "reference_lock_sha256": gate.sha256(gate.LOCK), "correctness_run_sha256": correctness_prerequisite["run_sha256"],
            "correctness_prerequisite": correctness_prerequisite, "model_hashes": lock["models"], "fingerprints": fingerprints}
    gate.write_json(out / "plan.json", plan)
    frozen_plan_sha256 = gate.sha256(out / "plan.json")
    report = {"schema_version": 1, "complete": False, "passed": False, "comparisons": [], "host_load_checks": [],
              "frozen_plan_sha256": frozen_plan_sha256,
              "task_id": args.task_id, "round_policy": planned_round_policy, "retained_memory_policy": RETAINED_MEMORY_POLICY}
    gate.write_json(out / "report.json", report)

    def verify_frozen_inputs():
        if gate.sha256(out / "plan.json") != frozen_plan_sha256:
            gate.fail("Frozen acceptance plan changed")
        if jit_environment(environment("latency", 1)) != frozen_jit_environment:
            gate.fail("Inherited .NET JIT controls changed after the immutable plan")
        if gate.rust_sources() != sources or gate.exporter_sources() != exporter_sources:
            gate.fail("Sources changed after frozen acceptance build")
        if gate.build_flags() != configuration["build_flags"]:
            gate.fail("Rust build configuration changed after the immutable plan")
        if (gate.sha256(executable) != fingerprints["rust_executable_sha256"]
                or gate.sha256(Path(__file__)) != fingerprints["driver_sha256"]
                or gate.sha256(Path(gate.__file__)) != fingerprints["comparator_sha256"]
                or gate.sha256(Path(full_gate.__file__)) != fingerprints["full_gate_helper_sha256"]
                or gate.sha256(Path(cpu_load.__file__)) != fingerprints["cpu_load_helper_sha256"]
                or any(gate.sha256(exporter.parent / name) != digest for name, digest in fingerprints["csharp_runtime_files"].items())):
            gate.fail("Frozen acceptance executables or comparison tools changed")
        if any(gate.sha256(models / name) != digest for name, digest in lock["models"].items()):
            gate.fail("Pinned model or dictionary bytes changed during acceptance")
        verify_frozen_corpus(out / "corpus", fingerprints["corpus_manifest_sha256"])

    def produce(engine, mode, budget, scratch_only=False, phase_only=False):
        phase_arguments = []
        if phase_only:
            if engine != "csharp" or scratch_only:
                gate.fail("Full-OCR phase diagnostics require a separate C# process")
            phase_arguments = phase_diagnostic_arguments(args.ledger.resolve(), args.task_id, report, out,
                planned_round_policy, frozen_plan_sha256)
        verify_frozen_inputs()
        child_environment = environment(mode, budget)
        if jit_environment(child_environment) != frozen_jit_environment:
            gate.fail("Inherited .NET JIT controls changed after the immutable plan")
        suffix = "-phase" if phase_only else "-scratch" if scratch_only else ""
        destination = out / f"{engine}-{mode}-{budget}{suffix}.json"
        command = ([dotnet, str(exporter)] if engine == "csharp" else [str(executable)]) + ["--acceptance", "--models-dir", str(models), "--corpus", str(out / "corpus"), "--output", str(destination), "--mode", mode, "--threads", str(budget), "--warmups", str(args.warmups), "--samples", str(args.samples), "--iterations", str(args.iterations)]
        if scratch_only:
            command.append("--scratch-only")
        command.extend(phase_arguments)
        gate.run(command, env=child_environment)
        verify_frozen_inputs()
        capture = gate.read_json(destination)
        if engine == "csharp" and capture.get("runtime_identity", {}).get("jit_environment") != frozen_jit_environment:
            gate.fail("C# capture did not record the planned inherited JIT controls")
        return capture if scratch_only or phase_only else validate_capture(capture, engine, mode, budget, args.samples, args.iterations, args.warmups)

    idle_gate = IdleGate(report, out / "report.json", verify_frozen_inputs, args.wait_for_idle)

    def preflight_once():
        preflight = {engine: produce(engine, "validate", 1) for engine in ("csharp", "rust")}
        report["preflight"] = compare_pair(preflight["csharp"], preflight["rust"], lock["tolerances"], "validate")
        gate.write_json(out / "report.json", report)
        if not report["preflight"]["passed"]:
            gate.fail("New acceptance workload correctness failed before timing")
        return preflight

    def reserve_once():
        round_id = str(uuid.uuid4())
        reserve_round(args.ledger.resolve(), args.task_id, [image["sha256"] for image in corpus["images"]], round_id, out,
                      expected_policy=planned_round_policy)
        report["round_id"] = round_id

    try:
        preflight = idle_gate.preflight_and_reserve(preflight_once, reserve_once)
        exact = {name: 0.0 for name in lock["tolerances"]}
        for mode_index, mode in enumerate(("latency", "throughput")):
            for budget_index, budget in enumerate(BUDGETS):
                idle_gate.check(f"before-{mode}-{budget}")
                order = ("csharp", "rust") if (mode_index + budget_index) % 2 == 0 else ("rust", "csharp")
                captures = {engine: produce(engine, mode, budget) for engine in order}
                idle_gate.check(f"after-{mode}-{budget}-before-scratch-diagnostic")
                attach_pool_probe(captures["csharp"], produce("csharp", mode, budget, scratch_only=True), mode, budget)
                idle_gate.check(f"after-{mode}-{budget}-before-phase-diagnostic")
                attach_phase_capture(captures["csharp"], produce("csharp", mode, budget, phase_only=True), mode, budget,
                    report["round_id"], frozen_plan_sha256)
                gate.write_json(out / f"csharp-{mode}-{budget}-memory.json", {"cases": captures["csharp"]})
                paired = compare_pair(captures["csharp"], captures["rust"], lock["tolerances"], mode)
                deterministic = {engine: {id_: gate.compare_result(preflight[engine][id_]["result"], case["result"], exact)
                                          for id_, case in captures[engine].items()} for engine in order}
                paired.update({"mode": mode, "budget": budget, "order": order, "thread_determinism_errors": deterministic})
                paired["passed"] &= not any(errors for cases in deterministic.values() for errors in cases.values())
                report["comparisons"].append(paired)
                gate.write_json(out / "report.json", report)
        report["complete"] = True
        report["passed"] = all(item["passed"] for item in report["comparisons"])
        gate.write_json(out / "report.json", report)
        return 0 if report["passed"] else 1
    except Exception as error:
        report["error"] = str(error)
        gate.write_json(out / "report.json", report)
        raise


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
