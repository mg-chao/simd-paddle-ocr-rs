"""Local correctness matrix for the plan/request API; never runs benchmarks.

Run on the physical development machine. Hardware checks are opt-in and reject
software adapters through the tests' existing device checks. Logs and a compact
manifest are written under target/architecture-validation.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = [
    "gpu_integration_tests",
    "gpu_hardware_resident_storage_has_no_dense_transfers_and_reshape_rolls_back",
    "gpu_hardware_strict_auto_callbacks_and_concurrent_sessions",
    "gpu_hardware_reshape_cache_compaction_rollback_and_trace_reentry",
    "gpu_hardware_full_precision_variant_shares_context_and_isolates_weights",
    "gpu_resident_pending_readbacks_hold_admission_and_reservations",
    "gpu_resident_projection_ctc_contract",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--hardware", nargs="*", choices=["vulkan", "dx12", "metal"], default=[])
    parser.add_argument("--report", type=Path, default=ROOT / "target/architecture-validation")
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    results = []

    def check(name, command, env=None, timeout=900):
        started = time.monotonic()
        with subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                              encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              start_new_session=os.name != "nt") as process:
            try:
                output, _ = process.communicate(timeout=timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                # Cargo launches children; stopping only Cargo leaves the test or
                # compiler alive and can keep its output pipe open indefinitely.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                   check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                output = f"TIMEOUT after {timeout}s\n{output}"
                code = 124
        (args.report / f"{name}.log").write_text(output, encoding="utf-8")
        summary = [line for line in output.splitlines() if line.startswith("test result:")]
        results.append(dict(name=name, command=command, returncode=code,
                            elapsed_seconds=round(time.monotonic() - started, 3), summary=summary))
        (args.report / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(name, "PASS" if code == 0 else "FAIL", *summary, flush=True)
        if code:
            print(output[-8000:], flush=True)
            raise SystemExit(code)

    check("format", ["cargo", "fmt", "--all", "--check"])
    check("clippy-all", ["cargo", "clippy", "--locked", "--all-targets", "--all-features", "--", "-D", "warnings"])
    check("clippy-serial", ["cargo", "clippy", "--locked", "--all-targets", "--no-default-features", "--", "-D", "warnings"])
    profile = ["--release"] if args.release else []
    for parallel, gpu, profiling in itertools.product([True, False], repeat=3):
        features = [name for name, enabled in [("gpu", gpu), ("profiling", profiling)] if enabled]
        flags = ([] if parallel else ["--no-default-features"]) + (["--features", ",".join(features)] if features else [])
        name = "-".join(["default" if parallel else "serial", *features])
        check(name, ["cargo", "test", "--locked", *profile, *flags, "--lib", "--", "--test-threads=1"])
    for name, flags in [("all", ["--all-features"]), ("serial", ["--no-default-features"])]:
        check(f"examples-{name}", ["cargo", "build", "--locked", *profile, *flags, "--examples"])
        check(f"doctests-{name}", ["cargo", "test", "--locked", *profile, *flags, "--doc"])
    for backend in args.hardware:
        env = dict(os.environ, SIMD_OCR_GPU_BACKEND=backend)
        for test in HARDWARE:
            check(f"{backend}-{test}", ["cargo", "test", "--locked", *profile,
                  "--all-features", "--lib", test, "--", "--ignored", "--test-threads=1"], env, timeout=120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
