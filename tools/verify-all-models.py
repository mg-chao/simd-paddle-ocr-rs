"""Run native all-node REC/OCR gates and retain immutable executable/source proofs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def source_identity(root):
    files = sorted(set(root.glob("src/**/*.rs")) | set(root.glob("tests/**/*.rs"))
                   | {root / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")})
    return {path.relative_to(root).as_posix(): digest(path) for path in files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference", action="append", required=True, help="family=restored-bundle-path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=("compact", "expanded"), required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent.parent
    start = source_identity(root)
    original_binary = digest(binary)
    references, comparisons = {}, {}
    for specification in args.reference:
        family, value = specification.split("=", 1)
        if family not in ("tiny", "small", "medium") or family in references:
            raise ValueError("Invalid or duplicate family")
        reference = Path(value).resolve(strict=True)
        manifest_path = reference / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if manifest["contract"]["model"] != family:
            raise ValueError("Reference family differs")
        references[family] = {"path": str(reference), "manifest_sha256": digest(manifest_path),
                              "models": manifest["models"], "case_count": len(manifest["cases"]),
                              "artifact_count": sum(len(case["artifacts"]) for case in manifest["cases"])}
        for stage in ("rec", "ocr"):
            candidate = out / family / stage
            candidate.parent.mkdir(parents=True, exist_ok=True)
            for operation, tail in (("run", ["--out", str(candidate)]), ("compare", [str(candidate)])):
                command = [str(binary), f"{operation}-{stage}", str(reference), *tail, "--json"]
                name = f"{family}-{stage}-{operation}"
                (out / f"{name}.command.json").write_text(json.dumps(command) + "\n", encoding="utf-8")
                with (out / f"{name}.json").open("w", encoding="utf-8") as stdout, (out / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr)
                if completed.returncode:
                    raise RuntimeError(f"{name} failed ({completed.returncode}); see retained logs")
                if operation == "compare":
                    report = json.loads((out / f"{name}.json").read_bytes())
                    if report["first_difference"] is not None:
                        raise RuntimeError("Comparator reported a mismatch")
                    comparisons[f"{family}-{stage}"] = report
                print(f"PASS {name}", flush=True)
    completion = source_identity(root)
    if start != completion or digest(binary) != original_binary:
        raise RuntimeError("Source or executable changed during validation")
    for identity in references.values():
        if digest(Path(identity["path"]) / "manifest.json") != identity["manifest_sha256"]:
            raise RuntimeError("Reference changed during validation")
    proof = {"schema_version": 1, "status": "all_native_rec_ocr_gates_passed", "scope": args.scope,
             "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "platform": platform.platform(), "architecture": platform.machine(),
             "executable": {"path": str(binary), "sha256": original_binary},
             "source_files": completion, "source_changed_during_validation": False,
             "references": references, "comparisons": comparisons,
             "logs": {path.name: digest(path) for path in sorted(out.glob("*.json"))}}
    (out / "proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
