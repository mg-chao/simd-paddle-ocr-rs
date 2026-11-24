"""Capture the public resident GPU OCR path against the frozen source hashes."""
import argparse
import json
from pathlib import Path
import subprocess

from collect import FILES, MODELS, OUT, ROOT, digest, save


def run(args):
    rows = json.loads((OUT / "inputs.json").read_text("utf-8"))
    binary = args.exe.resolve(strict=True)
    dest = OUT / args.capture / args.preset
    reference = json.loads((OUT / "reference" / args.preset / "config.json").read_text("utf-8"))
    files = {name: digest(MODELS / name) for name in (*FILES[args.preset], "cls.onnx")}
    assert files == reference["files"]
    meta = dict(executable_sha256=digest(binary), files=files, target="gpu", precision="f32",
                matrix="f32", threads=8, recognition_wave_size=1, classification_batch_size=4)
    config = dest / "config.json"
    if config.exists() and json.loads(config.read_text("utf-8")) != meta:
        raise ValueError(f"Different configuration in {dest}")
    save(config, meta)
    pending = []
    for index in args.indices if args.indices is not None else range(len(rows)):
        row = rows[index]
        assert digest(Path(row["input"])) == row.get("input_sha256", row["sha256"])
        out = dest / (row["path"] + ".json")
        if out.exists():
            assert json.loads(out.read_text("utf-8"))["source_sha256"] == row["sha256"]
        else:
            pending.append(json.dumps(row, ensure_ascii=True))
    if pending:
        command = [str(binary), args.preset, str(dest)]
        if args.timing:
            command.append("--timing")
        subprocess.run(command, input="\n".join(pending)+"\n", encoding="utf-8", check=True,
                       cwd=ROOT, timeout=60 if args.timing else 1800)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("preset", choices=FILES)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.timing and args.indices is None:
        parser.error("--timing requires explicit --indices")
    run(args)
