"""Replay a reference recognition row on the GPU without changing the frozen OCR results."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from collect import FILES, MODELS, OUT, ROOT, reference, save


def main(args):
    rows = json.loads((OUT / "inputs.json").read_text("utf-8"))
    row = rows[args.index]

    def load_line(group):
        path = OUT / group / args.preset / (row["path"] + ".json")
        result = json.loads(path.read_text("utf-8"))
        assert result["status"] == "ok" and result["source_sha256"] == row["sha256"]
        return result["lines"][args.line]

    expected, actual = load_line("reference"), load_line(args.capture)
    engine, _ = reference(args.preset)
    rec = engine.text_rec
    session = rec.session
    matches = []

    def trace(batch):
        probs = session(batch)
        for j, (text, score) in enumerate(rec.postprocess_op(probs)[0]):
            if text == expected["text"] and abs(score - expected["score"]) < 1e-7:
                matches.append((batch[j:j + 1].copy(), probs[j:j + 1].copy()))
        return probs

    rec.session = trace
    try:
        engine(row["input"])
    finally:
        rec.session = session
    if len(matches) != 1:
        raise ValueError(f"Expected one reference row with the selected text and score, found {len(matches)}")
    tensor, ref = matches[0]
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    tensor.tofile(directory / "input.f32")
    ref.tofile(directory / "python.f32")
    subprocess.run([args.probe.resolve(strict=True), MODELS / FILES[args.preset][1],
                    directory / "input.f32", directory / "gpu.f32", "x".join(map(str, tensor.shape)),
                    "--gpu"], cwd=ROOT, timeout=120, check=True)
    gpu = np.fromfile(directory / "gpu.f32", "<f4").reshape(ref.shape)
    result = dict(path=row["path"], preset=args.preset, line=args.line, shape=tensor.shape,
                  max_abs=float(np.abs(ref - gpu).max()),
                  winner_changes=int(np.count_nonzero(ref.argmax(-1) != gpu.argmax(-1))),
                  reference=expected["score"], resident=actual["score"],
                  same_input_python_decode=rec.postprocess_op(ref)[0],
                  same_input_gpu_decode=rec.postprocess_op(gpu)[0])
    save(directory / "replay.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=FILES)
    parser.add_argument("index", type=int)
    parser.add_argument("line", type=int)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--capture", default="gpu-aligned")
    parser.add_argument("--probe", type=Path, default=ROOT / "target/release/examples/rapidocr_tensor_probe.exe")
    main(parser.parse_args())
