"""Capture each OCR entry point in a dataset-specific directory. Run from repo root."""
import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/rapidocr-parity"
MODELS = ROOT.parent / "SimdPaddleOCR/models"
RAPID = Path("E:/RapidOCR/python")
FILES = {"tiny": ("det.onnx", "rec.onnx", "ppocr_keys.txt"),
         "small": ("small_det.onnx", "small_rec.onnx", "rec_keys.txt"),
         "medium": ("medium_det.onnx", "medium_rec.onnx", "rec_keys.txt")}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def prepare(root):
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        name = p.relative_to(root).as_posix()
        row = {"path": name, "sha256": digest(p), "input": str(p)}
        try:
            with Image.open(p) as im:
                row.update(mode=im.mode, size=im.size, exif_orientation=im.getexif().get(274))
                if p.suffix.lower() == ".webp":
                    png = OUT / "pixels" / (name + ".png")
                    png.parent.mkdir(parents=True, exist_ok=True)
                    im.save(png)
                    row.update(input=str(png), input_sha256=digest(png))
        except Exception as exc:
            row["read_error"] = repr(exc)
        rows.append(row)
    save(OUT / "inputs.json", rows)


def reference(preset):
    sys.path.insert(0, str(RAPID))
    import cv2
    import numpy
    import onnxruntime
    from omegaconf import OmegaConf
    from rapidocr import RapidOCR, ModelType, OCRVersion

    det, rec, keys = FILES[preset]
    params = {"Det.model_path": str(MODELS / det), "Rec.model_path": str(MODELS / rec),
              "Det.model_type": ModelType(preset), "Rec.model_type": ModelType(preset),
              "Cls.model_path": str(MODELS / "cls.onnx"), "Cls.ocr_version": OCRVersion.PPOCRV5,
              "Rec.rec_keys_path": str(MODELS / keys), "Global.log_level": "error",
              "EngineConfig.onnxruntime.intra_op_num_threads": 8,
              "EngineConfig.onnxruntime.inter_op_num_threads": 1}
    engine = RapidOCR(params=params)
    assert Path(inspect.getfile(type(engine))).resolve() == RAPID / "rapidocr/main.py"
    dictionary = (MODELS / keys).read_text(encoding="utf-8-sig").removesuffix("\n").split("\n")
    assert engine.text_rec.postprocess_op.character[1:] == dictionary + [" "]
    sessions = [s.session.session for s in (engine.text_det, engine.text_cls, engine.text_rec)]
    assert all(s.get_providers() == ["CPUExecutionProvider"] for s in sessions)
    assert list(sessions[1].get_inputs()[0].shape[1:]) == list(engine.text_cls.cls_image_shape)
    meta = {"source": inspect.getfile(type(engine)), "rapid_git": subprocess.check_output(
        ["git", "-C", str(RAPID), "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "ort": onnxruntime.__version__, "cv2": cv2.__version__,
        "numpy": numpy.__version__, "config": OmegaConf.to_yaml(engine.cfg),
        "classifier_shape": engine.text_cls.cls_image_shape,
        "labels": len(dictionary) + 2, "providers": [s.get_providers() for s in sessions],
        "files": {f: digest(MODELS / f) for f in (det, rec, keys, "cls.onnx")}}
    return engine, meta


def capture(kind, preset, indices, exe):
    rows = json.loads((OUT / "inputs.json").read_text("utf-8"))
    dest = OUT / kind / preset
    if kind == "reference":
        engine, meta = reference(preset)
    else:
        meta = {"binary": str(exe), "sha256": digest(exe), "threads": 8, "backend": "cpu",
                "files": {f: digest(MODELS / f) for f in (*FILES[preset], "cls.onnx")}}
    cfg = dest / "config.json"
    if cfg.exists() and json.loads(cfg.read_text("utf-8")) != meta:
        raise RuntimeError(f"settings changed; choose a fresh capture directory: {cfg}")
    save(cfg, meta)
    for i in indices if indices is not None else range(len(rows)):
        row = rows[i]
        path = dest / (row["path"] + ".json")
        if path.exists():
            assert json.loads(path.read_text("utf-8"))["source_sha256"] == row["sha256"]
            continue
        result = {"path": row["path"], "source_sha256": row["sha256"]}
        try:
            if "read_error" in row:
                raise ValueError(row["read_error"])
            assert digest(Path(row["input"])) == row.get("input_sha256", row["sha256"])
            if kind == "reference":
                r = engine(row["input"])
                if type(r).__name__ != "RapidOCROutput":
                    raise RuntimeError(f"incomplete output {type(r).__name__}")
                lines = [dict(text=t, box=b.tolist(), score=float(s))
                         for t, b, s in zip(r.txts or (), [] if r.boxes is None else r.boxes,
                                            r.scores or (), strict=True)]
            else:
                run = subprocess.run([str(exe), "--preset", preset, "--models-dir", str(MODELS),
                                      "--threads", "8", "--json", row["input"]],
                                     capture_output=True, timeout=180)
                if run.returncode:
                    raise RuntimeError(run.stderr.decode(errors="replace"))
                raw = json.loads(run.stdout)
                assert raw["text"] == "\n".join(l["text"] for l in raw["lines"])
                lines = [dict(text=l["text"], box=l["detection"]["points"], score=l["recognition_score"])
                         for l in raw["lines"]]
                result["raw"] = raw
            result.update(status="ok", lines=lines)
        except Exception as e:
            result.update(status="error", error=repr(e))
            print(i, repr(e), flush=True)
        save(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind")
    parser.add_argument("preset", nargs="?", choices=FILES)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--exe", type=Path, default=ROOT / "target/release/examples/ocr.exe")
    parser.add_argument("--images", type=Path, default=Path.home() / "Desktop/test-ocr-imgs")
    args = parser.parse_args()
    if args.kind == "prepare":
        prepare(args.images.resolve())
    else:
        capture(args.kind, args.preset, args.indices, args.exe.resolve())
