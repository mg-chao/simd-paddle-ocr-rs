"""Diagnose inference and DB independently using identical detector tensors and probability maps."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from collect import FILES, MODELS, OUT, ROOT, reference, save


def main(preset, index, probe, directory, mixed=False):
    row = json.loads((OUT / "inputs.json").read_text("utf-8"))[index]
    engine, meta = reference(preset)
    from rapidocr.utils.process_img import apply_vertical_padding
    page, ops = engine.preprocess_img(engine.load_img(row["input"]))
    page, _ = apply_vertical_padding(page, ops, engine.width_height_ratio, engine.min_height)
    det = engine.text_det
    x = det.get_preprocess(max(page.shape[:2]))(page)
    directory.mkdir(parents=True, exist_ok=True)
    x.astype("<f4", copy=False).tofile(directory / "input.f32")
    ref = det.session(x)
    ref.astype("<f4", copy=False).tofile(directory / "python.f32")
    model = MODELS / FILES[preset][0]
    def run(src, dst, shape, mode):
        subprocess.run([probe, model, directory / src, directory / dst, "x".join(map(str,shape)), mode],
                       check=True, cwd=ROOT, timeout=120)
    mode = "--gpu-mixed" if mixed else "--gpu"
    run("input.f32", "gpu.f32", x.shape, mode)
    gpu = np.fromfile(directory / "gpu.f32", "<f4").reshape(ref.shape)
    boxes = {}
    for name, pred in [("python", ref), ("gpu", gpu)]:
        ref_boxes = det.sorted_boxes(det.postprocess_op(pred, page.shape[:2])[0])
        boxes[name + "_map_python_db"] = np.asarray(ref_boxes).reshape(-1,4,2).tolist()
        run(name+".f32", name+"-map-gpu-db.json", (*ref.shape[-2:], *page.shape[:2]), mode + "-db")
        native = json.loads((directory / (name+"-map-gpu-db.json")).read_text("utf-8"))
        boxes[name + "_map_gpu_db"] = [b["points"] for b in native]
    save(directory / "boxes.json", boxes)
    def compare(key1, key2):
        a,b = (boxes[key] for key in (key1,key2))
        return dict(counts=[len(a),len(b)], identical=a==b,
                    shared_quads=len({tuple(map(tuple,q)) for q in a} & {tuple(map(tuple,q)) for q in b}))
    result = dict(path=row["path"], preset=preset, precision="mixed-f16/f16" if mixed else "f32/f32", shape=x.shape, models=meta["files"],
                  map_max_error=float(np.max(np.abs(gpu-ref))),
                  mask_disagreements=int(np.count_nonzero((gpu > .3) != (ref > .3))),
                  inference=compare("python_map_python_db", "gpu_map_python_db"),
                  geometry=compare("gpu_map_python_db", "gpu_map_gpu_db"),
                  geometry_on_reference=compare("python_map_python_db", "python_map_gpu_db"))
    save(directory / "replay.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=FILES)
    parser.add_argument("index", type=int)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--probe", type=Path, default=ROOT / "target/release/examples/rapidocr_tensor_probe.exe")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    main(args.preset, args.index, args.probe.resolve(strict=True), args.directory.resolve(), args.mixed)
