"""Compare one Rust trace to RapidOCR and replay its detector map through Python DB."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from collect import OUT, ROOT, reference, save


def difference(a, b):
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    delta = np.abs(a.astype(np.float32) - b)
    return {"shape": list(a.shape), "changed": int(np.count_nonzero(delta)),
            "max_abs": float(delta.max(initial=0)), "mae": float(delta.mean())}


def replay_rec(engine, preset, path, directory, index):
    probe = ROOT / "target/release/examples/rapidocr_parity_probe"
    subprocess.run([str(probe), preset, path, str(directory), "--replay-rec", str(index)], check=True)
    tensor = np.fromfile(directory / f"rec-{index}.f32", "<f4").reshape(1, 3, 48, -1)
    ort = engine.text_rec.session(tensor)
    native = np.fromfile(directory / f"optimized-rec-{index}.f32", "<f4").reshape(ort.shape)
    decode = engine.text_rec.postprocess_op
    a, b = ort.argmax(-1)[0], native.argmax(-1)[0]
    changed = []
    for step in np.flatnonzero(a != b):
        i, j = int(a[step]), int(b[step])
        changed.append(dict(step=int(step), ort_class=i, native_class=j,
                            ort_label=decode.character[i], native_label=decode.character[j],
                            ort_probabilities=[float(ort[0, step, k]) for k in (i, j)],
                            native_probabilities=[float(native[0, step, k]) for k in (i, j)]))
    save(directory / "rec-replay.json", dict(line=index, shape=list(tensor.shape),
         ort=decode(ort)[0][0], optimized_native=decode(native)[0][0],
         max_probability_error=float(np.max(np.abs(ort - native))), changed_winners=changed))


def main(preset, index, directory, downstream_only=False, map_only=False, rec_line=None):
    row = json.loads((OUT / "inputs.json").read_text("utf-8"))[index]
    engine, _ = reference(preset)
    if rec_line is not None:
        return replay_rec(engine, preset, row["input"], directory, rec_line)
    image = engine.load_img(row["input"])
    page, ops = engine.preprocess_img(image)
    from rapidocr.utils.process_img import apply_vertical_padding
    page, _ = apply_vertical_padding(page, ops, engine.width_height_ratio, engine.min_height)
    if map_only:
        # Hold Python pixels/preprocessing fixed and vary only the detector executor.
        directory.mkdir(parents=True, exist_ok=True)
        det = engine.text_det
        tensor = det.get_preprocess(max(page.shape[:2]))(page)
        tensor.astype("<f4", copy=False).tofile(directory / "det-input.f32")
        save(directory / "boxes.json", {"resized_height": tensor.shape[2], "resized_width": tensor.shape[3]})
        probe = ROOT / "target/release/examples/rapidocr_parity_probe"
        subprocess.run([str(probe), preset, row["input"], str(directory), "--replay-det"], check=True)
        ref = det.session(tensor)
        prod = np.fromfile(directory / "optimized-map.f32", "<f4").reshape(ref.shape)
        post = lambda pred: det.sorted_boxes(det.postprocess_op(pred, page.shape[:2])[0])
        ref_boxes, prod_boxes = post(ref), post(prod)
        save(directory / "production-map-boxes.json", prod_boxes.tolist())
        save(directory / "map-replay.json", {"image": row["path"], "production_det_map": difference(ref, prod),
             "production_mask_disagreements": int(np.count_nonzero((ref > .3) != (prod > .3))),
             "production_map_python_db": difference(ref_boxes, prod_boxes)})
    if downstream_only or map_only:
        prod_boxes = np.array(json.loads((directory / "production-map-boxes.json").read_text("utf-8")), dtype=np.float32)
        lines = []
        if len(prod_boxes):
            crops, _ = engine.cls_and_rotate(engine.crop_text_regions(page, prod_boxes))
            rec = engine.recognize_txt(crops)
            lines = [dict(text=t, score=s) for t, s in zip(rec.txts, rec.scores, strict=True)
                     if t.strip() and s >= engine.text_score]
        save(directory / "downstream-replay.json", {"path": row["path"], "lines": lines})
        return
    rgb = np.fromfile(directory / "source.rgb", np.uint8).reshape(image.shape)
    det = engine.text_det
    tensor = det.get_preprocess(max(page.shape[:2]))(page)
    pred = det.session(tensor)
    rust_input = np.fromfile(directory / "det-input.f32", "<f4").reshape(tensor.shape)
    rust_map = np.fromfile(directory / "det-map.f32", "<f4").reshape(pred.shape)
    post = lambda p: det.sorted_boxes(det.postprocess_op(p, page.shape[:2])[0])
    python_boxes = post(pred)
    replay = post(rust_map)
    rust = json.loads((directory / "boxes.json").read_text("utf-8"))["boxes"]
    rust_boxes = np.array([b["points"] for b in rust], dtype=np.float32)
    result = {"image": row["path"], "source_pixels": difference(image, rgb[:, :, ::-1]),
              "det_input": difference(tensor, rust_input), "det_map": difference(pred, rust_map),
              "mask_disagreements": int(np.sum((pred > 0.3) != (rust_map > 0.3))),
              "same_map_db": difference(replay, rust_boxes),
              "different_map_python_db": difference(python_boxes, replay), "lines": []}
    optimized = directory / "optimized-map.f32"
    if optimized.exists():
        prod = np.fromfile(optimized, "<f4").reshape(pred.shape)
        prod_boxes = post(prod)
        result["production_det_map"] = difference(pred, prod)
        result["production_mask_disagreements"] = int(np.sum((pred > .3) != (prod > .3)))
        result["production_map_python_db"] = difference(python_boxes, prod_boxes)
        save(directory / "production-map-boxes.json", prod_boxes.tolist())
    # Use identical Rust quads for crop/rotation comparisons downstream of DB.
    if rust:
        raw = engine.crop_text_regions(page, rust_boxes)
        crops, _ = engine.cls_and_rotate(raw)
        ratios = [c.shape[1] / c.shape[0] for c in crops]
        order = np.argsort(ratios)
        tensors = {}
        for start in range(0, len(order), 6):
            batch = order[start:start+6]
            ratio = max(320/48, max(ratios[i] for i in batch))
            for i in batch:
                tensors[i] = engine.text_rec.resize_norm_img(crops[i], ratio)
        for i, crop in enumerate(crops):
            file, = directory.glob(f"crop-{i}-*.bgr")
            w, h = map(int, file.stem.split("-")[-1].split("x"))
            rust_crop = np.fromfile(file, np.uint8).reshape(h, w, 3)
            actual = np.fromfile(directory / f"rec-{i}.f32", "<f4").reshape(3, 48, -1)
            detail = {"index": i, "fixed_quad_crop": difference(crop, rust_crop),
                      "rec_input": difference(tensors[i], actual)}
            decoded = directory / f"decoded-{i}.json"
            if decoded.exists():
                rust_result = json.loads(decoded.read_text("utf-8"))
                pred = engine.text_rec.session(actual[None, ...])
                ort_result = engine.text_rec.postprocess_op(pred)[0][0]
                detail["same_tensor_rec"] = {"rust": rust_result, "ort": list(ort_result)}
            result["lines"].append(detail)
    save(directory / "diagnosis.json", result)
    print({k: v for k, v in result.items() if k != "lines"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preset")
    p.add_argument("index", type=int)
    p.add_argument("directory", type=Path)
    p.add_argument("--downstream-only", action="store_true")
    p.add_argument("--map-only", action="store_true", help="replay only the native detector on Python input, then run Python downstream")
    p.add_argument("--rec-line", type=int, help="compare CTC winners of both optimized graphs on a saved tensor")
    args = p.parse_args()
    main(args.preset, args.index, args.directory, args.downstream_only, args.map_only, args.rec_line)
