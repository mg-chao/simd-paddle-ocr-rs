"""Replay one captured input across the native CPU and ORT, isolating spatial reduction and the RGB stem."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from collect import FILES, MODELS, ROOT, save


def session(model, optimized_path=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if optimized_path:
        options.optimized_model_filepath = str(optimized_path)
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def subgraph(model, nodes, input_name, shape, output_name, output_shape):
    used = {name for n in nodes for name in n.input}
    graph = helper.make_graph(
        nodes, "operator-replay",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)],
        initializer=[t for t in model.graph.initializer if t.name in used])
    return helper.make_model(graph, ir_version=model.ir_version, opset_imports=model.opset_import)


def run_native(exe, model, tensor, directory, shape):
    path = directory / "native-output.f32"
    subprocess.run([str(exe.resolve()), str(model), str(tensor), str(path),
                    "x".join(map(str, shape))], check=True, timeout=180)
    return np.fromfile(path, "<f4")


def delta(a, b):
    diff = np.abs(a - b)
    return dict(unequal=int(np.count_nonzero(diff)), maximum=float(diff.max()), mae=float(diff.mean()))


def task_delta(x, ref, detection):
    result = delta(x, ref)
    result["threshold_flips" if detection else "ctc_winner_changes"] = int(np.count_nonzero(
        (x > .3) != (ref > .3) if detection else x.argmax(-1) != ref.argmax(-1)))
    return result


def packet_mean(row, offset):
    # The logical NCHW channel begins at offset F32 elements from an aligned base.
    prefix = (-offset) % 4
    count = max(0, len(row) - prefix) // 4
    if not count:
        return row.cumsum()[-1] / np.float32(len(row))
    end = prefix + count * 4
    groups = row[prefix:end].reshape(count, 4)
    if count == 1:
        v = groups[0]
    else:
        pairs = count // 2
        even, odd = groups[:pairs * 2].reshape(pairs, 2, 4).cumsum(axis=0)[-1]
        v = even + odd
        if count % 2:
            v = v + groups[-1]
    total = (v[0] + v[2]) + (v[1] + v[3])
    for tail in (row[:prefix], row[end:]):
        for value in tail:
            total += value
    return total / np.float32(len(row))


def stem_arithmetic(model, nodes, x):
    conv = nodes[0]
    constants = {t.name: t for t in model.graph.initializer}
    w = numpy_helper.to_array(constants[conv.input[1]])
    bias_name = conv.input[2] if len(conv.input) == 3 else nodes[1].input[1]
    bias = numpy_helper.to_array(constants[bias_name]).reshape(-1, 1, 1)
    attrs = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
    assert tuple(w.shape[1:]) == (3, 3, 3) and attrs["strides"] == [2, 2] and attrs["pads"] == [1]*4
    h, width = x.shape[2:]
    padded = np.pad(x[0], ((0, 0), (1, 1), (1, 1)))
    shape = (len(w), (h + 1)//2, (width + 1)//2)
    has_conv_bias = len(conv.input) == 3
    old = np.broadcast_to(bias, shape).copy() if has_conv_bias else np.zeros(shape, dtype="f")
    merged = np.zeros_like(old)
    for c in range(3):
        partial = np.zeros_like(old)
        for y in range(3):
            for j in range(3):
                sample = padded[c, y:y+h:2, j:j+width:2]
                # f64 has enough precision for these measured F32 products and sums.
                product = w[:, c, y, j, None, None].astype("d") * sample
                old = (product + old).astype("f")
                partial = (product + partial).astype("f")
        merged += partial
    return np.maximum(old if has_conv_bias else old + bias, 0), np.maximum(merged + bias, 0)


def main(preset, capture, directory, rec_line, exe, before):
    directory.mkdir(parents=True, exist_ok=True)
    detection = rec_line is None
    if detection:
        boxes = json.loads((capture / "boxes.json").read_text("utf-8"))
        tensor = capture / "det-input.f32"
        shape = (1, 3, boxes["resized_height"], boxes["resized_width"])
    else:
        tensor = capture / f"rec-{rec_line}.f32"
        shape = (1, 3, 48, -1)
    x = np.fromfile(tensor, "<f4").reshape(shape)
    model_path = MODELS / FILES[preset][0 if detection else 1]
    model = onnx.load(model_path)
    name = model.graph.input[0].name
    output = model.graph.output[0].name
    saved = directory / "optimized.onnx"
    ref = session(model, saved).run(None, {name: x})[0]
    physical = onnx.load(saved)
    mean = next(n for n in physical.graph.node if n.op_type == "ReduceMean"
                and any(a.name == "axes" and list(a.ints) == [2, 3] for a in n.attribute))
    extra = [mean.input[0], mean.output[0]]
    for item in extra:
        physical.graph.output.append(helper.make_tensor_value_info(item, TensorProto.FLOAT, None))
    check, a, true_mean = session(physical).run([output, *extra], {name: x})
    assert np.array_equal(check, ref)
    rows = a.reshape(-1, a.shape[2] * a.shape[3])
    linear = rows.cumsum(axis=1)[:, -1] / np.float32(rows.shape[1])
    packets = np.array([packet_mean(row, i * len(row)) for i, row in enumerate(rows)])
    results = dict(preset=preset, stage="detection" if detection else "recognition", input=list(x.shape),
                   ort_build=ort.get_build_info(), mean_node=mean.name, mean_shape=list(a.shape),
                   old_serial_mean=delta(linear, true_mean.ravel()), packet_mean=delta(packets, true_mean.ravel()))
    # An identity 1x1 convolution lets the public executor enter the actual
    # channels-last segment without changing the supplied activation values.
    channels = a.shape[1]
    identity = numpy_helper.from_array(np.eye(channels, dtype="f").reshape(channels, channels, 1, 1), "eye")
    conv = helper.make_node("Conv", ["input", "eye"], ["packed"], kernel_shape=[1, 1])
    reduce = helper.make_node("ReduceMean", ["packed"], ["result"], axes=[2, 3], keepdims=1)
    isolated = subgraph(model, [conv, reduce], "input", list(a.shape), "result", list(true_mean.shape))
    isolated.graph.initializer.append(identity)
    mean_path, activation = directory / "mean.onnx", directory / "mean-input.f32"
    onnx.save(isolated, mean_path)
    a.tofile(activation)
    expected = true_mean.ravel()
    results["isolated_native_after"] = delta(run_native(exe, mean_path, activation, directory, a.shape), expected)
    if before:
        previous_mean = run_native(before, mean_path, activation, directory, a.shape)
        results["isolated_native_before"] = delta(previous_mean, expected)
        results["old_native_equals_serial"] = bool(np.array_equal(previous_mean, linear))
    native = run_native(exe, model_path, tensor, directory, x.shape).reshape(ref.shape)
    results["native_after"] = task_delta(native, ref, detection)
    if before:
        old = run_native(before, model_path, tensor, directory, x.shape).reshape(ref.shape)
        results["native_before"] = task_delta(old, ref, detection)

    cut = next(i for i, n in enumerate(model.graph.node) if n.op_type == "Relu")
    nodes = list(model.graph.node[:cut+1])
    mid = nodes[-1].output[0]
    stem = subgraph(model, nodes, name, list(x.shape), mid, None)
    expected_mid = session(stem).run(None, {name: x})[0]
    stem.graph.output[0].CopyFrom(helper.make_tensor_value_info(mid, TensorProto.FLOAT, list(expected_mid.shape)))
    stem_path, tail_path = directory / "stem.onnx", directory / "suffix.onnx"
    onnx.save(stem, stem_path)
    suffix = subgraph(model, list(model.graph.node[cut+1:]), mid, list(expected_mid.shape), output, list(ref.shape))
    onnx.save(suffix, tail_path)
    native_mid = run_native(exe, stem_path, tensor, directory, x.shape).reshape(expected_mid.shape)
    # Compare an interior patch. A one-pixel input margin preserves the first layer's receptive field.
    iy, ix = (x.shape[2]//4)*2, (x.shape[3]//4)*2
    patch = x[:, :, iy:iy+48, ix:ix+64]
    old_order, ort_order = stem_arithmetic(model, nodes, patch)
    h, w = old_order.shape[1:]
    area = (0, slice(None), slice(iy//2+1, iy//2+h-1), slice(ix//2+1, ix//2+w-1))
    results["native_stem_vs_node_order"] = delta(native_mid[area], old_order[:, 1:-1, 1:-1])
    results["ort_stem_vs_per_channel"] = delta(expected_mid[area], ort_order[:, 1:-1, 1:-1])
    results["two_stems_on_patch"] = delta(native_mid[area], expected_mid[area])
    for label, values in [("native_stem_control", native_mid), ("ort_stem_swap", expected_mid)]:
        path = directory / "stem-input-to-suffix.f32"
        values.tofile(path)
        y = run_native(exe, tail_path, path, directory, values.shape).reshape(ref.shape)
        results[label] = task_delta(y, ref, detection)
        if label == "native_stem_control":
            results["cut_preserves_native_output"] = bool(np.array_equal(y, native))
    save(directory / "operators.json", results)
    print(json.dumps(results))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preset", choices=FILES)
    p.add_argument("capture", type=Path)
    p.add_argument("directory", type=Path)
    p.add_argument("--rec-line", type=int)
    p.add_argument("--exe", type=Path, default=ROOT / "target/release/examples/rapidocr_tensor_probe")
    p.add_argument("--before", type=Path)
    args = p.parse_args()
    main(args.preset, args.capture, args.directory, args.rec_line, args.exe, args.before)
