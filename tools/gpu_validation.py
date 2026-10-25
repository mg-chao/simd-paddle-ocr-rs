"""Local developer validation. Stdlib only; never downloads models or runs benchmarks.

Inventory preserves original ONNX tensor names and attributes. --boundary creates
a pruned model whose graph output is that named semantic boundary, so neither
Rust nor C# node numbering is used to align tensors after normalization/fusion.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = "Metal" if sys.platform == "darwin" else "Vulkan"
MODELS = ("cls", "det", "small_det", "medium_det", "rec", "small_rec", "medium_rec")


def varint(data, cursor):
    value = shift = 0
    while True:
        byte = data[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, cursor
        shift += 7


def fields(data):
    cursor = 0
    while cursor < len(data):
        key, cursor = varint(data, cursor)
        wire = key & 7
        if wire == 0:
            value, cursor = varint(data, cursor)
        elif wire == 2:
            length, cursor = varint(data, cursor)
            value = data[cursor:cursor + length]
            cursor += length
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
            value = data[cursor:cursor + length]
            cursor += length
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield key >> 3, wire, value


def encode_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def encode(items):
    result = bytearray()
    for number, wire, value in items:
        result.extend(encode_varint(number << 3 | wire))
        if wire == 0:
            result.extend(encode_varint(value))
        else:
            if wire == 2:
                result.extend(encode_varint(len(value)))
            result.extend(value)
    return bytes(result)


def text(data, field):
    return next((v.decode() for n, _, v in fields(data) if n == field), "")


def attribute(data):
    result = {}
    for n, w, v in fields(data):
        if n == 1:
            result["name"] = v.decode()
        elif n == 2:
            result["float"] = struct.unpack("<f", v)[0]
        elif n == 3:
            result["int"] = v if v < 1 << 63 else v - (1 << 64)
        elif n == 4:
            result["string"] = v.decode(errors="replace")
        elif n == 7:
            result.setdefault("floats", []).extend(struct.unpack("<" + "f" * (len(v) // 4), v))
        elif n == 8:
            values = [v] if w == 0 else []
            if w == 2:
                cursor = 0
                while cursor < len(v):
                    item, cursor = varint(v, cursor)
                    values.append(item)
            result.setdefault("ints", []).extend(x if x < 1 << 63 else x - (1 << 64) for x in values)
        elif n == 5:
            result["tensor_sha256"] = hashlib.sha256(v).hexdigest()
    return result


def inventory(source):
    graph = next(v for n, _, v in fields(source) if n == 7)
    nodes = []
    for n, _, v in fields(graph):
        if n == 1:
            nodes.append({"name": text(v, 3), "operator": text(v, 4),
                          "inputs": [x.decode() for f, _, x in fields(v) if f == 1],
                          "outputs": [x.decode() for f, _, x in fields(v) if f == 2],
                          "attributes": [attribute(x) for f, _, x in fields(v) if f == 5]})
    def input_info(value):
        type_info = next(v for n, _, v in fields(value) if n == 2)
        tensor_type = next(v for n, _, v in fields(type_info) if n == 1)
        shape = next((v for n, _, v in fields(tensor_type) if n == 2), b"")
        dimensions = []
        for n, _, dimension in fields(shape):
            if n == 1:
                dimensions.append(next((v if f == 1 else v.decode() for f, _, v in fields(dimension) if f in (1, 2)), None))
        return {"name": text(value, 1), "shape": dimensions, "dtype": next((v for n, _, v in fields(tensor_type) if n == 1), None)}
    tensors = {text(v, 1): input_info(v) for n, _, v in fields(graph) if n in (11, 12, 13)}
    for node in nodes:
        for name in node["outputs"]:
            tensors.setdefault(name, {"name": name, "shape": None, "dtype": None, "note": "ONNX omits value_info; resolved by Rust shape binding"})
    initializers = []
    for n, _, v in fields(graph):
        if n != 5: continue
        dimensions = []
        for field, wire, item in fields(v):
            if field == 1:
                if wire == 0: dimensions.append(item)
                else:
                    cursor = 0
                    while cursor < len(item):
                        value, cursor = varint(item, cursor); dimensions.append(value)
        initializers.append({"name": text(v, 8), "shape": dimensions, "dtype": next((x for f, _, x in fields(v) if f == 2), None)})
    return {"sha256": hashlib.sha256(source).hexdigest(), "nodes": nodes, "tensors": tensors, "initializers": initializers,
            "operators": dict(collections.Counter(node["operator"] for node in nodes)),
            "inputs": [input_info(v) for n, _, v in fields(graph) if n == 11],
            "outputs": [text(v, 1) for n, _, v in fields(graph) if n == 12]}


def boundary(source, name):
    outer = list(fields(source))
    graph = list(fields(next(v for n, _, v in outer if n == 7)))
    nodes = [v for n, _, v in graph if n == 1]
    producers = {x.decode(): i for i, node in enumerate(nodes) for n, _, x in fields(node) if n == 2}
    if name not in producers:
        raise ValueError(f"unknown semantic tensor boundary: {name}")
    required, todo = set(), [name]
    while todo:
        value = todo.pop()
        if value in required:
            continue
        required.add(value)
        if value in producers:
            todo.extend(x.decode() for n, _, x in fields(nodes[producers[value]]) if n == 1 and x)
    kept = {producers[v] for v in required if v in producers}
    info = next((v for n, _, v in graph if n in (12, 13) and text(v, 1) == name), None)
    if info is None:
        # F32 with unknown shape: both runtimes evaluate it from the graph.
        info = encode([(1, 2, name.encode()), (2, 2, encode([(1, 2, encode([(1, 0, 1)]))]))])
    output, index = [], 0
    for n, w, v in graph:
        if n == 1:
            if index in kept:
                output.append((n, w, v))
            index += 1
        elif n != 12:
            output.append((n, w, v))
    output.append((12, 2, info))
    return encode([(n, w, encode(output) if n == 7 else v) for n, w, v in outer])


def compare(actual, reference, absolute, relative):
    a = [v[0] for v in struct.iter_unpack("<f", actual)]
    b = [v[0] for v in struct.iter_unpack("<f", reference)]
    if len(a) != len(b):
        raise ValueError("output shape/length mismatch")
    failures = sum(not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > absolute+relative*abs(y) for x, y in zip(a, b))
    return {"failures": failures, "max_abs": max((abs(x-y) for x, y in zip(a, b)), default=0),
            "absolute_tolerance": absolute, "relative_tolerance": relative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inventory", "compare"])
    parser.add_argument("--models", type=Path, default=ROOT.parent / "SimdPaddleOCR/models")
    parser.add_argument("--report", type=Path, default=ROOT / "target/metal-tensor-validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--precision", choices=["f32", "mixed"], default="f32")
    parser.add_argument("--reference", action="store_true", help="also execute C# CPU and GPU")
    parser.add_argument("--boundary", help="original ONNX output tensor name, never a node index")
    parser.add_argument("--shape", nargs=4, type=int)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    all_results = []
    for name in [args.model] if args.model else MODELS:
        path = args.models / f"{name}.onnx"
        source = path.read_bytes()
        inv = inventory(source)
        (args.report / f"{name}-inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
        if args.mode == "inventory":
            continue
        if args.boundary:
            source = boundary(source, args.boundary)
            path = args.report / f"{name}-boundary.onnx"
            path.write_bytes(source)
        shapes = [args.shape] if args.shape else ([[1, 3, 80, 160], [2, 3, 80, 160]] if name == "cls" else
                  [[1, 3, 128, 128], [1, 3, 96, 160]] if "det" in name else
                  [[batch, 3, 48, width] for batch in [1, 2] for width in [32, 128, 320]])
        for shape in shapes:
            label = f"{name}-{'x'.join(map(str, shape))}-{EXECUTOR.lower()}-{args.precision}"
            directory = args.report / label
            env = dict(os.environ, SIMD_OCR_COMPARE_DIR=str(directory.resolve()))
            executable = ROOT / ("target/release/examples/gpu_compare.exe" if os.name == "nt" else "target/release/examples/gpu_compare")
            command = [str(executable), str(path.resolve()), *map(str, shape)]
            if args.precision == "mixed": command.append("mixed")
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "stderr.txt").write_text(run.stderr, encoding="utf-8")
            result = json.loads(run.stdout.strip().splitlines()[-1]) if run.stdout.strip() else {"error": run.stderr}
            result.update(model_sha256=hashlib.sha256(source).hexdigest(), original_sha256=inv["sha256"],
                          boundary=args.boundary or inv["outputs"], returncode=run.returncode,
                          executable_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                          executor=EXECUTOR)
            if (directory / "input.f32").exists():
                result["input_sha256"] = hashlib.sha256((directory / "input.f32").read_bytes()).hexdigest()
            if args.reference and run.returncode == 0:
                for backend in ["cpu", "gpu"]:
                    prefix = directory / f"csharp-{backend}"
                    oracle = subprocess.run(["dotnet", str(ROOT / "tools/reference/bin/Release/net10.0/Sdcb.SimdPaddleOCR.GpuBench.dll"),
                                    str(path.resolve()), str(directory / "input.f32"), str(prefix), *map(str, shape), backend], capture_output=True, text=True, timeout=120)
                    if oracle.returncode:
                        result[f"csharp-{backend}"] = {"failures": 1, "error": oracle.stderr, "returncode": oracle.returncode}
                        continue
                    tolerances = (1e-3, 1e-3) if backend == "cpu" else (5e-2, 1e-2)
                    result[f"csharp-{backend}"] = compare(prefix.with_suffix(".f32").read_bytes(), (directory / "rust-cpu.f32").read_bytes(), *tolerances)
                    metadata = json.loads(prefix.with_suffix(".json").read_text())
                    result[f"csharp-{backend}"]["metadata"] = metadata
                    result[f"csharp-{backend}"]["executable_sha256"] = hashlib.sha256((ROOT / "tools/reference/bin/Release/net10.0/Sdcb.SimdPaddleOCR.GpuBench.dll").read_bytes()).hexdigest()
                    if metadata["output_shape"] != result["output_shape"]:
                        result[f"csharp-{backend}"]["failures"] += 1
            (directory / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            all_results.append(result)
            print(label, "PASS" if run.returncode == 0 else "FAIL", result.get("max_abs"), flush=True)
    (args.report / f"summary-{EXECUTOR.lower()}-{args.precision}.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    if any(r["returncode"] or any(r.get(f"csharp-{b}", {}).get("failures", 0) for b in ["cpu", "gpu"]) for r in all_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
