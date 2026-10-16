"""Exact-rational width-four C# generic softmax arithmetic, no execution oracle.

No Rust or C# inference runs. Full vectors use the pinned generic exp polynomial;
every scalar tail is a maximum (exp(0)=1 exactly). Scalar fallback cases are
uniform too, so their MathF.Exp(0) primitive needs no implementation model.
"""
import argparse
import contextlib
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference-dir", type=Path,
    default=ROOT.parent / "SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Kernels",
    help="Kernels directory from the pinned C# revision")
parser.add_argument("--output-dir", type=Path, default=ROOT / "target/portable-softmax-oracle",
    help="Destination for independent JSON and Rust literal goldens")
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
GENERATOR = ROOT / "tools/generate_unary_policy_oracle.py"
REFERENCE = args.reference_dir.resolve()
saved = sys.argv
try:
    sys.argv = [str(GENERATOR), "--reference-source", str(REFERENCE / "Tensor.Avx.cs"), "--output", str(OUT / "unary-oracle-support.json")]
    with contextlib.redirect_stdout(io.StringIO()):
        primitives = runpy.run_path(str(GENERATOR))
finally:
    sys.argv = saved
number, rounded = primitives["number"], primitives["rounded"]
add, mul, div, exp = (primitives[name] for name in ("add", "mul", "div", "exp"))
ONE = 0x3f800000
vector_source = (REFERENCE / "Tensor.Vector.cs").read_text(encoding="utf-8")
tensor_source = (REFERENCE / "Tensor.cs").read_text(encoding="utf-8")
if hashlib.sha256(tensor_source.encode()).hexdigest() != "207c887ad9b5360bd85863b608ed46d6d99db4f637db8bbcdc44a9733c754dfa":
    parser.error("Reference Tensor.cs differs from the pinned revision")
if hashlib.sha256(vector_source.encode()).hexdigest() != "d6a6d064ec148831866143a2fecaf0484b024a7426bb14a0591b950cefea8b73":
    parser.error("Reference Tensor.Vector.cs differs from the pinned revision")
if "VecMulAdd(Vector<float> x, Vector<float> y, Vector<float> z) => x * y + z;" not in vector_source:
    parser.error("Generic vector multiply/add must remain nonfused")
constants = {"Log2e": "1.4426950408889634", "C1": "0.693359375", "C2": "-2.12194440e-4",
             "Min": "-88.3762626647949", "Max": "88.3762626647949", "Half": "0.5", "One": "1",
             "P0": "2.4801587302e-5", "P1": "1.9841269841e-4", "P2": "1.3888888889e-3",
             "P3": "8.3333333333e-3", "P4": "4.1666666667e-2", "P5": "1.6666666667e-1"}
for name, literal in constants.items():
    found = re.search(r"VecExp" + name + r" = new\(([-+.\deE]+)f\)", vector_source)
    if found is None or Fraction(found[1]) != Fraction(literal):
        parser.error("Pinned generic exp coefficient changed: " + name)
if not ("ExpExactVector(VecLoad(inputRow + i) - vmax)" in tensor_source):
    parser.error("Reference generic vector exponential routing changed")
if not ("for (int lane = 0; lane < width; lane++) sum += sumVector.GetElement(lane);" in tensor_source):
    parser.error("Reference generic softmax lane order changed")
if not ("VecStore(outputRow + i, VecLoad(outputRow + i) * inv);" in tensor_source):
    parser.error("Reference generic reciprocal multiplication changed")


def fold(values):
    total = 0
    for value in values:
        total = add(total, value)
    return total


def softmax4(inputs):
    maximum = max(inputs, key=number)
    tiled = len(inputs) // 4 * 4
    shifted = [add(value, maximum ^ 0x80000000) for value in inputs]
    # The fixture deliberately avoids modeling the platform scalar exp.
    if not all(number(value) == 0 for value in shifted[tiled:]):
        raise ValueError("Scalar tails must be exact exp(0) cases")
    if len(inputs) < 4:
        if not all(number(value) == 0 for value in shifted):
            raise ValueError("Scalar fallback oracle requires uniform inputs")
        expected = [div(ONE, rounded(Fraction(len(inputs))))] * len(inputs)
        return {"expected": expected, "sum": rounded(Fraction(len(inputs))), "lane_order_sensitive": False,
                "reciprocal_multiply_sensitive": False}
    exponents = [exp(value) for value in shifted[:tiled]] + [ONE] * (len(inputs) - tiled)
    lane_sums = [0] * 4
    for offset in range(0, tiled, 4):
        lane_sums = [add(lane_sums[lane], exponents[offset + lane]) for lane in range(4)]
    total = fold(lane_sums + exponents[tiled:])
    inverse = div(ONE, total)
    expected = [mul(value, inverse) for value in exponents[:tiled]] + [div(ONE, total)] * (len(inputs) - tiled)
    return {"expected": expected, "sum": total, "lane_order_sensitive": total != fold(exponents),
            "reciprocal_multiply_sensitive": any(expected[i] != div(exponents[i], total) for i in range(tiled))}


cases = []
for columns in (3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 32, 33):
    tiled = columns // 4 * 4
    inputs = [rounded(-Fraction((i * 7 + 3) % 31, 3)) for i in range(tiled)] + [0] * (columns - tiled)
    if columns < 4:
        inputs = [0] * columns
    elif columns == tiled:
        inputs[-1] = 0
    cases.append({"columns": columns, "inputs": inputs, **softmax4(inputs)})
if not any(case["lane_order_sensitive"] for case in cases):
    raise ValueError("The oracle must distinguish lane summation order")
if not any(case["reciprocal_multiply_sensitive"] for case in cases):
    raise ValueError("The oracle must distinguish reciprocal multiplication")
report = {"schema_version": 1, "kind": "Independent exact-rational pinned generic width-four arithmetic; not C# execution",
          "reference_revision": "33ff8a81672452041d110e8ba52dcd2639e54228",
          "reference_normalized_sha256": {name: hashlib.sha256((REFERENCE / name).read_text(encoding="utf-8").encode()).hexdigest()
                                         for name in ("Tensor.cs", "Tensor.Vector.cs")},
          "primitive_generator_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
          "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "scalar_tail_contract": "All scalar tails are maxima: shifted input zero, exact MathF.Exp(0)=1",
          "nonfused_proof": "Tensor.Vector.cs:178 VecMulAdd => x * y + z",
          "cases": cases}
(OUT / "oracle.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
lines = ["// Independent exact-rational pinned generic width-four softmax goldens.",
         "// Regenerate: python tools/generate_portable_softmax_oracle.py --output-dir target/portable-softmax-oracle",
         "// See docs/cpu-validation.md; no C# execution.",
         "struct Case { inputs: &'static [u32], expected: &'static [u32] }", "const CASES: &[Case] = &["]
for case in cases:
    bits = lambda values: ", ".join(f"0x{value:08x}" for value in values)
    lines += [f"    Case {{ inputs: &[{bits(case['inputs'])}], expected: &[{bits(case['expected'])}] }},"]
lines += ["];"]
(OUT / "goldens.rs").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"cases": len(cases), "lane_order_sensitive": sum(c["lane_order_sensitive"] for c in cases),
                  "reciprocal_sensitive": sum(c["reciprocal_multiply_sensitive"] for c in cases)}, indent=2))
