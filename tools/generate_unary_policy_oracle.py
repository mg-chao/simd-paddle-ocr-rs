"""Independent exact-rational IEEE F32 oracle for pinned C# unary order.

No Rust code is loaded. Every primitive is rounded directly from Fraction to
binary32, including fused multiply-add; no f64 double-rounding is involved.
"""
from fractions import Fraction as Q
from pathlib import Path
import argparse, hashlib, json, re, struct

def f32bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]

def number(bits):
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7fffff
    assert exponent != 255
    if exponent:
        mantissa += 1 << 23
        exponent -= 150
    else:
        exponent = -149
    return sign * Q(mantissa) * (Q(2) ** exponent)

def rounded(value, zero_sign=0):
    sign = int(value < 0)
    value = abs(value)
    if not value:
        return zero_sign << 31
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Q(2) ** exponent:
        exponent -= 1
    shift = 149 if exponent < -126 else 23 - exponent
    scaled = value * Q(2) ** shift
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    integer += twice > scaled.denominator or (twice == scaled.denominator and integer % 2)
    if exponent < -126:
        return (sign << 31) | integer
    if integer == 1 << 24:
        integer >>= 1
        exponent += 1
    if exponent > 127:
        return (sign << 31) | 0x7f800000
    return (sign << 31) | ((exponent + 127) << 23) | (integer - (1 << 23))

def add(a, b):
    return rounded(number(a) + number(b), (a & b) >> 31)

def mul(a, b):
    return rounded(number(a) * number(b), (a ^ b) >> 31)

def div(a, b):
    return rounded(number(a) / number(b), (a ^ b) >> 31)

def madd(a, b, c, fused):
    if not fused:
        return add(mul(a, b), c)
    return rounded(number(a) * number(b) + number(c), ((a ^ b) & c) >> 31)

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-source', type=Path,
    default=ROOT.parent/'SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Kernels/Tensor.Avx.cs')
parser.add_argument('--output', type=Path, default=ROOT/'target/cpu-unary-policy-oracle-20261016.json')
args = parser.parse_args()
reference = args.reference_source
source = reference.read_text(encoding='utf-8')
# Pin content while accepting the same checkout with either CRLF or LF.
if hashlib.sha256(source.encode('utf-8')).hexdigest() != '0276e3381f361c30c1dade1dbf1c02f262337a615ce301d333f28368b73df5e4':
    parser.error('Reference Tensor.Avx.cs differs from the pinned revision')
coefficients = [[rounded(Q(re.search(r'V'+part+str(i)+r' = Vector256.Create\(([-+.\deE]+)f\)', source)[1])) for i in range(9)] for part in ('Small','Middle','Large')]
one, half = f32bits(1.0), f32bits(0.5)
inverse = rounded(Q('0.70710678118654752'))
sqrt2 = 0x3fb504f3

def erf(value, fused):
    magnitude = value & 0x7fffffff
    a = number(magnitude)
    if a >= 4:
        result = one
    else:
        interval = 0 if a < 1 else (1 if a < 2 else 2)
        x = mul(magnitude, magnitude) if interval == 0 else add(magnitude, f32bits(-1.5 if interval == 1 else -3.0))
        coeff = coefficients[interval]
        result = coeff[0]
        for c in coeff[1:]:
            result = madd(x, result, c, fused)
        if interval == 0:
            result = mul(magnitude, result)
    return result | (value & 0x80000000)

def gelu(value, fused, native):
    scaled = mul(value, inverse) if native else div(value, sqrt2)
    activated = add(erf(scaled, fused), one)
    return mul(mul(value, activated), half)

def exp(value):
    value = max(number(rounded(Q('-88.3762626647949'))), min(number(rounded(Q('88.3762626647949'))), number(value)))
    value = rounded(value)
    scaled = add(mul(value, rounded(Q('1.4426950408889634'))), half)
    integer = number(scaled).__floor__()
    n = rounded(Q(integer))
    reduced = add(value, mul(n, rounded(Q('0.693359375'))) ^ 0x80000000)
    reduced = add(reduced, mul(n, rounded(Q('-2.12194440e-4'))) ^ 0x80000000)
    p = rounded(Q('2.4801587302e-5'))
    for c in ['1.9841269841e-4','1.3888888889e-3','8.3333333333e-3','4.1666666667e-2','1.6666666667e-1','0.5']:
        p = add(rounded(Q(c)), mul(reduced, p))
    p = add(one, add(reduced, mul(mul(reduced, reduced), p)))
    return mul(p, (integer + 127) << 23)

def sigmoid(value):
    e = exp((value & 0x7fffffff) | 0x80000000)
    positive = div(one, add(one, e))
    return add(one, positive ^ 0x80000000) if value >> 31 else positive

inputs = list(map(f32bits, [-0.0,0.25,-0.5,1.0000001192092896,-1.5,1.9999998807907104,-2.25,3.999999761581421]))
for i in range(1,4096):
    bits = f32bits(i / 997)
    if erf(bits, False) != erf(bits, True):
        inputs = [bits, bits ^ 0x80000000] + inputs[:6]
        break
gelu_inputs = list(map(f32bits, [-2.75,-1.75,-1.13,-0.31,0.31,0.74,1.13,2.75]))
# Include exact cases where inverse multiply and sqrt2 division differ in the
# final GELU result. This search performs no implementation inference.
for i in range(1,1000):
    bits = f32bits(i / 997)
    if gelu(bits, False, True) != gelu(bits, False, False):
        gelu_inputs = [bits, bits ^ 0x80000000] + gelu_inputs[:6]
        break
for i in range(1,4096):
    bits = f32bits(i / 997)
    if gelu(bits, True, True) != gelu(bits, False, True):
        gelu_inputs = [bits, bits ^ 0x80000000] + gelu_inputs[:6]
        break
sigmoid_inputs = list(map(f32bits, [-17.0,-16.0,-3.3,-0.1,0.1,2.5,8.0,20.0]))
data = {'kind':'Exact-rational independent IEEE binary32 arithmetic oracle, not a C# executable capture', 'reference_revision':'33ff8a81672452041d110e8ba52dcd2639e54228', 'reference_file_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(), 'erf_inputs':inputs,'erf_separate':[erf(x,False) for x in inputs],'erf_fma':[erf(x,True) for x in inputs],'gelu_inputs':gelu_inputs,'gelu_native_separate':[gelu(x,False,True) for x in gelu_inputs],'gelu_native_fma':[gelu(x,True,True) for x in gelu_inputs],'gelu_old_divide':[gelu(x,False,False) for x in gelu_inputs], 'sigmoid_inputs':sigmoid_inputs,'sigmoid_native':[sigmoid(x) for x in sigmoid_inputs]}
data['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
for key,value in data.items():
    if isinstance(value,list):
        print(key, '['+', '.join(f'0x{x:08x}' for x in value)+']')
