#include <metal_stdlib>
using namespace metal;
#ifdef USE_MPP
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>
#endif

// Explicit storage conversion with round-to-nearest-even and preserved IEEE
// overflow/subnormal behavior, independent of the arithmetic compiler mode.
inline float round_to_half(float value) {
  uint bits = as_type<uint>(value), sign = bits & 0x80000000u,
       magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u)
    return value;
  if (magnitude >= 0x477ff000u)
    return as_type<float>(sign | 0x7f800000u);
  if (magnitude < 0x33000000u)
    return as_type<float>(sign);
  if (magnitude < 0x38800000u) {
    uint shift = 126u - (magnitude >> 23u),
         mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    uint rounded =
        (mantissa + ((1u << (shift - 1u)) - 1u) + ((mantissa >> shift) & 1u)) >>
        shift;
    return copysign(float(rounded) * 0x1p-24f, value);
  }
  return as_type<float>(
      sign | ((magnitude + 0xfffu + ((magnitude >> 13u) & 1u)) & 0xffffe000u));
}
inline float error_function(float x) {
  if (abs(x) >= 6.0f)
    return x < 0.0f ? -1.0f : 1.0f;
  float t = 1.0f / (1.0f + 0.3275911f * abs(x));
  float polynomial =
      (((((1.0614054f * t - 1.4531521f) * t) + 1.4214138f) * t - 0.28449672f) *
           t +
       0.2548296f) *
      t;
  return (x < 0.0f ? -1.0f : 1.0f) * (1.0f - polynomial * exp(-x * x));
}
inline float sigmoid(float x) {
  if (x >= 0.0f)
    return 1.0f / (1.0f + exp(-x));
  float e = exp(x);
  return e / (1.0f + e);
}
constant uint round_intermediate [[function_constant(0)]];
struct NeuralParameters {
  uint words[128];
};
struct Neural {
  device const STORAGE_A *a;
  device const STORAGE_B *b;
  device const STORAGE_C *c;
  device const STORAGE_D *d;
  device const STORAGE_E *e;
  device STORAGE_O *output;
  constant NeuralParameters &parameters;
  uint p(uint index) const { return parameters.words[index]; }
  float load_a(uint i) const { return float(a[i]); }
  float load_b(uint i) const { return float(b[i]); }
  float load_c(uint i) const { return float(c[i]); }
  float load_d(uint i) const { return float(d[i]); }
  float load_e(uint i) const { return float(e[i]); }
  float output_round(float v) const {
    return round_intermediate != 0 ? round_to_half(v) : v;
  }
  void store_output(uint i, float v) const {
#if HALF_OUTPUT
    output[i + p(60)] = stored_half(v);
#else
    output[i + p(60)] = v;
#endif
  }
  float epilogue(float value, uint index) const {
    for (uint i = 0; i < p(64); ++i) {
      uint base = 68 + i * 3, op = p(base);
      if (op <= 2) {
        uint offset = index;
        if (p(base + 1) == 0)
          offset = 0;
        else if (p(base + 1) == 1)
          offset = (index / p(66)) % p(65);
        value += op == 1 ? load_d(offset) : load_e(offset);
      } else if (op == 3)
        value = value > 0.0f ? value : 0.0f;
      else if (op == 4)
        value = sigmoid(value);
      else if (op == 5)
        value = clamp(value * as_type<float>(p(base + 1)) +
                          as_type<float>(p(base + 2)),
                      0.0f, 1.0f);
      else if (op == 6)
        value *= clamp(value * 0.16666667f + 0.5f, 0.0f, 1.0f);
      else if (op == 7)
        value = ((error_function(value / 1.4142135623730951f) + 1.0f) * value) *
                0.5f;
      else if (op == 8)
        value *= sigmoid(value);
    }
    return value;
  }
