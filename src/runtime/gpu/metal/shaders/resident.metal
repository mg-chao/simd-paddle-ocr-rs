#include <metal_stdlib>
using namespace metal;
struct Params {
  uint4 words[64];
};
inline float round_to_half(float value) {
  uint bits = as_type<uint>(value), sign = bits & 0x80000000u,
       magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u)
    return value;
  if (magnitude < 0x38800000u) {
    if (magnitude <= 0x33000000u)
      return as_type<float>(sign);
    uint shift = 126u - (magnitude >> 23u),
         mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    uint rounded =
        (mantissa + ((1u << (shift - 1u)) - 1u) + ((mantissa >> shift) & 1u)) >>
        shift;
    return as_type<float>(sign | as_type<uint>(float(rounded) * 0x1p-24f));
  }
  uint rounded = magnitude + 0xfffu + ((magnitude >> 13u) & 1u);
  return as_type<float>(
      sign | (rounded >= 0x47800000u ? 0x7f800000u : rounded & 0xffffe000u));
}
inline OUT normalize_store(float value) {
#if HALF_OUTPUT
  return stored_half(value);
#else
  return value;
#endif
}
