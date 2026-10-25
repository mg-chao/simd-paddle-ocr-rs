#include <metal_stdlib>
using namespace metal;

// Preserve an FP16 tensor-store boundary even when fast math folds local
// float(half(value)) casts. Integer RNE also handles subnormals and overflow.
inline float round_to_half(float value) {
    const uint bits = as_type<uint>(value);
    const uint sign = bits & 0x80000000u;
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return value;
    if (magnitude < 0x38800000u) {
        if (magnitude <= 0x33000000u) return as_type<float>(sign);
        const uint shift = 126u - (magnitude >> 23u);
        const uint mantissa = (magnitude & 0x007fffffu) | 0x00800000u;
        const uint rounded = (mantissa + ((1u << (shift - 1u)) - 1u)
                              + ((mantissa >> shift) & 1u)) >> shift;
        return as_type<float>(sign | as_type<uint>(float(rounded) * 0x1p-24f));
    }
    const uint rounded = magnitude + 0x00000fffu + ((magnitude >> 13u) & 1u);
    if (rounded >= 0x47800000u) return as_type<float>(sign | 0x7f800000u);
    return as_type<float>(sign | (rounded & 0xffffe000u));
}
