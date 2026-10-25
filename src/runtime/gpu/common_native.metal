#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Same ABI as common.wgsl: five read operands, output, 128 packed words.
// Operands and output are canonical FP32 NCHW. One 64-thread workgroup owns
// two independent 32x32 output tiles, phased across its actual simdgroups.
struct NativeParameters {
    uint4 words[32];
};

inline uint param(constant NativeParameters& params, uint i) {
    return params.words[i / 4u][i % 4u];
}

inline float erf_approx(float x) {
    if (abs(x) >= 6.0f) return x < 0.0f ? -1.0f : 1.0f;
    float t = 1.0f / (1.0f + 0.3275911f * abs(x));
    float poly = (((((1.0614054f * t - 1.4531521f) * t) + 1.4214138f) * t
                    - 0.28449672f) * t + 0.2548296f) * t;
    return (x < 0.0f ? -1.0f : 1.0f) * (1.0f - poly * exp(-x * x));
}

inline float epilogue(float initial, uint index,
                      device const float* raw_d, device const float* raw_e,
                      constant NativeParameters& params) {
    float value = initial;
    for (uint i = 0u; i < param(params, 64u); ++i) {
        uint base = 68u + i * 3u;
        uint op = param(params, base);
        if (op <= 2u) {
            uint offset = index;
            if (param(params, base + 1u) == 0u) offset = 0u;
            if (param(params, base + 1u) == 1u)
                offset = (index / param(params, 66u)) % param(params, 65u);
            value = value + (op == 1u ? raw_d[offset] : raw_e[offset]);
        } else if (op == 3u) {
            value = value > 0.0f ? value : 0.0f;
        } else if (op == 4u) {
            if (value >= 0.0f) value = 1.0f / (1.0f + exp(-value));
            else {
                float x = exp(value);
                value = x / (1.0f + x);
            }
        } else if (op == 5u) {
            value = clamp(value * as_type<float>(param(params, base + 1u))
                          + as_type<float>(param(params, base + 2u)), 0.0f, 1.0f);
        } else if (op == 6u) {
            value = value * clamp(value * 0.16666667f + 0.5f, 0.0f, 1.0f);
        } else if (op == 7u) {
            value = ((erf_approx(value / 1.4142135623730951f) + 1.0f) * value) * 0.5f;
        } else if (op == 8u) {
            float gate;
            if (value >= 0.0f) gate = 1.0f / (1.0f + exp(-value));
            else {
                float x = exp(value);
                gate = x / (1.0f + x);
            }
            value = value * gate;
        }
    }
    return value;
}

