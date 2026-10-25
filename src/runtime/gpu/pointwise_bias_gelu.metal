// Exact Add(channel D), GELU signature. The unused E operand is never read.
inline float epilogue_fixed_bg(float initial, uint index, uint channel,
    device const half* raw_d, device const half* raw_e, constant NativeParameters& params) {
    float value = initial + float(raw_d[channel]);
    return ((erf_approx(value / 1.4142135623730951f) + 1.0f) * value) * 0.5f;
}
