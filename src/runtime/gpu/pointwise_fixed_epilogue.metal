// The host proves Add(channel D), Add(full E), GELU before selecting this
// separate source. Preserve each Float operation and the final Half store.
inline float epilogue_fixed_brg(float initial, uint index, uint channel,
    device const half* raw_d, device const half* raw_e, constant NativeParameters& params) {
    float value = initial + float(raw_d[channel]);
    value = value + float(raw_e[index]);
    return ((erf_approx(value / 1.4142135623730951f) + 1.0f) * value) * 0.5f;
}
