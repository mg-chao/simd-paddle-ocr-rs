// Exact Add(channel D), Add(full E) signature. Preserve the Float addition
// order; the existing pointwise kernel performs the sole final Half store.
inline float epilogue_fixed_adds(float initial, uint index, uint channel,
    device const half* raw_d, device const half* raw_e, constant NativeParameters& params) {
    float value = initial + float(raw_d[channel]);
    return value + float(raw_e[index]);
}
