// Canonical Half MPP output row is the proven channel coordinate. The host
// selects this source only for bounded K and one to three epilogue packets.
// Float operations and order match the common generic epilogue exactly.
inline float epilogue_known_step(float value, uint index, uint known_channel, uint base,
    device const half* raw_d, device const half* raw_e, constant NativeParameters& params) {
        uint op = param(params, base);
        if (op <= 2u) {
            uint offset = index;
            if (param(params, base + 1u) == 0u) offset = 0u;
            if (param(params, base + 1u) == 1u)
                offset = known_channel;
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
    return value;
}
inline float epilogue_known(float initial, uint index, uint known_channel,
    device const half* raw_d, device const half* raw_e, constant NativeParameters& params) {
    const uint count = param(params, 64u);
    if (count > 3u) return epilogue(initial, index, raw_d, raw_e, params);
    float value = initial;
    if (count > 0u) value = epilogue_known_step(value, index, known_channel, 68u, raw_d, raw_e, params);
    if (count > 1u) value = epilogue_known_step(value, index, known_channel, 71u, raw_d, raw_e, params);
    if (count > 2u) value = epilogue_known_step(value, index, known_channel, 74u, raw_d, raw_e, params);
    return value;
}

