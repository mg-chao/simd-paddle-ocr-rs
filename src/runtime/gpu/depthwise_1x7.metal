// Canonical all-Half H1 depthwise1x7. Four adjacent spatial outputs share
// ten Half source values and seven weights; all arithmetic remains Float.
// Ordinary64-thread compute has no SIMD-width assumption or collective.
kernel void depthwise_1x7_half(
    device const half* raw_a [[buffer(0)]], device const half* raw_b [[buffer(1)]],
    device const half* raw_c [[buffer(2)]], device const half* raw_d [[buffer(3)]],
    device const half* raw_e [[buffer(4)]], device half* raw_output [[buffer(5)]],
    constant NativeParameters& params [[buffer(6)]],
    uint2 id [[thread_position_in_grid]]) {
    const uint tile = id.x + id.y * param(params, 1u);
    const uint ow = param(params, 8u), width = param(params, 6u);
    const uint row_tiles = (ow + 3u) / 4u;
    const uint row = tile / row_tiles;
    if (row >= param(params, 0u) / ow) return;
    const uint channels = param(params, 4u), channel = row % channels;
    const uint ox = (tile % row_tiles) * 4u;
    const int ix0 = int(ox) - int(param(params, 18u));
    device const half* source = raw_a + row * width;
    device const half* weights = raw_b + channel * 7u;
    float4 sum = float4(param(params, 19u) != 0u ? float(raw_c[channel]) : 0.0f);
    if (ox + 4u <= ow && ix0 >= 0 && ix0 + 10 <= int(width)) {
        const float4 v0 = float4(*reinterpret_cast<device const packed_half4*>(source + uint(ix0)));
        const float4 v1 = float4(*reinterpret_cast<device const packed_half4*>(source + uint(ix0) + 4u));
        const float2 v2 = float2(source[uint(ix0) + 8u], source[uint(ix0) + 9u]);
        sum = sum + v0 * float(weights[0]);
        sum = sum + float4(v0.y, v0.z, v0.w, v1.x) * float(weights[1]);
        sum = sum + float4(v0.z, v0.w, v1.x, v1.y) * float(weights[2]);
        sum = sum + float4(v0.w, v1.x, v1.y, v1.z) * float(weights[3]);
        sum = sum + v1 * float(weights[4]);
        sum = sum + float4(v1.y, v1.z, v1.w, v2.x) * float(weights[5]);
        sum = sum + float4(v1.z, v1.w, v2.x, v2.y) * float(weights[6]);
    } else {
        // Preserve skip-padding semantics, including nonfinite weights. The
        // partial last vector never loads or stores outside logical rows.
        #pragma clang loop unroll(full)
        for (uint k = 0u; k < 7u; ++k) {
            const float weight = float(weights[k]);
            #pragma clang loop unroll(full)
            for (uint pixel = 0u; pixel < 4u; ++pixel) {
                const int ix = ix0 + int(pixel + k);
                if (ox + pixel < ow && ix >= 0 && ix < int(width))
                    sum[pixel] = sum[pixel] + float(source[uint(ix)]) * weight;
            }
        }
    }
    const uint index = row * ow + ox;
    #pragma clang loop unroll(full)
    for (uint pixel = 0u; pixel < 4u; ++pixel) {
        if (ox + pixel < ow)
            raw_output[index + pixel + param(params, 60u)] =
                half(epilogue(sum[pixel], index + pixel, raw_d, raw_e, params));
    }
}
