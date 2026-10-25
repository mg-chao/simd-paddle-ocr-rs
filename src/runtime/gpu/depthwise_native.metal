// NativeParameters, param(), and the shared FP32 epilogue precede this file.
// Four adjacent output pixels share one input window and one set of weights.
kernel void depthwise_native(
    device const float* raw_a [[buffer(0)]],
    device const float* raw_b [[buffer(1)]],
    device const float* raw_c [[buffer(2)]],
    device const float* raw_d [[buffer(3)]],
    device const float* raw_e [[buffer(4)]],
    device float* raw_output [[buffer(5)]],
    constant NativeParameters& params [[buffer(6)]],
    uint2 id [[thread_position_in_grid]]) {
    constexpr uint K = $KERNELu;
    const uint tile = id.x + id.y * param(params, 1u);
    const uint ow = param(params, 8u);
    const uint row_tiles = (ow + 3u) / 4u;
    const uint row = tile / row_tiles;
    if (row >= param(params, 0u) / ow) return;
    const uint ox = (tile % row_tiles) * 4u;
    const uint oh = param(params, 7u);
    const uint channels = param(params, 4u);
    const uint channel = (row / oh) % channels;
    const uint batch = row / (oh * channels);
    const uint height = param(params, 5u);
    const uint width = param(params, 6u);
    const uint sh = param(params, 13u);
    const uint sw = param(params, 14u);
    const int iy0 = int((row % oh) * sh) - int(param(params, 17u));
    const int ix0 = int(ox * sw) - int(param(params, 18u));
    const uint plane = (batch * channels + channel) * height * width;
    device const float* weights = raw_b + channel * K * K;
    float4 sum = float4(param(params, 19u) != 0u ? raw_c[channel] : 0.0f);

    // Horizontally complete windows use packed vector loads. Each generated
    // row checks its own vertical bounds, retaining vector reuse at the top
    // and bottom of short recognition maps. The last vector may be partial.
    if (ox + 4u <= ow && ix0 >= 0
        && ix0 + int(3u * sw + K) <= int(width)) {
        if (sw == 1u) {
            $STRIDE1
        } else {
            $STRIDE2
        }
    } else {
        // Edges and narrow/tail rows retain the scalar convolution's exact
        // padding semantics. No speculative load may cross a tensor binding.
        #pragma clang loop unroll(full)
        for (uint ky = 0u; ky < K; ++ky) {
            const int iy = iy0 + int(ky);
            if (iy < 0 || iy >= int(height)) continue;
            device const float* source = raw_a + plane + uint(iy) * width;
            #pragma clang loop unroll(full)
            for (uint kx = 0u; kx < K; ++kx) {
                const float weight = weights[ky * K + kx];
                #pragma clang loop unroll(full)
                for (uint pixel = 0u; pixel < 4u; ++pixel) {
                    const int ix = ix0 + int(pixel * sw + kx);
                    if (ox + pixel < ow && ix >= 0 && ix < int(width))
                        sum[pixel] = sum[pixel] + source[uint(ix)] * weight;
                }
            }
        }
    }
    const uint index = row * ow + ox;
    #pragma clang loop unroll(full)
    for (uint pixel = 0u; pixel < 4u; ++pixel) {
        if (ox + pixel < ow)
            raw_output[index + pixel + param(params, 60u)] =
                epilogue(sum[pixel], index + pixel, raw_d, raw_e, params);
    }
}
