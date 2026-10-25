// MPP convolution on physical NHWC/HWIO operands, accumulating in FP32.
// $OPERAND is float for strict mode or half for separately qualified fast mode.
// MPP convolution requires contiguous channels: nonunit fastest-dimension
// strides in a tensor_inline view do not implement NCHW/OIHW on this compiler.
// The physical schedule packs activations and caches immutable HWIO weights.
// Weight output channels are zero-padded to a multiple of 32.
// Output remains canonical NCHW, with the common fused epilogue/output offset.
// Tile: 32 output channels x 8 output columns x 4 output rows, one image.
// Grid ABI: words32=ceil(Cout/32), words33=ceil(OW/8)*ceil(OH/4),
// word34=grid.x, word35=N, word36=padded Cout; 128 threads/workgroup.
// Descriptor source dimensions are literals specialized to the actual input.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#if !defined(__HAVE_TENSOR__)
#error "MPP convolution requires Metal tensor support"
#endif

using namespace mpp::tensor_ops;

template <int KERNEL, int STRIDE_H, int STRIDE_W>
inline void conv_mpp_body(
    device const $OPERAND* raw_a, device const $OPERAND* raw_b,
    device const float* raw_c, device const float* raw_d,
    device const float* raw_e, device float* raw_output,
    constant NativeParameters& params, uint2 group, uint tid, uint width) {
    uint block = group.x + group.y * param(params, 34u);
    uint batch = block / (param(params, 32u) * param(params, 33u));
    if (batch >= param(params, 35u)) return;
    uint output_channels = param(params, 4u);
    uint weight_output_stride = param(params, 36u);
    uint channels = param(params, 3u);
    uint input_height = param(params, 5u), input_width = param(params, 6u);
    uint output_height = param(params, 7u), output_width = param(params, 8u);
    uint output_plane = output_height * output_width;
    uint input_plane = input_height * input_width;
    uint channel_origin = ((block / param(params, 33u)) % param(params, 32u)) * 32u;
    uint spatial_tile = block % param(params, 33u);
    uint x_tiles = (output_width + 7u) / 8u;
    uint x_origin = (spatial_tile % x_tiles) * 8u;
    uint y_origin = (spatial_tile / x_tiles) * 4u;
    int source_x = int(x_origin * STRIDE_W) - int(param(params, 18u));
    int source_y = int(y_origin * STRIDE_H) - int(param(params, 17u));

    // The operation's full source view supplies zero padding, while final
    // stores mask channel/spatial tails. Keep the scalar path for unsupported
    // arithmetic geometry or a different actual SIMD width.
    bool interior = width == 32u && param(params, 9u) == channels
        && param(params, 10u) == output_channels
        && param(params, 15u) == 1u && param(params, 16u) == 1u
        && param(params, 52u) == 0u;
    if (!interior) {
        uint channels_per_group = param(params, 9u);
        uint reduction_count = channels_per_group * uint(KERNEL * KERNEL);
        for (uint local = tid; local < 32u * 8u * 4u; local += 128u) {
            uint channel = channel_origin + local / 32u;
            uint y = y_origin + (local % 32u) / 8u;
            uint x = x_origin + local % 8u;
            if (channel >= output_channels || y >= output_height || x >= output_width) continue;
            uint first = (channel / param(params, 10u)) * channels_per_group;
            float sum = param(params, 19u) != 0u ? raw_c[channel] : 0.0f;
            for (uint k = 0u; k < reduction_count; ++k) {
                uint input_channel = first + k / uint(KERNEL * KERNEL);
                uint tap = k % uint(KERNEL * KERNEL);
                int iy = int(y * STRIDE_H + (tap / uint(KERNEL)) * param(params, 15u)) - int(param(params, 17u));
                int ix = int(x * STRIDE_W + (tap % uint(KERNEL)) * param(params, 16u)) - int(param(params, 18u));
                if (iy < 0 || ix < 0 || iy >= int(input_height) || ix >= int(input_width)) continue;
                float value = float(raw_a[((batch * input_height + uint(iy)) * input_width + uint(ix)) * channels + input_channel]);
                if (param(params, 52u) == 1u) value *= raw_d[batch * param(params, 53u) + input_channel];
                else if (param(params, 52u) == 2u) value *= raw_e[batch * param(params, 53u) + input_channel];
                uint packed_weight = (tap * channels_per_group + k / uint(KERNEL * KERNEL)) * weight_output_stride + channel;
                sum += value * float(raw_b[packed_weight]);
            }
            uint index = (batch * output_channels + channel) * output_plane + y * output_width + x;
            raw_output[index + param(params, 60u)] = epilogue(sum, index, raw_d, raw_e, params);
        }
        return;
    }

    constexpr auto descriptor = convolution2d_descriptor(
        int4(32, 8, 4, 1),
        int4($INPUT_CHANNELS, $INPUT_WIDTH, $INPUT_HEIGHT, 1),
        int2(KERNEL, KERNEL), convolution2d_activation_layout::nhwc,
        convolution2d_weights_layout::hwio, int2(STRIDE_W, STRIDE_H), int2(1), 1, false,
        convolution2d_descriptor::mode::multiply_accumulate);
    convolution2d<descriptor, execution_simdgroups<4>> operation;
    // MPP offsets describe the center of the first kernel window in the full
    // image view. Signed coordinates retain asymmetric and oversized padding.
    operation.set_offsets(int2(source_x + KERNEL / 2, source_y + KERNEL / 2));
    using Tensor4 = tensor<device $OPERAND, dextents<int, 4>, tensor_inline>;
    // Physical channels are contiguous in both operands; arbitrary outer
    // strides and spatial/batch slices remain representable by tensor_inline.
    Tensor4 activations(const_cast<device $OPERAND*>(raw_a),
        dextents<int, 4>(int(channels), int(input_width), int(input_height), int(param(params, 35u))),
        array<int, 4>({1, int(channels), int(channels * input_width), int(channels * input_plane)}));
    Tensor4 weights(const_cast<device $OPERAND*>(raw_b),
        dextents<int, 4>(int(weight_output_stride), int(channels), KERNEL, KERNEL),
        array<int, 4>({1, int(weight_output_stride), int(weight_output_stride * channels), int(KERNEL * weight_output_stride * channels)}));
    auto image = activations.slice(0, 0, 0, int(batch));
    auto filter = weights.slice(int(channel_origin), 0, 0, 0);
    auto accum = operation.template get_destination_cooperative_tensor<decltype(image), decltype(filter), float>();
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < accum.get_capacity(); ++i) {
        if (accum.is_valid_element(i)) {
            auto coordinate = accum.get_multidimensional_index(i);
            uint channel = channel_origin + uint(coordinate[0]);
            accum[i] = param(params, 19u) != 0u && channel < output_channels ? raw_c[channel] : 0.0f;
        }
    }
    operation.run(image, filter, accum);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < accum.get_capacity(); ++i) {
        if (accum.is_valid_element(i)) {
            auto coordinate = accum.get_multidimensional_index(i);
            uint channel = channel_origin + uint(coordinate[0]);
            uint x = x_origin + uint(coordinate[1]);
            uint y = y_origin + uint(coordinate[2]);
            if (channel < output_channels && x < output_width && y < output_height) {
                uint index = (batch * output_channels + channel) * output_plane + y * output_width + x;
                raw_output[index + param(params, 60u)] = epilogue(accum[i], index, raw_d, raw_e, params);
            }
        }
    }
}

#define CONV_MPP_ENTRY(NAME, K, SH, SW) \
kernel void NAME(device const $OPERAND* a [[buffer(0)]], device const $OPERAND* b [[buffer(1)]], \
    device const float* c [[buffer(2)]], device const float* d [[buffer(3)]], \
    device const float* e [[buffer(4)]], device float* output [[buffer(5)]], \
    constant NativeParameters& params [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], uint width [[threads_per_simdgroup]]) { \
    conv_mpp_body<K, SH, SW>(a, b, c, d, e, output, params, group, tid, width); \
}

CONV_MPP_ENTRY($ENTRY, $KERNEL, $STRIDE_H, $STRIDE_W)
