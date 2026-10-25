// Strict FP32 pointwise convolution. One 128-thread workgroup owns one 64x64
// tile. Host words32/33 are row/column tile counts, 34 is grid width, 35 batch.
// NativeParameters, param(), and epilogue() come from common_native.metal.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#if !defined(__HAVE_TENSOR__)
#error "MPP pointwise64 requires Metal tensor support"
#endif

using namespace mpp::tensor_ops;

kernel void pointwise_mpp64(
    device const float* raw_a [[buffer(0)]],
    device const float* raw_b [[buffer(1)]],
    device const float* raw_c [[buffer(2)]],
    device const float* raw_d [[buffer(3)]],
    device const float* raw_e [[buffer(4)]],
    device float* raw_output [[buffer(5)]],
    constant NativeParameters& params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_simdgroup]]) {
    const uint rows = param(params, 4u);
    const uint inner = param(params, 3u);
    const uint columns = param(params, 7u) * param(params, 8u);
    const uint block = group.x + group.y * param(params, 34u);
    const uint batch = block / (param(params, 32u) * param(params, 33u));
    if (batch >= param(params, 35u)) return;
    const uint mr = ((block / param(params, 33u)) % param(params, 32u)) * 64u;
    const uint nc = (block % param(params, 33u)) * 64u;

    // The scope needs exactly four SIMD groups. Keep the 128-thread dispatch
    // correct on devices whose actual SIMD width differs from the measured 32.
    // This branch is uniform across the complete workgroup.
    if (width != 32u) {
        for (uint local = tid; local < 64u * 64u; local += 128u) {
            const uint row = mr + local / 64u;
            const uint column = nc + local % 64u;
            if (row >= rows || column >= columns) continue;
            float sum = param(params, 19u) != 0u ? raw_c[row] : 0.0f;
            for (uint k = 0u; k < inner; ++k)
                sum = sum + raw_b[row * inner + k]
                    * raw_a[(batch * inner + k) * columns + column];
            const uint index = (batch * rows + row) * columns + column;
            raw_output[index + param(params, 60u)] =
                epilogue(sum, index, raw_d, raw_e, params);
        }
        return;
    }

    // The sixth argument is relaxed_precision=false. Both operands and the
    // destination remain float throughout the MPP operation.
    constexpr auto descriptor = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;

    auto compute = [&](auto a, auto b) {
        auto accum = multiply.get_destination_cooperative_tensor<
            decltype(a), decltype(b), float>();
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < accum.get_capacity(); ++i) {
            if (accum.is_valid_element(i)) {
                const auto coord = accum.get_multidimensional_index(i);
                const uint row = mr + uint(coord[1]);
                accum[i] = param(params, 19u) != 0u && row < rows
                    ? raw_c[row] : 0.0f;
            }
        }
        multiply.run(a, b, accum);
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < accum.get_capacity(); ++i) {
            if (accum.is_valid_element(i)) {
                const auto coord = accum.get_multidimensional_index(i);
                const uint row = mr + uint(coord[1]);
                const uint column = nc + uint(coord[0]);
                if (row < rows && column < columns) {
                    const uint index = (batch * rows + row) * columns + column;
                    raw_output[index + param(params, 60u)] =
                        epilogue(accum[i], index, raw_d, raw_e, params);
                }
            }
        }
    };
    if (mr + 64u <= rows && nc + 64u <= columns) {
        using WA = tensor<device float, extents<int, dynamic_extent, 64>, tensor_inline>;
        using XB = tensor<device float, extents<int, 64, dynamic_extent>, tensor_inline>;
        compute(WA(const_cast<device float*>(raw_b) + mr * inner,
                   extents<int, dynamic_extent, 64>(int(inner)),
                   array<int, 2>({1, int(inner)})),
                XB(const_cast<device float*>(raw_a) + batch * inner * columns + nc,
                   extents<int, 64, dynamic_extent>(int(inner)),
                   array<int, 2>({1, int(columns)})));
    } else {
        using FloatMatrix = tensor<device float, dextents<int, 2>, tensor_inline>;
        FloatMatrix weights(const_cast<device float*>(raw_b),
                            dextents<int, 2>(int(inner), int(rows)),
                            array<int, 2>({1, int(inner)}));
        FloatMatrix activations(const_cast<device float*>(raw_a) + batch * inner * columns,
                                dextents<int, 2>(int(columns), int(inner)),
                                array<int, 2>({1, int(columns)}));
        compute(weights.slice(0, int(mr)), activations.slice(int(nc), 0));
    }
}
