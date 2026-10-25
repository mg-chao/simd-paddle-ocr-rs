// Explicit fast-mode pointwise convolution. common_native.metal precedes this
// source: parameter, bias, residual, output and epilogue storage remain FP32.
// Binding0: physical NHWC half[N,HW,Cin]. Binding1: K-major half[Cin,CoutPad64].
// Four SIMD groups / 128 threads compute one Cout x HW tile of 64 x 64.
// words32/33=ceil(Cout/64),ceil(HW/64);34=grid.x;35=batch;36=CoutPad64.
// Input prescaling is folded into the parent's input pack before conversion.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#if !defined(__HAVE_TENSOR__)
#error "FP16 pointwise requires Metal tensor support"
#endif

using namespace mpp::tensor_ops;

kernel void pointwise_f16(
    device const half* raw_a [[buffer(0)]],
    device const half* raw_b [[buffer(1)]],
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
    const uint padded_rows = param(params, 36u);
    const uint block = group.x + group.y * param(params, 34u);
    const uint batch = block / (param(params, 32u) * param(params, 33u));
    if (batch >= param(params, 35u)) return;
    const uint mr = ((block / param(params, 33u)) % param(params, 32u)) * 64u;
    const uint nc = (block % param(params, 33u)) * 64u;

    // This branch is uniform over the workgroup. Exactly four 32-lane SIMD
    // groups are required by the collective MPP path; all other widths retain
    // bounded scalar GPU execution over the same already converted operands.
    if (width != 32u) {
        for (uint local = tid; local < 64u * 64u; local += 128u) {
            const uint row = mr + local / 64u;
            const uint column = nc + local % 64u;
            if (row >= rows || column >= columns) continue;
            float sum = param(params, 19u) != 0u ? raw_c[row] : 0.0f;
            for (uint k = 0u; k < inner; ++k)
                sum = sum + float(raw_b[k * padded_rows + row])
                    * float(raw_a[(batch * columns + column) * inner + k]);
            const uint index = (batch * rows + row) * columns + column;
            raw_output[index + param(params, 60u)] =
                epilogue(sum, index, raw_d, raw_e, params);
        }
        return;
    }

    // Both physical operands are transposes of canonical NCHW/OIHW matrices.
    // Their half values accumulate into float without relaxed MPP arithmetic.
    constexpr auto descriptor = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), true, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    auto compute = [&](auto weights, auto activations) {
        auto accum = multiply.get_destination_cooperative_tensor<
            decltype(weights), decltype(activations), float>();
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < accum.get_capacity(); ++i) {
            if (accum.is_valid_element(i)) {
                const auto coord = accum.get_multidimensional_index(i);
                const uint row = mr + uint(coord[1]);
                accum[i] = param(params, 19u) != 0u && row < rows
                    ? raw_c[row] : 0.0f;
            }
        }
        multiply.run(weights, activations, accum);
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
        using WA = tensor<device half, extents<int, 64, dynamic_extent>, tensor_inline>;
        using XB = tensor<device half, extents<int, dynamic_extent, 64>, tensor_inline>;
        compute(WA(const_cast<device half*>(raw_b) + mr,
                   extents<int, 64, dynamic_extent>(int(inner)),
                   array<int, 2>({1, int(padded_rows)})),
                XB(const_cast<device half*>(raw_a) + (batch * columns + nc) * inner,
                   extents<int, dynamic_extent, 64>(int(inner)),
                   array<int, 2>({1, int(inner)})));
    } else {
        using HalfMatrix = tensor<device half, dextents<int, 2>, tensor_inline>;
        HalfMatrix weights(const_cast<device half*>(raw_b),
                           dextents<int, 2>(int(rows), int(inner)),
                           array<int, 2>({1, int(padded_rows)}));
        HalfMatrix activations(const_cast<device half*>(raw_a) + batch * columns * inner,
                               dextents<int, 2>(int(inner), int(columns)),
                               array<int, 2>({1, int(inner)}));
        compute(weights.slice(int(mr), 0), activations.slice(0, int(nc)));
    }
}
