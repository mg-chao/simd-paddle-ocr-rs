#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if !defined(__HAVE_TENSOR__)
#error "MPP token projection requires Metal tensor support"
#endif
using namespace metal;
using namespace mpp::tensor_ops;

struct NativeParameters { uint4 words[32]; };
inline uint param(constant NativeParameters& p, uint i) { return p.words[i / 4u][i % 4u]; }

// Every removed graph store is retained; bias must not initialize the GEMM.
inline void ln_store_scalar(float value, uint row, uint column,
    device const half* bias, device const half* residual,
    device half* output, constant NativeParameters& p) {
    const uint index = row * param(p, 5u) + column;
    float v = round_to_half(value);
    if (param(p, 19u) != 0u) v = round_to_half(v + float(bias[column]));
    if (param(p, 21u) != 0u) v = round_to_half(v + float(residual[index]));
    if (param(p, 20u) != 0u) {
        float s;
        if (v >= 0.0f) s = 1.0f / (1.0f + exp(-v));
        else { float e = exp(v); s = e / (1.0f + e); }
        v = v * round_to_half(s);
    }
    output[index + param(p, 60u)] = half(v);
}

kernel void linear_mpp(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device const half* residual [[buffer(3)]],
    device const half* unused [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant NativeParameters& p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_simdgroup]]) {
    const uint rows = param(p, 3u), inner = param(p, 4u), columns = param(p, 5u);
    const uint block = group.x + group.y * param(p, 1u);
    const uint column_tiles = (columns + 63u) / 64u;
    const uint first_row = (block / column_tiles) * 32u;
    const uint first_column = (block % column_tiles) * 64u;
    if (first_row >= rows || first_column >= columns) return;
    // Word24 only permits the standalone proof to force the alternate-width
    // branch. All measured commands use zero and the actual SIMD width.
    if (width != 32u || param(p, 24u) != 0u) {
        for (uint local = tid; local < 32u * 64u; local += 128u) {
            const uint row = first_row + local / 64u;
            const uint column = first_column + local % 64u;
            if (row >= rows || column >= columns) continue;
            float value = 0.0f;
            for (uint k = 0u; k < inner; ++k)
                value = value + float(a[row * inner + k]) * float(b[k * columns + column]);
            ln_store_scalar(value, row, column, bias, residual, output, p);
        }
        return;
    }
    constexpr auto descriptor = matmul2d_descriptor(
        32, 64, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    auto compute = [&](auto left, auto right) {
        auto accum = multiply.get_destination_cooperative_tensor<
            decltype(left), decltype(right), float>();
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < accum.get_capacity(); ++i)
            if (accum.is_valid_element(i)) accum[i] = 0.0f;
        multiply.run(left, right, accum);
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < accum.get_capacity(); ++i) {
            if (accum.is_valid_element(i)) {
                const auto coord = accum.get_multidimensional_index(i);
                const uint row = first_row + uint(coord[1]);
                const uint column = first_column + uint(coord[0]);
                if (row < rows && column < columns)
                    ln_store_scalar(accum[i], row, column, bias, residual, output, p);
            }
        }
    };
    if (first_row + 32u <= rows && first_column + 64u <= columns) {
        using A = tensor<device half, extents<int, dynamic_extent, 32>, tensor_inline>;
        using B = tensor<device half, extents<int, 64, dynamic_extent>, tensor_inline>;
        compute(A(const_cast<device half*>(a) + first_row * inner,
                  extents<int, dynamic_extent, 32>(int(inner)), array<int, 2>({1, int(inner)})),
                B(const_cast<device half*>(b) + first_column,
                  extents<int, 64, dynamic_extent>(int(inner)), array<int, 2>({1, int(columns)})));
    } else {
        using Matrix = tensor<device half, dextents<int, 2>, tensor_inline>;
        Matrix left(const_cast<device half*>(a), dextents<int, 2>(int(inner), int(rows)),
                    array<int, 2>({1, int(inner)}));
        Matrix right(const_cast<device half*>(b), dextents<int, 2>(int(columns), int(inner)),
                     array<int, 2>({1, int(columns)}));
        compute(left.slice(0, int(first_row)), right.slice(int(first_column), 0));
    }
}
