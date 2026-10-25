// Optional Half-operand resident vocabulary projection. Admission requires
// explicit F16 matrix policy, actual Half features and its separate probe.
// The original heads_native.metal remains the strict F32 implementation.
// Resident Head ABI, unchanged buffers and parameter words:
//   0 features [rows,K] Half, 1 weights [K,classes] Half,
//   2 bias [classes] FP32, 3 partial [rows,ceil(classes/64)] uint4,
//   4 Params: uint4 words[64], words 0..4 = rows,K,classes,tiles,probability.
// Dispatch ceil(classes/64) x ceil(rows/32) workgroups of 128 threads.
// Each partial is [lowest winning class, bits(local maximum),
//                  bits(sum(exp(logit-local maximum))), invalid flag].
// Existing head_reduce.wgsl consumes this buffer without conversion or copies.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#if !defined(__HAVE_TENSOR__)
#error "MPP projection requires Metal tensor support"
#endif

using namespace metal;
using namespace mpp::tensor_ops;

struct Params { uint4 words[64]; };
inline uint hp(constant Params& params, uint i) {
    return params.words[i / 4u][i % 4u];
}

constant constexpr float head_limit = 3.402823e38f;

// Match the existing head's finite-score gate using bits so native compiler
// math modes cannot discard nonfinite detection under finite-only assumptions.
inline uint head_invalid(float value) {
    return uint((as_type<uint>(value) & 0x7fffffffu) > as_type<uint>(head_limit));
}

inline void head_choose(float candidate, uint index,
                        thread float& maximum, thread uint& winner) {
    if (candidate > maximum || (candidate == maximum && index < winner)) {
        maximum = candidate;
        winner = index;
    }
}

kernel void head_mpp32x64_f16(
    device const half* features [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device uint4* partial [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    const uint rows = hp(params, 0u);
    const uint inner = hp(params, 1u);
    const uint classes = hp(params, 2u);
    const uint tiles = hp(params, 3u);
    const uint first_row = group.y * 32u;
    const uint first_class = group.x * 64u;
    const bool probability = hp(params, 4u) != 0u;
    if (rows == 0u || inner == 0u || classes == 0u
        || first_row >= rows || first_class >= classes) return;

    // A four-simdgroup operation requires exactly four 32-lane groups here.
    // Other widths retain bounded GPU Half loads with explicit FP32 products.
    // Only 32 threads participate in this uncommon path; no collective or
    // threadgroup barrier occurs before its uniform return.
    if (width != 32u) {
        if (tid >= 32u || first_row + tid >= rows) return;
        const uint row = first_row + tid;
        float values[64];
        float maximum = -head_limit;
        uint winner = 0xffffffffu;
        uint invalid = 0u;
        for (uint local = 0u; local < 64u; ++local) {
            const uint c = first_class + local;
            float value = -head_limit;
            if (c < classes) {
                value = 0.0f;
                for (uint k = 0u; k < inner; ++k)
                    value = value + float(features[row * inner + k]) * float(weights[k * classes + c]);
                value = value + bias[c];
                invalid |= head_invalid(value);
                head_choose(value, c, maximum, winner);
            }
            values[local] = value;
        }
        float sum = 0.0f;
        if (probability) {
            for (uint local = 0u; local < 64u && first_class + local < classes; ++local)
                sum = sum + exp(values[local] - maximum);
        }
        partial[row * tiles + group.x] =
            uint4(winner, as_type<uint>(maximum), as_type<uint>(sum), invalid);
        return;
    }

    // Explicit relaxed_precision=false: Half sources, FP32 accumulators/scores.
    // Bias remains the original F32 constant and is added after the matrix dot.
    constexpr auto descriptor = matmul2d_descriptor(
        32, 64, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    threadgroup float scores[32u * 64u];
    for (uint i = tid; i < 32u * 64u; i += 128u) scores[i] = -head_limit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto compute = [&](auto a, auto b) {
        auto accum = multiply.get_destination_cooperative_tensor<
            decltype(a), decltype(b), float>();
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < accum.get_capacity(); ++i) {
            if (accum.is_valid_element(i)) accum[i] = 0.0f;
        }
        multiply.run(a, b, accum);
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < accum.get_capacity(); ++i) {
            if (accum.is_valid_element(i)) {
                auto coord = accum.get_multidimensional_index(i);
                const uint row = uint(coord[1]);
                const uint column = uint(coord[0]);
                if (first_row + row < rows && first_class + column < classes)
                    scores[row * 64u + column] =
                        accum[i] + bias[first_class + column];
            }
        }
    };

    if (first_row + 32u <= rows && first_class + 64u <= classes) {
        // Static outer dimensions remove tensor-edge checks on interior tiles.
        // Strides address the original resident matrices directly.
        using A = tensor<device half, extents<int, dynamic_extent, 32>, tensor_inline>;
        using B = tensor<device half, extents<int, 64, dynamic_extent>, tensor_inline>;
        compute(A(const_cast<device half*>(features) + first_row * inner,
                  extents<int, dynamic_extent, 32>(int(inner)),
                  array<int, 2>({1, int(inner)})),
                B(const_cast<device half*>(weights) + first_class,
                  extents<int, 64, dynamic_extent>(int(inner)),
                  array<int, 2>({1, int(classes)})));
    } else {
        // Tensor-offset slices retain the full extents, so MPP supplies zero
        // outside the original matrices. Stores above mask both outer tails.
        using Matrix = tensor<device half, dextents<int, 2>, tensor_inline>;
        Matrix a(const_cast<device half*>(features),
                 dextents<int, 2>(int(inner), int(rows)),
                 array<int, 2>({1, int(inner)}));
        Matrix b(const_cast<device half*>(weights),
                 dextents<int, 2>(int(classes), int(inner)),
                 array<int, 2>({1, int(classes)}));
        compute(a.slice(0, int(first_row)), b.slice(int(first_class), 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One simdgroup reduces one row at a time. Contiguous 32-lane loads
    // avoid shared-memory bank conflicts from interleaving many rows per SG.
    // Each lane owns two classes; no index/sum/invalid workgroup arrays exist.
    for (uint row = sg; row < 32u; row += 4u) {
        float maximum = -head_limit;
        uint winner = 0xffffffffu;
        uint invalid = 0u;
        const uint c0 = first_class + lane;
        const uint c1 = c0 + 32u;
        const float v0 = scores[row * 64u + lane];
        const float v1 = scores[row * 64u + lane + 32u];
        if (c0 < classes) {
            invalid |= head_invalid(v0);
            head_choose(v0, c0, maximum, winner);
        }
        if (c1 < classes) {
            invalid |= head_invalid(v1);
            head_choose(v1, c1, maximum, winner);
        }
        for (uint mask = 16u; mask != 0u; mask >>= 1u) {
            const float other = simd_shuffle_xor(maximum, mask);
            const uint index = simd_shuffle_xor(winner, mask);
            head_choose(other, index, maximum, winner);
            invalid |= simd_shuffle_xor(invalid, mask);
        }
        float sum = 0.0f;
        if (probability) {
            if (c0 < classes) sum += exp(v0 - maximum);
            if (c1 < classes) sum += exp(v1 - maximum);
            sum = simd_sum(sum);
        }
        if (lane == 0u && first_row + row < rows)
            partial[(first_row + row) * tiles + group.x] =
                uint4(winner, as_type<uint>(maximum), as_type<uint>(sum), invalid);
    }
}
