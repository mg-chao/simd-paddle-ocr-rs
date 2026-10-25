#include <metal_stdlib>
using namespace metal;

// Shared graph ABI. Five canonical read operands, output, 128 packet words.
// A=[rows,K], B=[K,N], C=optional last-axis bias, D=optional residual.
// Flattening leading A dimensions changes no storage or mathematical batch.
struct NativeParameters { uint4 words[32]; };
inline uint param(constant NativeParameters& p, uint i) {
    return p.words[i / 4u][i % 4u];
}
inline float ln_round(float value) { return $ROUND; }

inline float4 ln_operands(device const $STORAGE* data, uint offset, uint available) {
    float4 value = float4(0.0f);
    if (available >= 4u) {
        // Packed types have element alignment, including odd K/N row strides.
        value = float4(*(device const $PACKED*)(data + offset));
    } else {
        for (uint i = 0u; i < available; ++i) value[i] = float(data[offset + i]);
    }
    // Explicit fast-matrix precision is independent of backing storage.
    return float4(half4(value));
}

inline void ln_accumulate(device const $STORAGE* a, device const $STORAGE* b,
                          uint row, uint column, uint rows, uint inner, uint columns,
                          uint first, uint step,
                          thread float4& a0, thread float4& a1,
                          thread float4& a2, thread float4& a3) {
    for (uint k4 = first; k4 < (inner + 3u) / 4u; k4 += step) {
        uint k0 = k4 * 4u;
        uint count = min(4u, inner - k0);
        float4 x0 = row < rows ? ln_operands(a, row * inner + k0, count) : float4(0.0f);
        float4 x1 = row + 1u < rows ? ln_operands(a, (row + 1u) * inner + k0, count) : float4(0.0f);
        float4 x2 = row + 2u < rows ? ln_operands(a, (row + 2u) * inner + k0, count) : float4(0.0f);
        float4 x3 = row + 3u < rows ? ln_operands(a, (row + 3u) * inner + k0, count) : float4(0.0f);
        for (uint i = 0u; i < count; ++i) {
            float4 w = ln_operands(b, (k0 + i) * columns + column, min(4u, columns - column));
            a0 += x0[i] * w; a1 += x1[i] * w;
            a2 += x2[i] * w; a3 += x3[i] * w;
        }
    }
}

inline void ln_store(float4 value, uint row, uint column,
                     device const $STORAGE* bias, device const $STORAGE* residual,
                     device $STORAGE* output, constant NativeParameters& p) {
    uint rows = param(p, 3u), columns = param(p, 5u);
    if (row >= rows) return;
    for (uint i = 0u; i < 4u && column + i < columns; ++i) {
        uint index = row * columns + column + i;
        float v = ln_round(value[i]); // Original MatMul output store.
        if (param(p, 19u) != 0u) v = ln_round(v + float(bias[column + i]));
        if (param(p, 21u) != 0u) v = ln_round(v + float(residual[index]));
        if (param(p, 20u) != 0u) {
            float s;
            if (v >= 0.0f) s = 1.0f / (1.0f + exp(-v));
            else { float e = exp(v); s = e / (1.0f + e); }
            v = v * ln_round(s); // Original Sigmoid store before Mul.
        }
        output[index + param(p, 60u)] = $STORAGE(v);
    }
}

kernel void linear_native(
    device const $STORAGE* raw_a [[buffer(0)]],
    device const $STORAGE* raw_b [[buffer(1)]],
    device const $STORAGE* raw_c [[buffer(2)]],
    device const $STORAGE* raw_d [[buffer(3)]],
    device const $STORAGE* raw_e [[buffer(4)]],
    device $STORAGE* raw_output [[buffer(5)]],
    constant NativeParameters& p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    uint rows = param(p, 3u), inner = param(p, 4u), columns = param(p, 5u);
    threadgroup float partial[256 * 17];
    uint group_index = group.x + group.y * param(p, 1u);
    uint column_tiles = (columns + 3u) / 4u;
    bool split = param(p, 23u) != 0u;
    uint local_tile = tid % 16u, partition = tid / 16u;
    uint tile = split ? group_index * 16u + local_tile : group_index * 256u + tid;
    bool live = tile < param(p, 0u);
    uint row = live ? (tile / column_tiles) * 4u : 0u;
    uint column = live ? (tile % column_tiles) * 4u : 0u;
    float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
    if (live) ln_accumulate(raw_a, raw_b, row, column, rows, inner, columns,
                            split ? partition : 0u, split ? 16u : 1u,
                            a0, a1, a2, a3);
    if (split) {
        // 16 independent tiles × 16 K partitions. Padding avoids bank aliases.
        // All invocations, including inactive tail tiles, reach every barrier.
        uint offset = tid * 17u;
        for (uint i = 0u; i < 4u; ++i) {
            partial[offset + i] = a0[i]; partial[offset + 4u + i] = a1[i];
            partial[offset + 8u + i] = a2[i]; partial[offset + 12u + i] = a3[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        for (uint k = 0u; k < 16u; ++k)
            sum += partial[(k * 16u + local_tile) * 17u + partition];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        partial[local_tile * 17u + partition] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live && partition < 4u) {
            uint source = local_tile * 17u + partition * 4u;
            float4 value = float4(partial[source], partial[source + 1u],
                                  partial[source + 2u], partial[source + 3u]);
            ln_store(value, row + partition, column, raw_c, raw_d, raw_output, p);
        }
    } else if (live) {
        ln_store(a0, row, column, raw_c, raw_d, raw_output, p);
        ln_store(a1, row + 1u, column, raw_c, raw_d, raw_output, p);
        ln_store(a2, row + 2u, column, raw_c, raw_d, raw_output, p);
        ln_store(a3, row + 3u, column, raw_c, raw_d, raw_output, p);
    }
}
