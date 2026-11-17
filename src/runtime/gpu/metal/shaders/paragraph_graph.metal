kernel void paragraph_graph(device const float *a [[buffer(0)]],
                            device const float *b [[buffer(1)]],
                            device const float *c [[buffer(2)]],
                            device const uint *indices [[buffer(3)]],
                            device float *output [[buffer(4)]],
                            constant Params &params [[buffer(5)]],
                            uint3 position [[thread_position_in_grid]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float mass;
    uint mode = params.words[0].x;
    uint count = params.words[0].y;
    uint stride = params.words[0].z;
    if (mode == 1u) {
        uint row = group.x + group.y * stride;
        if (row >= count) return;
        uint first = indices[params.words[0].w + row];
        uint last = indices[params.words[0].w + row + 1u];
        uint segment = params.words[1].x;
        if (lane == 0u) {
            float sum = 0.0f;
            for (uint i = first; i < last; ++i) sum += b[indices[segment + i]];
            mass = max(sum, 1e-12f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint col = lane; col < 96u; col += 64u) {
            float value = 0.0f;
            for (uint i = first; i < last; ++i) {
                uint edge = indices[segment + i];
                value += a[edge * 96u + col] * (b[edge] / mass);
            }
            output[row * 192u + col] = c[row * 96u + col];
            output[row * 192u + col + 96u] = value;
        }
        return;
    }
    uint id = position.x + position.y * stride;
    if (id >= count) return;
    if (mode == 0u) {
        uint row = id / 336u;
        uint col = id % 336u;
        if (col < 192u) {
            uint endpoint = indices[2u * row + col / 96u];
            output[id] = a[endpoint * 96u + col % 96u];
        } else {
            output[id] = b[row * 144u + col - 192u];
        }
    } else if (mode == 2u) {
        float x = a[id];
        output[id] = x >= 0.0f ? x : x * 0.2f;
    } else if (mode == 3u) {
        output[id] = exp(clamp(a[id], -10.0f, 10.0f));
    } else {
        output[id] = (a[id] + a[id + count]) * 0.5f;
    }
}
