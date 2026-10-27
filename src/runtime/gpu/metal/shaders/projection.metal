// A 32-row x 64-class tile; only winners and softmax partials leave the tile.
kernel void projection(device const MATRIX *features [[buffer(0)]],
                       device const MATRIX *weights [[buffer(1)]],
                       device const float *bias [[buffer(2)]],
                       device uint4 *partial [[buffer(3)]],
                       constant Params &params [[buffer(4)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint2 group [[threadgroup_position_in_grid]]) {
  uint rows = params.words[0].x, inner = params.words[0].y, classes = params.words[0].z;
  uint tiles = params.words[0].w, probability = params.words[1].x;
  threadgroup MATRIX left[32 * 16], right[16 * 64];
  threadgroup float values[32 * 64];
  uint sg = lane / 32, sm = (sg / 2) * 16, sn = (sg % 2) * 32;
  simdgroup_float8x8 acc[2][4];
#pragma unroll
  for (uint r = 0; r < 2; ++r)
#pragma unroll
    for (uint c = 0; c < 4; ++c) acc[r][c] = simdgroup_float8x8(0.0f);
  for (uint base = 0; base < inner; base += 16) {
    for (uint t = lane; t < 32 * 16; t += 128) {
      uint row = group.y * 32 + t / 16, k = base + t % 16;
      left[t] = row < rows && k < inner ? features[row * inner + k] : MATRIX(0.0f);
    }
    for (uint t = lane; t < 16 * 64; t += 128) {
      uint col = group.x * 64 + t % 64, k = base + t / 64;
      right[t] = col < classes && k < inner ? weights[k * classes + col] : MATRIX(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint k = 0; k < 16; k += 8) {
      MATRIX8 a[2], b[4];
#pragma unroll
      for (uint r = 0; r < 2; ++r) simdgroup_load(a[r], left + (sm + r * 8) * 16 + k, 16);
#pragma unroll
      for (uint c = 0; c < 4; ++c) simdgroup_load(b[c], right + k * 64 + sn + c * 8, 64);
#pragma unroll
      for (uint r = 0; r < 2; ++r)
#pragma unroll
        for (uint c = 0; c < 4; ++c) simdgroup_multiply_accumulate(acc[r][c], a[r], b[c], acc[r][c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (uint r = 0; r < 2; ++r)
#pragma unroll
    for (uint c = 0; c < 4; ++c) simdgroup_store(acc[r][c], values + (sm + r * 8) * 64 + sn + c * 8, 64);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sl = lane % 32;
  for (uint r = sg; r < 32; r += 4) {
    uint row = group.y * 32 + r;
    if (row >= rows) break;
    uint c0 = group.x * 64 + sl, c1 = c0 + 32;
    float a = c0 < classes ? values[r * 64 + sl] + bias[c0] : -FLT_MAX;
    float b = c1 < classes ? values[r * 64 + sl + 32] + bias[c1] : -FLT_MAX;
    uint invalid = simd_or(uint(!isfinite(a) || !isfinite(b)));
    float maximum = simd_max(max(a, b));
    uint index = simd_min(min(a == maximum ? c0 : 0xffffffffu, b == maximum ? c1 : 0xffffffffu));
    float sum = probability ? simd_sum((c0 < classes ? exp(a - maximum) : 0.0f) + (c1 < classes ? exp(b - maximum) : 0.0f)) : 0.0f;
    if (sl == 0) partial[row * tiles + group.x] = uint4(index, as_type<uint>(maximum), as_type<uint>(sum), invalid);
  }
}
