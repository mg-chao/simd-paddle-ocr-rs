#define SCRATCH_FLOATS 4352
void linear_store(float4 value, uint row, uint column) const {
  if (row >= p(3))
    return;
  for (uint j = 0; j < 4 && column + j < p(5); ++j) {
    uint index = row * p(5) + column + j;
    float v = output_round(value[j]);
    if (p(19))
      v = output_round(v + load_c(column + j));
    if (p(21))
      v = output_round(v + load_d(index));
    if (p(20))
      v *= output_round(sigmoid(v));
    store_output(index, v);
  }
}
void run(uint3, uint3 group, uint lane, threadgroup float *partial) const {
  uint block = group.x + group.y * p(1), partition = lane / 16,
       local = lane % 16;
  bool split = p(23) != 0;
  uint tile = split ? block * 16 + local : block * 256 + lane;
  uint column_tiles = (p(5) + 3) / 4, row = (tile / column_tiles) * 4,
       column = (tile % column_tiles) * 4;
  bool live = tile < p(0);
  float4 accum[4] = {float4(0), float4(0), float4(0), float4(0)};
  if (live)
    for (uint k4 = split ? partition : 0; k4 < (p(4) + 3) / 4;
         k4 += split ? 16 : 1) {
      for (uint k = k4 * 4; k < min(k4 * 4 + 4, p(4)); ++k) {
        float4 weight(0);
        for (uint j = 0; j < 4 && column + j < p(5); ++j)
          weight[j] = float(half(load_b(k * p(5) + column + j)));
        for (uint i = 0; i < 4 && row + i < p(3); ++i)
          accum[i] += float(half(load_a((row + i) * p(4) + k))) * weight;
      }
    }
  if (split) {
    for (uint i = 0; i < 4; ++i)
      for (uint j = 0; j < 4; ++j)
        partial[lane * 17 + i * 4 + j] = accum[i][j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint k = 0; k < 16; ++k)
      sum += partial[(k * 16 + local) * 17 + partition];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial[local * 17 + partition] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && partition < 4) {
      uint offset = local * 17 + partition * 4;
      linear_store(float4(partial[offset], partial[offset + 1],
                          partial[offset + 2], partial[offset + 3]),
                   row + partition, column);
    }
  } else if (live)
    for (uint i = 0; i < 4; ++i)
      linear_store(accum[i], row + i, column);
}
