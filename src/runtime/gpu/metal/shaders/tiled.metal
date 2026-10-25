#define SCRATCH_FLOATS (TILE * 32)
void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  threadgroup float *left = scratch;
  threadgroup float *right = scratch + TILE * 16;
  uint block = group.x + group.y * p(34), batch = block / (p(32) * p(33));
  if (batch >= p(35))
    return;
  uint mr = (block / p(33)) % p(32) * TILE, nc = block % p(33) * TILE;
  uint rows = IS_CONV ? p(4) : p(3),
       inner = IS_CONV ? p(9) * p(11) * p(12) : p(4),
       columns = IS_CONV ? p(7) * p(8) : p(5);
  uint ai = 0, bi = 0;
  if (!IS_CONV) {
    uint remaining = batch;
    for (uint axis = p(6); axis > 0;) {
      axis--;
      uint coordinate = remaining % p(8 + axis);
      remaining /= p(8 + axis);
      ai += coordinate * p(16 + axis);
      bi += coordinate * p(24 + axis);
    }
  }
  uint lr = lane / TILE, lc = lane % TILE;
  float sums[TILE * TILE / 64];
#pragma unroll
  for (uint r = 0; r < TILE * TILE / 64; r++)
    sums[r] = (IS_CONV && p(19) && mr + lr + r * 64 / TILE < rows)
                  ? load_c(mr + lr + r * 64 / TILE)
                  : 0.0f;
  for (uint base = 0; base < inner; base += 16) {
    for (uint t = lane; t < TILE * 16; t += 64) {
      uint row = mr + t / 16, k = base + t % 16;
      float value = 0.0f;
      if (row < rows && k < inner)
        value =
            IS_CONV ? load_b(row * inner + k) : load_a(ai + row * inner + k);
      left[t] = value;
      uint col = nc + t % TILE, rk = base + t / TILE;
      value = 0.0f;
      if (col < columns && rk < inner) {
        if (IS_CONV) {
          uint taps = p(11) * p(12), ci = rk / taps;
          int y = int((col / p(8)) * p(13) + (rk % taps) / p(12)) - int(p(17)),
              x = int((col % p(8)) * p(14) + rk % p(12)) - int(p(18));
          if (y >= 0 && x >= 0 && y < int(p(5)) && x < int(p(6)))
            value =
                load_a(((batch * p(3) + ci) * p(5) + uint(y)) * p(6) + uint(x));
        } else
          value = load_b(bi + rk * columns + col);
      }
      right[t] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < min(16u, inner - base); k++) {
      float value = right[k * TILE + lc];
#pragma unroll
      for (uint r = 0; r < TILE * TILE / 64; r++)
        sums[r] += left[(lr + r * 64 / TILE) * 16 + k] * value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (uint r = 0; r < TILE * TILE / 64; r++) {
    uint row = mr + lr + r * 64 / TILE, col = nc + lc;
    if (row < rows && col < columns) {
      uint index = (batch * rows + row) * columns + col;
      store_output(index, epilogue(sums[r], index));
    }
  }
}
