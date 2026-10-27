// Canonical NCHW implicit convolution and batched matrix multiplication.
// SIMD groups share input tiles, with FP32 accumulators throughout.
#if IS_CONV
#define INNER_TILE 32
#else
#define INNER_TILE 16
#endif
#define STAGING_FLOATS (TILE * 2 * (INNER_TILE + 1) * sizeof(MATRIX) / sizeof(float))
#define SCRATCH_FLOATS ((TILE * TILE > STAGING_FLOATS) ? TILE * TILE : STAGING_FLOATS)
#if TILE == 64
#define TILE_THREADS 256
#define SG_ROWS 16
#define SG_COLS 32
#else
#define TILE_THREADS 64
#define SG_ROWS (TILE / 2)
#define SG_COLS TILE
#endif
void run(uint3, uint3 group, uint lane, threadgroup float *scratch) const {
  threadgroup MATRIX *left = (threadgroup MATRIX *)scratch;
  threadgroup MATRIX *right = left + TILE * (INNER_TILE + 1);
  threadgroup float *result = scratch; // staging is dead before accumulator stores
  uint block = group.x + group.y * p(34), batch = block / (p(32) * p(33));
  if (batch >= p(35)) return;
  uint mr = (block / p(33)) % p(32) * TILE, nc = block % p(33) * TILE;
  uint rows = IS_CONV ? p(4) : p(3), inner = IS_CONV ? p(9) * p(11) * p(12) : p(4);
  uint columns = IS_CONV ? p(7) * p(8) : p(5);
  uint ai = 0, bi = 0;
  if (!IS_CONV) {
    uint remaining = batch;
    for (uint axis = p(6); axis > 0;) {
      --axis;
      uint coordinate = remaining % p(8 + axis); remaining /= p(8 + axis);
      ai += coordinate * p(16 + axis); bi += coordinate * p(24 + axis);
    }
  }
  uint sm = ((lane / 32) % (TILE / SG_ROWS)) * SG_ROWS;
  uint sn = ((lane / 32) / (TILE / SG_ROWS)) * SG_COLS;
  simdgroup_float8x8 sums[SG_ROWS / 8][SG_COLS / 8];
#pragma unroll
  for (uint r = 0; r < SG_ROWS / 8; ++r)
#pragma unroll
    for (uint c = 0; c < SG_COLS / 8; ++c) sums[r][c] = simdgroup_float8x8(0.0f);
#if DIRECT_MATRIX
  // Full left-hand rows can load device fragments directly. A right-hand
  // column tail is allowed only when binding-time range checks prove that
  // the unused fragment cells stay inside an initialized backing allocation.
  // A 1x1 kernel with bottom/right padding still needs spatial coordinates;
  // its output plane is not a valid matrix row stride for the input.
  bool pointwise = !IS_CONV || (p(11) == 1 && p(12) == 1 && p(13) == 1 && p(14) == 1 && p(17) == 0 && p(18) == 0 && p(5) == p(7) && p(6) == p(8));
  if (pointwise && mr + TILE <= rows && (nc + TILE <= columns || p(61)) && inner % 16 == 0) {
#if IS_CONV
    device const MATRIX *aa = b;
    device const MATRIX *bb = a + batch * inner * columns;
#else
    device const MATRIX *aa = a + ai;
    device const MATRIX *bb = b + bi;
#endif
    for (uint base = 0; base < inner; base += 16) {
#pragma unroll
      for (uint k = 0; k < 16; k += 8) {
        MATRIX8 lhs[SG_ROWS / 8], rhs[SG_COLS / 8];
#pragma unroll
        for (uint r = 0; r < SG_ROWS / 8; ++r)
          simdgroup_load(lhs[r], aa + (mr + sm + r * 8) * inner + base + k, inner);
#pragma unroll
        for (uint c = 0; c < SG_COLS / 8; ++c)
          simdgroup_load(rhs[c], bb + (base + k) * columns + nc + sn + c * 8, columns);
#pragma unroll
        for (uint r = 0; r < SG_ROWS / 8; ++r)
#pragma unroll
          for (uint c = 0; c < SG_COLS / 8; ++c)
            simdgroup_multiply_accumulate(sums[r][c], lhs[r], rhs[c], sums[r][c]);
      }
    }
  } else
#endif
  for (uint base = 0; base < inner; base += INNER_TILE) {
    for (uint t = lane; t < TILE * INNER_TILE; t += TILE_THREADS) {
      uint row = mr + t / INNER_TILE, k = base + t % INNER_TILE;
      float value = 0.0f;
      if (row < rows && k < inner)
        value = IS_CONV ? load_b(row * inner + k) : load_a(ai + row * inner + k);
      left[(t / INNER_TILE) * (INNER_TILE + 1) + t % INNER_TILE] = MATRIX(value);
    }
    // Share tap/spatial indexing across four adjacent matrix columns. OCR
    // widths are usually multiples of four; the bounded scalar border also
    // handles general shapes and rows that cross an output-image boundary.
    for (uint t = lane; t < (TILE / 4) * INNER_TILE; t += TILE_THREADS) {
      uint local_col = (t % (TILE / 4)) * 4, rk = base + t / (TILE / 4);
      uint col = nc + local_col;
      float4 values = float4(0.0f);
      if (rk < inner) {
        if (IS_CONV && p(11) == 1 && p(12) == 1 && p(13) == 1 && p(14) == 1 && p(17) == 0 && p(18) == 0 && p(5) == p(7) && p(6) == p(8)) {
#pragma unroll
          for (uint j = 0; j < 4; ++j)
            if (col + j < columns) values[j] = load_a((batch * p(3) + rk) * columns + col + j);
        } else if (IS_CONV) {
          uint taps = p(11) * p(12), ci = rk / taps, ky = (rk % taps) / p(12), kx = rk % p(12);
          uint row = col / p(8), ox = col % p(8);
          bool same_row = ox + 3 < p(8);
          int y0 = int(row * p(13) + ky) - int(p(17));
          int x0 = int(ox * p(14) + kx) - int(p(18));
#pragma unroll
          for (uint j = 0; j < 4; ++j) {
            uint pixel = col + j;
            int y = same_row ? y0 : int((pixel / p(8)) * p(13) + ky) - int(p(17));
            int x = same_row ? x0 + int(j * p(14)) : int((pixel % p(8)) * p(14) + kx) - int(p(18));
            if (pixel < columns && y >= 0 && x >= 0 && y < int(p(5)) && x < int(p(6)))
              values[j] = load_a(((batch * p(3) + ci) * p(5) + uint(y)) * p(6) + uint(x));
          }
        } else {
#pragma unroll
          for (uint j = 0; j < 4; ++j)
            if (col + j < columns) values[j] = load_b(bi + rk * columns + col + j);
        }
      }
#pragma unroll
      for (uint j = 0; j < 4; ++j)
        right[(local_col + j) * (INNER_TILE + 1) + t / (TILE / 4)] = MATRIX(values[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint k = 0; k < INNER_TILE; k += 8) {
      MATRIX8 lhs[SG_ROWS / 8], rhs[SG_COLS / 8];
#pragma unroll
      for (uint r = 0; r < SG_ROWS / 8; ++r) simdgroup_load(lhs[r], left + (sm + r * 8) * (INNER_TILE + 1) + k, INNER_TILE + 1);
#pragma unroll
      for (uint c = 0; c < SG_COLS / 8; ++c) simdgroup_load(rhs[c], right + (sn + c * 8) * (INNER_TILE + 1) + k, INNER_TILE + 1, ulong2(0, 0), true);
#pragma unroll
      for (uint r = 0; r < SG_ROWS / 8; ++r)
#pragma unroll
        for (uint c = 0; c < SG_COLS / 8; ++c) simdgroup_multiply_accumulate(sums[r][c], lhs[r], rhs[c], sums[r][c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (uint r = 0; r < SG_ROWS / 8; ++r)
#pragma unroll
    for (uint c = 0; c < SG_COLS / 8; ++c) simdgroup_store(sums[r][c], result + (sm + r * 8) * TILE + sn + c * 8, TILE);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < TILE * TILE; i += TILE_THREADS) {
    uint row = mr + i / TILE, col = nc + i % TILE;
    if (row < rows && col < columns) {
      uint index = (batch * rows + row) * columns + col;
      float value = result[i] + ((IS_CONV && p(19)) ? load_c(row) : 0.0f);
      store_output(index, epilogue<IS_CONV, !IS_CONV>(value, index, row));
    }
  }
}
