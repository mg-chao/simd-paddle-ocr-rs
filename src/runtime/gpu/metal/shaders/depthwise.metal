// Adjacent output pixels reuse the same weights. Every tail and each source
// coordinate is checked before loading; no padding relies on speculative reads.
void run(uint3 id, uint3, uint, threadgroup float *) const {
  uint tile = id.x + id.y * p(1), row_tiles = (p(8) + 3) / 4,
       row = tile / row_tiles;
  if (row >= p(0) / p(8))
    return;
  uint x0 = (tile % row_tiles) * 4, channel = (row / p(7)) % p(4),
       batch = row / (p(7) * p(4));
  int y0 = int((row % p(7)) * p(13)) - int(p(17));
  float4 sum = float4(p(19) ? load_c(channel) : 0.0f);
  for (uint ky = 0; ky < p(11); ++ky) {
    int y = y0 + int(ky * p(15));
    if (y < 0 || y >= int(p(5)))
      continue;
    for (uint kx = 0; kx < p(12); ++kx) {
      float weight = load_b((channel * p(11) + ky) * p(12) + kx);
      for (uint j = 0; j < 4; ++j) {
        int x = int((x0 + j) * p(14) + kx * p(16)) - int(p(18));
        if (x0 + j < p(8) && x >= 0 && x < int(p(6)))
          sum[j] += load_a(((batch * p(4) + channel) * p(5) + uint(y)) * p(6) +
                           uint(x)) *
                    weight;
      }
    }
  }
  for (uint j = 0; j < 4 && x0 + j < p(8); ++j) {
    uint i = row * p(8) + x0 + j;
    store_output(i, epilogue(sum[j], i));
  }
}
