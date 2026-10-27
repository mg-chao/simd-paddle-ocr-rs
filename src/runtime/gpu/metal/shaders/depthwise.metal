template <bool EDGE>
float depthwise_sum(uint batch, uint channel, int y, int x) const {
  float sum = p(19) ? load_c(channel) : 0.0f;
  uint base = (batch * p(3) + channel) * p(5) * p(6);
#pragma unroll
  for (uint ky = 0; ky < KH; ++ky) {
#pragma unroll
    for (uint kx = 0; kx < KH; ++kx) {
      int sy = y + int(ky), sx = x + int(kx);
      if (!EDGE || (sy >= 0 && sx >= 0 && sy < int(p(5)) && sx < int(p(6))))
        sum += load_a(base + uint(sy) * p(6) + uint(sx)) * load_b(channel * KH * KH + ky * KH + kx);
    }
  }
  return sum;
}
void run(uint3 id, uint3, uint, threadgroup float *) const {
  uint index = (id.x + id.y * p(1)) * 4;
  if (index >= p(0)) return;
  uint plane = p(7) * p(8), channel = (index / plane) % p(4);
  uint batch = index / (plane * p(4));
  int y = int(((index / p(8)) % p(7)) * p(13)) - int(p(17));
  int x = int((index % p(8)) * p(14)) - int(p(18));
  bool interior = index % p(8) + 3 < p(8) && y >= 0 && x >= 0
      && y + KH <= int(p(5)) && x + 3 * int(p(14)) + KH <= int(p(6));
  if (interior) {
    float4 sums = float4(p(19) ? load_c(channel) : 0.0f);
    uint base = (batch * p(3) + channel) * p(5) * p(6);
#pragma unroll
    for (uint ky = 0; ky < KH; ++ky) {
      uint offset = base + uint(y + ky) * p(6) + uint(x);
      if (p(14) == 1) {
        float samples[KH + 3];
#pragma unroll
        for (uint i = 0; i < KH + 3; ++i) samples[i] = load_a(offset + i);
#pragma unroll
        for (uint kx = 0; kx < KH; ++kx)
          sums += float4(samples[kx], samples[kx+1], samples[kx+2], samples[kx+3]) * load_b(channel * KH * KH + ky * KH + kx);
      } else {
#pragma unroll
        for (uint kx = 0; kx < KH; ++kx)
          sums += float4(load_a(offset+kx), load_a(offset+kx+2), load_a(offset+kx+4), load_a(offset+kx+6)) * load_b(channel * KH * KH + ky * KH + kx);
      }
    }
#pragma unroll
    for (uint i = 0; i < 4; ++i) store_output(index+i, epilogue<true>(sums[i], index+i, channel));
  } else {
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
      uint at = index + i;
      if (at >= p(0)) break;
      uint c = (at / plane) % p(4), b = at / (plane * p(4));
      int sy = int(((at / p(8)) % p(7)) * p(13)) - int(p(17));
      int sx = int((at % p(8)) * p(14)) - int(p(18));
      store_output(at, epilogue<true>(depthwise_sum<true>(b, c, sy, sx), at, c));
    }
  }
}
