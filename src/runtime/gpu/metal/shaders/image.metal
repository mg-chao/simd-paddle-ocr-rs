// Native Metal buffer and parameter contract for image.
struct Resident {
  device const uint *pixels;
  device const float *lut;
  device OUT *output;
  constant Params &params;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  int pixel(uint x, uint y, uint c) {
    if (x >= p(0u) || y >= p(1u)) {
      return 0;
    }
    auto channel = select(c, 2u - c, p(4u) != 0u);
    auto byte = y * p(2u) + x * p(3u) + channel;
    return int((pixels[byte / 4u] >> ((byte % 4u) * 8u)) & 255u);
  }

  uint2 multiply_divide(uint a, uint b, uint d) {
    if (a == 0u) {
      return uint2(0u);
    }
    if (b <= 0xffffffffu / a) {
      auto product = a * b;
      return uint2(product / d, product % d);
    }
    auto a0 = a & 65535u;
    auto a1 = a >> 16u;
    auto b0 = b & 65535u;
    auto b1 = b >> 16u;
    auto w0 = a0 * b0;
    auto t = a1 * b0 + (w0 >> 16u);
    auto w1 = (t & 65535u) + a0 * b1;
    auto high = a1 * b1 + (t >> 16u) + (w1 >> 16u);
    auto low = (w1 << 16u) + (w0 & 65535u);
    auto rem = high;
    auto quotient = 0u;
    for (auto bit = 32u; bit > 0u; bit--) {
      rem = (rem << 1u) | ((low >> (bit - 1u)) & 1u);
      if (rem >= d) {
        rem -= d;
        quotient |= 1u << (bit - 1u);
      }
    }
    return uint2(quotient, rem);
  }
  int2 coefficient(uint position, uint source, uint destination) {
    auto denominator = 2u * destination;
    auto numerator = multiply_divide(2u * position + 1u, source, denominator);
    auto base = int(numerator.x) - select(1, 0, numerator.y >= destination);
    auto rem = (numerator.y + denominator - destination) % denominator;
    auto scaled = multiply_divide(rem, 2048u, denominator);
    auto whole = scaled.x;
    auto rest = scaled.y;
    auto up = rest * 2u > denominator ||
              (rest * 2u == denominator && (whole & 1u) != 0u);
    return int2(base, int(whole + uint(up)));
  }
  void run(uint3 id, uint3 grid) {
    auto pixel_index = id.x + id.y * grid.x * 64u;
    auto plane = p(8u) * p(9u);
    if (pixel_index >= plane) {
      return;
    }
    auto y = pixel_index / p(9u);
    auto x = pixel_index % p(9u);
    if (x >= p(7u)) {
      for (auto c = 0u; c < 3u; c++) {
        output[c * plane + pixel_index + p(12u)] =
            normalize_store(as_type<float>(p(11u)));
      }
      return;
    }
    auto xc = coefficient(x, p(5u), p(7u));
    if (xc.x < 0) {
      xc = int2(0, 0);
    }
    if (xc.x >= int(p(5u)) - 1) {
      xc = int2(int(p(5u)) - 1, 0);
    }
    auto yc = coefficient(y, p(6u), p(8u));
    auto x0 = uint(xc.x);
    auto x1 = min(x0 + 1u, p(5u) - 1u);
    auto y0 = uint(clamp(yc.x, 0, int(p(6u)) - 1));
    auto y1 = uint(clamp(yc.x + 1, 0, int(p(6u)) - 1));
    auto a = 2048 - xc.y;
    auto b = xc.y;
    for (auto c = 0u; c < 3u; c++) {
      auto h0 = pixel(x0, y0, c) * a + pixel(x1, y0, c) * b;
      auto h1 = pixel(x0, y1, c) * a + pixel(x1, y1, c) * b;
      auto value = clamp(
          (((h0 >> 4) * (2048 - yc.y) >> 16) + ((h1 >> 4) * yc.y >> 16) + 2) >>
              2,
          0, 255);
      output[c * plane + pixel_index + p(12u)] =
          normalize_store(lut[c * 256u + uint(value)]);
    }
  }
};
kernel void image(device const uint *pixels [[buffer(0)]],
                  device const float *lut [[buffer(1)]],
                  device OUT *output [[buffer(2)]],
                  constant Params &params [[buffer(3)]],
                  uint3 id [[thread_position_in_grid]],
                  uint3 grid [[threadgroups_per_grid]]) {

  Resident state{pixels, lut, output, params};
  state.run(id, grid);
}
