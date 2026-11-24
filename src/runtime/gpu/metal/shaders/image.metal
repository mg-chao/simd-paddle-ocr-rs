uint2 multiply_divide(uint a, uint b, uint d) {
  // Metal supports 64-bit integers on every supported Apple GPU. Widen
  // before multiplying so large image coordinates retain exact rational
  // resize coefficients without a software long-division loop.
  ulong product = ulong(a) * ulong(b);
  return uint2(uint(product / d), uint(product % d));
}
int2 coefficient(uint position, uint source, uint destination) {
  auto denominator = 2u * destination;
  auto numerator = multiply_divide(2u * position + 1u, source, denominator);
  // OpenCV rounds the complete source coordinate to float before
  // quantizing its fractional part to an 11-bit interpolation weight.
  if (numerator.x >= (1u << 24)) {
    // FP32 midpoints here are integers. Preserve which side of the midpoint
    // the fraction falls on by encoding a half-step before the FP32 cast.
    uint twice = 2u * numerator.x + uint(numerator.y > destination) -
                 uint(numerator.y < destination);
    uint rounded = uint(float(twice) * 0.5f);
    return int2(int(min(rounded, source - 1u)), 0);
  }
  float whole = float(numerator.x);
  float divisor = float(denominator);
  float fraction = float(int(numerator.y) - int(destination));
  float coordinate = whole + fraction / divisor;
  float residual = fma(whole - coordinate, divisor, fraction);
  if (residual != 0.0f) {
    float neighbor =
        nextafter(coordinate, residual > 0.0f ? INFINITY : -INFINITY);
    float other = abs(fma(whole - neighbor, divisor, fraction));
    if (other < abs(residual) ||
        (other == abs(residual) && (as_type<uint>(coordinate) & 1u) != 0u)) {
      coordinate = neighbor;
    }
  }
  float base = floor(coordinate);
  return int2(int(base), int(rint((coordinate - base) * 2048.0f)));
}

#ifdef BUILD_RESIZE_AXES
kernel void resize_axes(device int2 *axes [[buffer(0)]],
                        constant Params &params [[buffer(1)]],
                        uint3 id [[thread_position_in_grid]],
                        uint3 grid [[threadgroups_per_grid]]) {
  uint i = id.x + id.y * grid.x * 64u;
  uint4 dims = params.words[0]; // source width/height, destination width/height
  if (i < dims.z)
    axes[i] = coefficient(i, dims.x, dims.z);
  else if (i - dims.z < dims.w)
    axes[i] = coefficient(i - dims.z, dims.y, dims.w);
}
#else
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

  void run(uint3 id, uint3 grid, device const int2 *axes) {
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
    auto xc = axes ? axes[x] : coefficient(x, p(5u), p(7u));
    if (xc.x < 0) {
      xc = int2(0, 0);
    }
    if (xc.x >= int(p(5u)) - 1) {
      xc = int2(int(p(5u)) - 1, 0);
    }
    auto yc = axes ? axes[p(7u) + y] : coefficient(y, p(6u), p(8u));
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
#ifdef IMAGE_FROM_TABLE
                  device const int2 *axes [[buffer(3)]],
                  constant Params &params [[buffer(4)]],
#else
                  constant Params &params [[buffer(3)]],
#endif
                  uint3 id [[thread_position_in_grid]],
                  uint3 grid [[threadgroups_per_grid]]) {

  Resident state{pixels, lut, output, params};
#ifndef IMAGE_FROM_TABLE
  device const int2 *axes = nullptr;
#endif
  state.run(id, grid, axes);
}
#endif
