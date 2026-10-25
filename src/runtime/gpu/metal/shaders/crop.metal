// Native Metal buffer and parameter contract for crop.
struct Resident {
  device const uint *pixels;
  device const uint *transforms;
  device uint *output;
  constant Params &params;
  GEOMETRY_MATH

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  uint t(uint i) { return transforms[p(5u) * 32u + i]; }
  float2 coefficient(uint i) {
    return float2(as_type<float>(t(8u + i * 2u)),
                  as_type<float>(t(9u + i * 2u)));
  }
  float pixel(int x, int y, uint c) {
    auto px = uint(clamp(x, 0, int(p(0u)) - 1));
    auto py = uint(clamp(y, 0, int(p(1u)) - 1));
    auto channel = select(c, 2u - c, p(4u) != 0u);
    auto offset = py * p(2u) + px * p(3u) + channel;
    return float((pixels[offset / 4u] >> ((offset % 4u) * 8u)) & 255u);
  }
  float2 cubic(float2 value) {
    auto x = value;
    if (dvalue(x) < 0.0) {
      x = -x;
    }
    if (dvalue(x) <= 1.0) {
      return da(dm(da(dm(ds(1.25), x), ds(-2.25)), dm(x, x)), ds(1.0));
    }
    if (dvalue(x) < 2.0) {
      return da(dm(da(dm(da(dm(ds(-0.75), x), ds(3.75)), x), ds(-6.0)), x),
                ds(3.0));
    }
    return ds(0.0);
  }
  void run(uint3 id, uint3 grid) {
    auto index = id.x + id.y * grid.x * 64u;
    auto width = t(0u);
    auto height = t(1u);
    if (index >= width * height) {
      return;
    }
    auto ox = index % width;
    auto oy = index / width;
    auto vertical = t(4u) != 0u;
    auto x = select(ox, t(2u) - 1u - oy, vertical);
    auto y = select(oy, ox, vertical);
    auto u = dd(ds(float(x)), ds(float(t(2u))));
    auto v = dd(ds(float(y)), ds(float(t(3u))));
    auto den = da(da(dm(coefficient(6u), u), dm(coefficient(7u), v)), ds(1.0));
    auto sx = dd(
        da(da(dm(coefficient(0u), u), dm(coefficient(1u), v)), coefficient(2u)),
        den);
    auto sy = dd(
        da(da(dm(coefficient(3u), u), dm(coefficient(4u), v)), coefficient(5u)),
        den);
    auto xb = int(dfloor(sx));
    auto yb = int(dfloor(sy));
    auto result = 0xff000000u;
    array<float2, 4> wx{};
    array<float2, 4> wy{};
    for (auto k = 0; k < 4; k++) {
      wx[k] = cubic(da(sx, ds(-float(xb + k - 1))));
      wy[k] = cubic(da(sy, ds(-float(yb + k - 1))));
    }
    for (auto c = 0u; c < 3u; c++) {
      auto sum = ds(0.0);
      for (auto ky = 0; ky < 4; ky++) {
        for (auto kx = 0; kx < 4; kx++) {
          sum = da(sum, dm(dm(ds(pixel(xb + kx - 1, yb + ky - 1, c)), wx[kx]),
                           wy[ky]));
        }
      }
      result |= uint(clamp(dfloor(da(sum, ds(0.5))), 0.0, 255.0)) << (c * 8u);
    }
    output[index] = result;
  }
};
kernel void crop(device const uint *pixels [[buffer(0)]],
                 device const uint *transforms [[buffer(1)]],
                 device uint *output [[buffer(2)]],
                 constant Params &params [[buffer(3)]],
                 uint3 id [[thread_position_in_grid]],
                 uint3 grid [[threadgroups_per_grid]]) {

  Resident state{pixels, transforms, output, params};
  state.run(id, grid);
}
