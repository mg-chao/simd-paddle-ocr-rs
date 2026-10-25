// Native Metal buffer and parameter contract for crop_meta.
struct Resident {
  device const uint *boxes;
  device uint *transforms;
  constant Params &params;
  GEOMETRY_MATH

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  void store(uint base, uint i, float2 value) {
    transforms[base + 8u + i * 2u] = as_type<uint>(value.x);
    transforms[base + 9u + i * 2u] = as_type<uint>(value.y);
  }
  void run(uint3 id) {
    auto index = id.x;
    if (index >= p(0u)) {
      return;
    }

    if (p(1u) != 0u && index >= boxes[0]) {
      return;
    }
    auto base = index * 32u;
    auto input = (index + 1u) * 16u;
    array<float2, 4> q{};
    for (auto i = 0u; i < 4u; i++) {
      q[i] = float2(as_type<float>(boxes[input + i * 2u]),
                    as_type<float>(boxes[input + i * 2u + 1u]));
    }
    auto width =
        uint(max(dfloor(dlength(q[0], q[1])), dfloor(dlength(q[2], q[3]))));
    auto height =
        uint(max(dfloor(dlength(q[0], q[3])), dfloor(dlength(q[1], q[2]))));
    auto vertical = height * 2u >= width * 3u;
    transforms[base] = select(width, height, vertical);
    transforms[base + 1u] = select(height, width, vertical);
    transforms[base + 2u] = width;
    transforms[base + 3u] = height;
    transforms[base + 4u] = uint(vertical);
    transforms[base + 5u] = 0u;
    if (width == 0u || height == 0u) {
      transforms[base + 5u] = 1u;
      return;
    }
    auto dx1 = da(ds(q[1].x), ds(-q[2].x));
    auto dx2 = da(ds(q[3].x), ds(-q[2].x));
    auto dx3 = da(da(ds(q[0].x), ds(-q[1].x)), da(ds(q[2].x), ds(-q[3].x)));
    auto dy1 = da(ds(q[1].y), ds(-q[2].y));
    auto dy2 = da(ds(q[3].y), ds(-q[2].y));
    auto dy3 = da(da(ds(q[0].y), ds(-q[1].y)), da(ds(q[2].y), ds(-q[3].y)));
    auto g = ds(0.0);
    auto h = ds(0.0);
    if (abs(dvalue(dx3)) > 1e-12 || abs(dvalue(dy3)) > 1e-12) {
      auto denominator = da(dm(dx1, dy2), dn(dm(dx2, dy1)));
      if (abs(dvalue(denominator)) <= 1e-12) {
        transforms[base + 5u] = 1u;
        return;
      }
      g = dd(da(dm(dx3, dy2), dn(dm(dx2, dy3))), denominator);
      h = dd(da(dm(dx1, dy3), dn(dm(dx3, dy1))), denominator);
    }
    auto a = da(da(ds(q[1].x), ds(-q[0].x)), dm(g, ds(q[1].x)));
    auto b = da(da(ds(q[3].x), ds(-q[0].x)), dm(h, ds(q[3].x)));
    auto d = da(da(ds(q[1].y), ds(-q[0].y)), dm(g, ds(q[1].y)));
    auto e = da(da(ds(q[3].y), ds(-q[0].y)), dm(h, ds(q[3].y)));
    store(base, 0u, a);
    store(base, 1u, b);
    store(base, 2u, ds(q[0].x));
    store(base, 3u, d);
    store(base, 4u, e);
    store(base, 5u, ds(q[0].y));
    store(base, 6u, g);
    store(base, 7u, h);
    auto gu = dm(g, dd(ds(float(width - 1u)), ds(float(width))));
    auto hv = dm(h, dd(ds(float(height - 1u)), ds(float(height))));
    if (dvalue(da(gu, ds(1.0))) <= 1e-12 || dvalue(da(hv, ds(1.0))) <= 1e-12 ||
        dvalue(da(da(gu, hv), ds(1.0))) <= 1e-12) {
      transforms[base + 5u] = 1u;
    }
  }
};
kernel void crop_meta(device const uint *boxes [[buffer(0)]],
                      device uint *transforms [[buffer(1)]],
                      constant Params &params [[buffer(2)]],
                      uint3 id [[thread_position_in_grid]]) {

  Resident state{boxes, transforms, params};
  state.run(id);
}
