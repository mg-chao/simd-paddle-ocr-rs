#define GEOMETRY_MATH                                                          \
  float rf(float x) {                                                          \
    return as_type<float>(as_type<uint>(x) & params.words[63].w);              \
  }                                                                            \
  float af(float a, float b) { return rf(a + b); }                             \
  float sf(float a, float b) { return rf(a - b); }                             \
  float mf(float a, float b) { return rf(a * b); }                             \
  float2 ds(float x) { return float2(x, 0.0); }                                \
  float2 da(float2 a, float2 b) {                                              \
    auto s = af(a.x, b.x);                                                     \
    auto v = sf(s, a.x);                                                       \
    auto e = af(af(af(sf(a.x, sf(s, v)), sf(b.x, v)), a.y), b.y);              \
    auto hi = af(s, e);                                                        \
    return float2(hi, sf(e, sf(hi, s)));                                       \
  }                                                                            \
  float2 dn(float2 a) { return -a; }                                           \
  float2 dm(float2 a, float2 b) {                                              \
    auto hi = mf(a.x, b.x);                                                    \
    auto ah = as_type<float>(as_type<uint>(a.x) & 0xfffff000u);                \
    auto al = sf(a.x, ah);                                                     \
    auto bh = as_type<float>(as_type<uint>(b.x) & 0xfffff000u);                \
    auto bl = sf(b.x, bh);                                                     \
    auto error =                                                               \
        af(af(af(sf(mf(ah, bh), hi), mf(ah, bl)), mf(al, bh)), mf(al, bl));    \
    auto low = af(af(error, mf(a.x, b.y)), mf(a.y, b.x));                      \
    auto s = af(hi, low);                                                      \
    return float2(s, sf(low, sf(s, hi)));                                      \
  }                                                                            \
  float2 dd(float2 a, float2 b) {                                              \
    auto q = rf(a.x / b.x);                                                    \
    auto remainder = da(a, dn(dm(b, ds(q))));                                  \
    return da(ds(q), ds((remainder.x + remainder.y) / b.x));                   \
  }                                                                            \
  float2 dq(float2 a) {                                                        \
    auto q = rf(sqrt(a.x));                                                    \
    if (q == 0.0) {                                                            \
      return ds(0.0);                                                          \
    }                                                                          \
    return da(ds(q), dd(da(a, dn(dm(ds(q), ds(q)))), ds(2.0 * q)));            \
  }                                                                            \
  float dfloor(float2 a) {                                                     \
    auto base = floor(a.x);                                                    \
    auto tail = af(sf(a.x, base), a.y);                                        \
    return af(base, floor(tail));                                              \
  }                                                                            \
  float dvalue(float2 a) { return a.x + a.y; }                                 \
  float dround(float2 a) {                                                     \
    auto lo = dfloor(a);                                                       \
    auto tail = dvalue(da(a, ds(-lo)));                                        \
    if (tail > 0.5 || (tail == 0.5 && (int(lo) & 1) != 0)) {                   \
      return lo + 1.0;                                                         \
    }                                                                          \
    return lo;                                                                 \
  }                                                                            \
  float2 dlength(float2 a, float2 b) {                                         \
    auto x = da(ds(a.x), ds(-b.x));                                            \
    auto y = da(ds(a.y), ds(-b.y));                                            \
    return dq(da(dm(x, x), dm(y, y)));                                         \
  }                                                                            \
  float2 dpi() { return float2(3.1415927410125732, -8.742278000372485e-8); }   \
  float2 datan(float2 value) {                                                 \
    auto x = value;                                                            \
    auto negative = dvalue(x) < 0.0;                                           \
    if (negative) {                                                            \
      x = -x;                                                                  \
    }                                                                          \
    auto invert = dvalue(x) > 1.0;                                             \
    if (invert) {                                                              \
      x = dd(ds(1.0), x);                                                      \
    }                                                                          \
    auto shift = dvalue(x) > 0.414213562373095;                                \
    if (shift) {                                                               \
      x = dd(da(x, ds(-1.0)), da(x, ds(1.0)));                                 \
    }                                                                          \
    auto square = dm(x, x);                                                    \
    auto term = x;                                                             \
    auto sum = x;                                                              \
    for (auto n = 1u; n < 18u; n++) {                                          \
      term = dm(-term, square);                                                \
      sum = da(sum, dd(term, ds(float(2u * n + 1u))));                         \
    }                                                                          \
    if (shift) {                                                               \
      sum = da(sum, dm(dpi(), ds(0.25)));                                      \
    }                                                                          \
    if (invert) {                                                              \
      sum = da(dm(dpi(), ds(0.5)), -sum);                                      \
    }                                                                          \
    return select(sum, -sum, negative);                                        \
  }                                                                            \
  float2 datan2(float2 y, float2 x) {                                          \
    if (dvalue(x) == 0.0) {                                                    \
      return dm(dpi(), ds(select(-0.5, 0.5, dvalue(y) >= 0.0)));               \
    }                                                                          \
    auto a = datan(dd(y, x));                                                  \
    if (dvalue(x) < 0.0) {                                                     \
      a = da(a, select(-dpi(), dpi(), dvalue(y) >= 0.0));                      \
    }                                                                          \
    return a;                                                                  \
  }                                                                            \
  float4 dsincos(float2 angle) {                                               \
    auto turns = floor(dvalue(dd(da(angle, dpi()), dm(dpi(), ds(2.0)))));      \
    auto x = da(angle, -dm(dpi(), ds(turns * 2.0)));                           \
    auto square = dm(x, x);                                                    \
    auto sn = x;                                                               \
    auto cs = ds(1.0);                                                         \
    auto st = x;                                                               \
    auto ct = ds(1.0);                                                         \
    for (auto n = 1u; n < 16u; n++) {                                          \
      st = dd(dm(-st, square), ds(float((2u * n) * (2u * n + 1u))));           \
      ct = dd(dm(-ct, square), ds(float((2u * n - 1u) * (2u * n))));           \
      sn = da(sn, st);                                                         \
      cs = da(cs, ct);                                                         \
    }                                                                          \
    return float4(sn, cs);                                                     \
  }                                                                            \
  float2 dacos(float2 x) { return datan2(dq(da(ds(1.0), -dm(x, x))), x); }     \
  float daway(float2 x) {                                                      \
    if (dvalue(x) < 0.0) {                                                     \
      return -dfloor(da(-x, ds(0.5)));                                         \
    }                                                                          \
    return dfloor(da(x, ds(0.5)));                                             \
  }
