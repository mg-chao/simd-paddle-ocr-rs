// Native Metal buffer and parameter contract for db_geometry.
struct Resident {
  device const INPUT *prediction;
  device const uint *metadata;
  device const uint *rows;
  device float2 *scratch;
  device uint *boxes;
  constant Params &params;
  GEOMETRY_MATH

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  struct Rect {
    array<float2, 4> points;
    float2 side;
    uint valid;
  };

  float cross_(float2 o, float2 a, float2 b) {
    float2 u = a - o, v = b - o;
    float d = u.x * v.y - u.y * v.x;
    return d;
  }
  // Pick the nearest f32 quotient using the exact product residual. Merely
  // adding a rounded correction double-rounds 1 / (1 - 2^-24) down to one.
  float divide(float a, float b) {
    if (a == 0.)
      return a;
    float magnitude = abs(a), q = rf(magnitude / b);
    float residual = rf(fma(-q, b, magnitude));
    if (residual != 0.) {
      uint bits = as_type<uint>(q);
      float neighbor = as_type<float>(residual > 0. ? bits + 1u : bits - 1u);
      float other = abs(rf(fma(-neighbor, b, magnitude)));
      if (other < abs(residual) ||
          (other == abs(residual) && (bits & 1u) != 0u))
        q = neighbor;
    }
    return a < 0. ? -q : q;
  }

  array<float2, 4> ordered(array<float2, 4> points) {
    for (int i = 1; i < 4; ++i) {
      float2 v = points[i];
      int j = i;
      while (j > 0 && points[j - 1].x > v.x) {
        points[j] = points[j - 1];
        --j;
      }
      points[j] = v;
    }
    float2 a = points[0], b = points[1], c = points[2], d = points[3];
    return array<float2, 4>{a.y <= b.y ? a : b, c.y <= d.y ? c : d,
                            c.y <= d.y ? d : c, a.y <= b.y ? b : a};
  }

  // Same rotating calipers and RotatedRect rounding as src/db/calipers.rs.
  // The first edge participates in the last-wins tie break for equal areas.
  Rect rectangle(uint base, uint count, uint kind) {
    Rect r;
    r.valid = 0u;
    if (count < 3u)
      return r;
    uint top = 0u;
    for (uint i = 1u; i < count; ++i) {
      float2 a = scratch[base + i], b = scratch[base + top];
      if (a.y < b.y || (a.y == b.y && (kind == 3u ? a.x > b.x : a.x < b.x)))
        top = i;
    }
    uint start = (top + (kind == 2u ? count - 1u : 1u)) % count;
    uint4 support = uint4(start);
    for (uint k = 1u; k < count; ++k) {
      uint i = (start + k) % count;
      float2 v = scratch[base + i];
      if (v.y < scratch[base + support.x].y)
        support.x = i;
      if (v.x > scratch[base + support.y].x)
        support.y = i;
      if (v.y > scratch[base + support.z].y)
        support.z = i;
      if (v.x < scratch[base + support.w].x)
        support.w = i;
    }
    float best = 3.402823e38;
    float2 origin = float2(0), u = float2(0), v = float2(0);
    for (uint i = 0u; i < count; ++i) {
      float2 dirs[4];
      for (int k = 0; k < 4; ++k) {
        float2 d = scratch[base + (support[k] + 1u) % count] -
                   scratch[base + support[k]];
        dirs[k] = k == 0   ? d
                  : k == 1 ? float2(d.y, -d.x)
                  : k == 2 ? -d
                           : float2(-d.y, d.x);
      }
      int side = 0;
      for (int k = 1; k < 4; ++k)
        if (dirs[k].y * dirs[side].x < dirs[k].x * dirs[side].y)
          side = k;
      float2 d = dirs[side];
      float inv = dvalue(dd(ds(1.), dlength(d, float2(0))));
      float a = d.x * inv, b = d.y * inv;
      support[side] = (support[side] + 1u) % count;
      float2 left = scratch[base + support.w],
             bottom = scratch[base + support.x];
      float2 dx = scratch[base + support.y] - left,
             dy = scratch[base + support.z] - bottom;
      float w = fma(dx.x, a, dx.y * b), h = fma(-dy.x, b, dy.y * a);
      float area = w * h;
      if (area <= best) {
        best = area;
        float l = fma(a, left.x, b * left.y),
              t = fma(-b, bottom.x, a * bottom.y);
        float inv_det = divide(1., fma(a, a, b * b));
        origin = float2(fma(l, a, -t * b), fma(a, t, b * l)) * inv_det;
        u = float2(a, b) * w;
        v = float2(-b, a) * h;
      }
    }
    float2 center =
        float2(fma(u.x + v.x, 0.5f, origin.x), fma(u.y + v.y, 0.5f, origin.y));
    float w = dvalue(dlength(u, float2(0))), h = dvalue(dlength(v, float2(0)));
    float radians = dvalue(datan2(ds(u.y), ds(u.x)));
    float degrees = dvalue(dd(ds(radians * 180.), dpi()));
    float4 sc = dsincos(dm(ds(degrees), dd(dpi(), ds(180.))));
    // Exact OpenCV binary64 sin/cos rounded to f32 at quadrant boundaries.
    float a = dvalue(sc.xy) * 0.5, b = dvalue(sc.zw) * 0.5;
    if (degrees == 90. || degrees == -90.) {
      a = sign(degrees) * 0.5;
      b = 3.061617e-17;
    }
    if (abs(degrees) == 180.) {
      a = sign(degrees) * 6.123234e-17;
      b = -0.5;
    }
    float2 first = float2(fma(-b, w, fma(-a, h, center.x)),
                          fma(-a, w, fma(b, h, center.y)));
    float2 second = float2(fma(-b, w, fma(a, h, center.x)),
                           fma(-a, w, fma(-b, h, center.y)));
    r.points = ordered(array<float2, 4>{first, second, 2. * center - first,
                                        2. * center - second});
    r.side = float2(w, h);
    r.valid = 1u;
    return r;
  }

  uint push_hull(uint base, uint count, uint lower, float2 point) {
    while (count >= lower + 2u &&
           cross_(scratch[base + count - 2u], scratch[base + count - 1u],
                  point) <= 0.)
      --count;
    scratch[base + count] = point;
    return count + 1u;
  }

  // A hole's contour follows the four adjacent foreground pixels. Its convex
  // hull is therefore the hull of its row extrema expanded by one cardinal
  // step.
  uint2 row_span(uint rowbase, int y) {
    return uint2(rows[(rowbase + uint(y)) * 2u],
                 rows[(rowbase + uint(y)) * 2u + 1u]);
  }
  uint2 contour_span(uint rowbase, int y, int height, bool hole) {
    uint2 span = uint2(0xffffffffu, 0u);
    if (y >= 0 && y < height) {
      span = row_span(rowbase, y);
      if (hole) {
        --span.x;
        ++span.y;
      }
    }
    if (hole) {
      for (int k = -1; k <= 1; k += 2) {
        if (y + k < 0 || y + k >= height)
          continue;
        uint2 adjacent = row_span(rowbase, y + k);
        span = uint2(min(span.x, adjacent.x), max(span.y, adjacent.y));
      }
    }
    return span;
  }
  uint hull_rows(uint base, uint rowbase, uint ybase, uint height, bool hole) {
    uint n = 0u, lower = 0u;
    int first = hole ? -1 : 0, last = int(height) - 1 - first;
    for (int pass = 0; pass < 2; ++pass) {
      for (int k = first; k <= last; ++k) {
        int y = pass == 0 ? k : last - (k - first);
        uint2 span = contour_span(rowbase, y, int(height), hole);
        float yy = float(int(ybase) + y);
        if (pass == 0 || k != first)
          n = push_hull(base, n, lower,
                        float2(pass == 0 ? span.x : span.y, yy));
        if (span.y != span.x)
          n = push_hull(base, n, lower,
                        float2(pass == 0 ? span.y : span.x, yy));
      }
      if (n < 2u)
        return 0u;
      lower = n - 1u;
    }
    return n - 1u;
  }

  uint hull_points(uint base, uint count) {
    for (uint i = 1u; i < count; ++i) {
      float2 q = scratch[base + i];
      uint j = i;
      while (j > 0u) {
        float2 prev = scratch[base + j - 1u];
        if (prev.x < q.x || (prev.x == q.x && prev.y <= q.y))
          break;
        scratch[base + j] = prev;
        --j;
      }
      scratch[base + j] = q;
    }
    uint unique = 0u;
    for (uint i = 0u; i < count; ++i) {
      float2 q = scratch[base + i];
      if (unique == 0u || any(q != scratch[base + unique - 1u]))
        scratch[base + unique++] = q;
    }
    if (unique < 3u)
      return 0u;
    uint n = 0u, dest = base + 512u;
    for (uint i = 0u; i < unique; ++i)
      n = push_hull(dest, n, 0u, scratch[base + i]);
    uint lower = n - 1u;
    for (uint i = unique - 1u; i > 0u; --i)
      n = push_hull(dest, n, lower, scratch[base + i - 1u]);
    return n - 1u;
  }

  // Integer LINE_8 mask, including both Bresenham edge pixels on shallow
  // sides. For each row invert the edge's integer error recurrence.
  float score(array<float2, 4> points) {
    int2 extent = int2(p(0u), p(1u));
    float2 lo = min(min(points[0], points[1]), min(points[2], points[3]));
    float2 hi = max(max(points[0], points[1]), max(points[2], points[3]));
    int2 origin = clamp(int2(floor(lo)), int2(0), extent - 1);
    int bottom = clamp(int(ceil(hi.y)), 0, extent.y - 1);
    int2 pts[4];
    for (int i = 0; i < 4; ++i)
      pts[i] = int2(points[i] - float2(origin)) + origin;
    float2 sum = ds(0.);
    int count = 0;
    for (int y = origin.y; y <= bottom; ++y) {
      int left = extent.x, right = -1;
      for (int i = 0; i < 4; ++i) {
        int2 a = pts[i], b = pts[(i + 1) % 4];
        if (a.x > b.x) {
          int2 tmp = a;
          a = b;
          b = tmp;
        }
        if (y < min(a.y, b.y) || y > max(a.y, b.y))
          continue;
        int k = abs(y - a.y), dx = b.x - a.x, dy = abs(b.y - a.y);
        int l = 0, r = dx;
        if (dy > dx) {
          l = (2 * k * dx + dy - 1) / (2 * dy);
          r = l;
        } else if (dy > 0) {
          l = max(0, (2 * k * dx - dx + 1 + 2 * dy - 1) / (2 * dy));
          r = min(dx, (2 * (k + 1) * dx - dx) / (2 * dy));
        }
        left = min(left, a.x + l);
        right = max(right, a.x + r);
      }
      left = max(left, 0);
      right = min(right, extent.x - 1);
      for (int x = left; x <= right; ++x)
        sum = da(sum, ds(float(prediction[y * extent.x + x])));
      count += max(0, right - left + 1);
    }
    return count == 0 ? 0. : dvalue(dd(sum, ds(float(count))));
  }

  float2 offset_point(float2 q, float4 normal, float2 radius) {
    return float2(daway(da(ds(q.x), dm(normal.xy, radius))),
                  daway(da(ds(q.y), dm(normal.zw, radius))));
  }

  uint unclip(uint base, array<float2, 4> points) {
    float2 area = ds(0.), perimeter = ds(0.);
    float2 q[4];
    int n = 0;
    for (int i = 0; i < 4; ++i) {
      float2 a = points[i], b = points[(i + 1) % 4];
      area = da(area, da(dm(ds(a.x), ds(b.y)), -dm(ds(b.x), ds(a.y))));
      perimeter = da(perimeter, dlength(a, b));
      if (n == 0 || any(trunc(a) != q[n - 1]))
        q[n++] = trunc(a);
    }
    if (n > 1 && all(q[n - 1] == q[0]))
      --n;
    if (n < 3)
      return 0u;
    float configured = as_type<float>(p(3u));
    // Keep the default rational factor intact until the division. Computing
    // 8/5 first can push an exact 4.5-pixel offset across Clipper's tie.
    float2 delta = configured == 1.6
                       ? dd(dm(area, ds(4.)), dm(perimeter, ds(5.)))
                       : dd(dm(dm(area, ds(0.5)), ds(configured)), perimeter);
    if (dvalue(delta) < 0.)
      delta = -delta;
    if (!(dvalue(delta) > 0.))
      return 0u;
    float winding = 0.;
    for (int i = 0; i < n; ++i) {
      float2 a = q[i], b = q[(i + 1) % n];
      winding += a.x * b.y - b.x * a.y;
    }
    if (winding < 0.) {
      for (int i = 0; i < n / 2; ++i) {
        float2 t = q[i];
        q[i] = q[n - 1 - i];
        q[n - 1 - i] = t;
      }
    }
    float4 normals[4];
    for (int i = 0; i < n; ++i) {
      float2 d = q[(i + 1) % n] - q[i], len = dlength(d, float2(0));
      normals[i] = float4(dd(ds(d.y), len), dd(ds(-d.x), len));
    }
    float2 tolerance = dvalue(delta) < 1. ? dm(delta, ds(0.25)) : ds(0.25);
    float2 steps = dd(dpi(), dacos(da(ds(1.), -dd(tolerance, delta))));
    float2 limit = dm(delta, dpi());
    if (dvalue(steps) > dvalue(limit))
      steps = limit;
    float2 angle = dd(dm(ds(2.), dpi()), steps);
    float4 sc = dsincos(angle);
    uint written = 0u;
    int prev = n - 1;
    for (int i = 0; i < n; ++i) {
      float4 normal = normals[prev], end = normals[i];
      float2 cross_val = da(dm(normal.xy, end.zw), -dm(normal.zw, end.xy));
      float2 dot_val = da(dm(normal.xy, end.xy), dm(normal.zw, end.zw));
      if (abs(dvalue(dm(cross_val, delta))) < 1. && dvalue(dot_val) > 0.) {
        scratch[base + written++] = offset_point(q[i], normal, delta);
        continue;
      }
      if (dvalue(cross_val) < 0.) {
        scratch[base + written++] = offset_point(q[i], normal, delta);
        scratch[base + written++] = q[i];
      } else {
        float2 arc = datan2(cross_val, dot_val);
        uint count = max(1u, uint(floor(abs(dvalue(dd(arc, angle))) + 0.5)));
        // The temporary offset hull occupies the first 512 slots.
        uint stride = max(1u, (count + 119u) / 120u);
        if (stride > 1u)
          sc = dsincos(dm(angle, ds(float(stride))));
        for (uint j = 0u; j < count; j += stride) {
          scratch[base + written++] = offset_point(q[i], normal, delta);
          normal = float4(da(dm(normal.xy, sc.zw), -dm(normal.zw, sc.xy)),
                          da(dm(normal.xy, sc.xy), dm(normal.zw, sc.zw)));
        }
        if (stride > 1u)
          sc = dsincos(angle);
      }
      scratch[base + written++] = offset_point(q[i], end, delta);
      prev = i;
    }
    return written;
  }

  void run(uint3 id) {
    uint index = id.x;
    if (index >= metadata[1])
      return;
    uint m = 16u + index * 8u, outbox = index * 16u;
    uint y = metadata[m + 2u], height = metadata[m + 4u] - y + 1u;
    uint rowbase = metadata[m + 5u], kind = metadata[m + 6u];
    uint base = rowbase * 4u + index * 1024u;
    boxes[outbox + 9u] = 0u;
    uint n = hull_rows(base, rowbase, y, height, kind == 2u);
    Rect r = rectangle(base, n, kind);
    if (r.valid == 0u || min(r.side.x, r.side.y) < 3.)
      return;
    float confidence = score(r.points);
    if (!(confidence >= as_type<float>(p(2u))))
      return;
    n = unclip(base, r.points);
    if (n < 3u)
      return;
    n = hull_points(base, n);
    r = rectangle(base + 512u, n, 3u);
    if (r.valid == 0u || min(r.side.x, r.side.y) < 5.)
      return;
    uint width = p(8u) == 0u ? p(0u) : p(8u);
    uint height_limit = p(9u) == 0u ? p(1u) : p(9u);
    for (int i = 0; i < 4; ++i) {
      float2 mapped = float2(divide(r.points[i].x, float(width)),
                             divide(r.points[i].y, float(height_limit))) *
                      float2(p(6u), p(7u));
      r.points[i] =
          clamp(rint(mapped), float2(0), float2(p(4u) - 1u, p(5u) - 1u));
    }
    r.points = ordered(r.points);
    if (p(10u) != 0u) {
        for (uint i = 0u; i < 4u; ++i) {
            float2 a = r.points[(i + 1u) % 4u] - r.points[i];
            float2 b = r.points[(i + 2u) % 4u] - r.points[(i + 1u) % 4u];
            if (a.x * b.y - a.y * b.x <= 0.)
                return;
        }
    }
    if (distance(r.points[0], r.points[1]) < 4. ||
        distance(r.points[0], r.points[3]) < 4.)
      return;
    for (uint i = 0u; i < 4u; ++i) {
      boxes[outbox + i * 2u] = as_type<uint>(r.points[i].x);
      boxes[outbox + i * 2u + 1u] = as_type<uint>(r.points[i].y);
    }
    boxes[outbox + 8u] = as_type<uint>(confidence);
    boxes[outbox + 9u] = 1u;
  }
};
kernel void db_geometry(device const INPUT *prediction [[buffer(0)]],
                        device const uint *metadata [[buffer(1)]],
                        device const uint *rows [[buffer(2)]],
                        device float2 *scratch [[buffer(3)]],
                        device uint *boxes [[buffer(4)]],
                        constant Params &params [[buffer(5)]],
                        uint3 id [[thread_position_in_grid]]) {

  Resident state{prediction, metadata, rows, scratch, boxes, params};
  state.run(id);
}
