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
  float cross(float2 o, float2 a, float2 b) {
    auto u = a - o;
    auto v = b - o;
    return sf(mf(u.x, v.y), mf(u.y, v.x));
  }
  float flength(float2 d) {
    return dvalue(dq(ds(af(mf(d.x, d.x), mf(d.y, d.y)))));
  }
  float divide(float a, float b) { return dvalue(dd(ds(a), ds(b))); }
  float project(float2 a, float2 b) { return af(mf(a.x, b.x), mf(a.y, b.y)); }
  float2 from_axes(float2 u, float2 v, float a, float b) {
    return float2(af(mf(u.x, a), mf(v.x, b)), af(mf(u.y, a), mf(v.y, b)));
  }

  float2 direction(float2 delta) {
    if (delta.x == 0.0) {
      return float2(0.0, sign(delta.y));
    }
    if (delta.y == 0.0) {
      return float2(sign(delta.x), 0.0);
    }
    auto len = flength(delta);
    return float2(divide(delta.x, len), divide(delta.y, len));
  }
  array<float2, 4> ordered(array<float2, 4> input) {
    auto lo = 0u;
    auto hi = 0u;
    for (auto i = 1u; i < 4u; i++) {
      if (input[i].x + input[i].y < input[lo].x + input[lo].y) {
        lo = i;
      }
      if (input[i].x + input[i].y > input[hi].x + input[hi].y) {
        hi = i;
      }
    }
    array<uint, 2> remain{};
    auto n = 0u;
    for (auto i = 0u; i < 4u; i++) {
      if (i != lo && i != hi && n < 2u) {
        remain[n] = i;
        n++;
      }
    }
    auto result = input;
    if (n == 2u && lo != hi) {
      auto a = input[remain[0]];
      auto b = input[remain[1]];
      result = array<float2, 4>{input[lo], select(b, a, a.y - a.x < b.y - b.x),
                                input[hi], select(a, b, a.y - a.x < b.y - b.x)};
      auto good = true;
      for (auto i = 0u; i < 4u; i++) {
        good = good && cross(result[i], result[(i + 1u) % 4u],
                             result[(i + 2u) % 4u]) > 0.0;
      }
      if (good) {
        return result;
      }
    }
    auto center = (input[0] + input[1] + input[2] + input[3]) * 0.25;
    array<float, 4> angles{};
    auto order = array<uint, 4>{0u, 1u, 2u, 3u};
    for (auto i = 0u; i < 4u; i++) {
      angles[i] = atan2(input[i].y - center.y, input[i].x - center.x);
    }
    for (auto i = 1u; i < 4u; i++) {
      auto k = order[i];
      auto j = i;
      while (j > 0u && angles[order[j - 1u]] > angles[k]) {
        order[j] = order[j - 1u];
        j--;
      }
      order[j] = k;
    }
    auto first = 0u;
    for (auto i = 1u; i < 4u; i++) {
      auto a = input[order[i]];
      auto b = input[order[first]];
      if (a.x + a.y < b.x + b.y || (a.x + a.y == b.x + b.y &&
                                    (a.y < b.y || (a.y == b.y && a.x < b.x)))) {
        first = i;
      }
    }
    for (auto i = 0u; i < 4u; i++) {
      result[i] = input[order[(first + i) % 4u]];
    }
    return result;
  }
  struct Rect {
    array<float2, 4> points;
    float2 side;
    uint valid;
  };
  Rect rectangle(uint base, uint count) {
    Rect result{};
    result.valid = 0u;
    if (count < 3u) {
      return result;
    }
    auto first = 0u;
    for (auto i = 1u; i < count; i++) {
      auto a = scratch[base + i];
      auto b = scratch[base + first];
      if (a.x < b.x || (a.x == b.x && a.y < b.y)) {
        first = i;
      }
    }
    auto best = 3.402823e38;
    for (auto e = 0u; e < count; e++) {
      auto edge = (first + e) % count;
      auto delta = scratch[base + (edge + 1u) % count] - scratch[base + edge];
      auto len = flength(delta);
      if (len <= 0.0) {
        continue;
      }
      auto u = direction(delta);
      auto v = float2(-u.y, u.x);
      auto low = float2(3.402823e38);
      auto high = float2(-3.402823e38);
      for (auto i = 0u; i < count; i++) {
        auto q = scratch[base + i];
        auto projected = float2(project(q, u), project(q, v));
        low = min(low, projected);
        high = max(high, projected);
      }
      auto side = high - low;
      auto area = mf(side.x, side.y);
      if (area < best) {
        best = area;
        result.side = side;
        result.points = array<float2, 4>{
            from_axes(u, v, low.x, low.y), from_axes(u, v, high.x, low.y),
            from_axes(u, v, high.x, high.y), from_axes(u, v, low.x, high.y)};
        result.valid = uint(area > 0.0);
      }
    }
    result.points = ordered(result.points);
    return result;
  }
  uint push_hull(uint base, uint count, uint lower, float2 point) {
    auto n = count;
    while (n >= lower + 2u && cross(scratch[base + n - 2u],
                                    scratch[base + n - 1u], point) <= 0.0) {
      n--;
    }
    scratch[base + n] = point;
    return n + 1u;
  }
  uint hull_rows(uint base, uint rowbase, uint ybase, uint height) {
    auto count = 0u;
    for (auto y = 0u; y < height; y++) {
      auto lo = rows[(rowbase + y) * 2u];
      auto hi = rows[(rowbase + y) * 2u + 1u];
      if (lo > hi) {
        continue;
      }
      count = push_hull(base, count, 0u, float2(float(lo), float(ybase + y)));
      if (hi != lo) {
        count = push_hull(base, count, 0u, float2(float(hi), float(ybase + y)));
      }
    }
    if (count < 2u) {
      return 0u;
    }
    auto lower = count - 1u;
    auto skipped = false;
    for (auto yy = height; yy > 0u; yy--) {
      auto y = yy - 1u;
      auto lo = rows[(rowbase + y) * 2u];
      auto hi = rows[(rowbase + y) * 2u + 1u];
      if (lo > hi) {
        continue;
      }
      if (skipped) {
        count =
            push_hull(base, count, lower, float2(float(hi), float(ybase + y)));
      } else {
        skipped = true;
      }
      if (hi != lo) {
        count =
            push_hull(base, count, lower, float2(float(lo), float(ybase + y)));
      }
    }
    return count - 1u;
  }
  uint hull_points(uint base, uint n) {
    for (auto i = 1u; i < n; i++) {
      auto q = scratch[base + i];
      auto j = i;
      while (j > 0u) {
        auto prev = scratch[base + j - 1u];
        if (prev.x < q.x || (prev.x == q.x && prev.y <= q.y)) {
          break;
        }
        scratch[base + j] = prev;
        j--;
      }
      scratch[base + j] = q;
    }
    auto unique = 0u;
    for (auto i = 0u; i < n; i++) {
      auto q = scratch[base + i];
      if (unique == 0u || any(q != scratch[base + unique - 1u])) {
        scratch[base + unique] = q;
        unique++;
      }
    }
    if (unique < 3u) {
      return 0u;
    }
    auto count = 0u;
    auto out = base + 512u;
    for (auto i = 0u; i < unique; i++) {
      count = push_hull(out, count, 0u, scratch[base + i]);
    }
    auto lower = count - 1u;
    for (auto j = unique - 1u; j > 0u; j--) {
      count = push_hull(out, count, lower, scratch[base + j - 1u]);
    }
    return count - 1u;
  }
  float score(array<float2, 4> points) {
    auto low = points[0];
    auto high = low;
    for (auto i = 1u; i < 4u; i++) {
      low = min(low, points[i]);
      high = max(high, points[i]);
    }
    auto left = max(int(floor(low.x)), 0);
    auto right = min(int(ceil(high.x)), int(p(0u)) - 1);
    auto top = max(int(floor(low.y)), 0);
    auto bottom = min(int(ceil(high.y)), int(p(1u)) - 1);
    array<int2, 4> q{};
    for (auto i = 0u; i < 4u; i++) {
      q[i] = int2(points[i]) - int2(left, top);
    }
    auto sum = 0.0;
    auto correction = 0.0;
    auto count = 0u;
    for (auto y = top; y <= bottom; y++) {
      auto cy = y - top;
      array<float, 4> thresholds{};
      array<int, 4> lows{};
      array<int, 4> highs{};
      auto nt = 0u;
      auto nh = 0u;
      for (auto i = 0u; i < 4u; i++) {
        auto a = q[i];
        auto b = q[(i + 3u) % 4u];
        auto d = b - a;
        if ((a.y > cy) != (b.y > cy)) {
          thresholds[nt] = af(
              divide(mf(float(d.x), float(cy - a.y)), float(d.y)), float(a.x));
          nt++;
        }
        if (cy < min(a.y, b.y) || cy > max(a.y, b.y)) {
          continue;
        }
        if (d.y == 0) {
          lows[nh] = min(a.x, b.x);
          highs[nh] = max(a.x, b.x);
          nh++;
        } else {
          auto num = (cy - a.y) * d.x;
          if (num % d.y == 0) {
            auto x = a.x + num / d.y;
            if (x >= min(a.x, b.x) && x <= max(a.x, b.x)) {
              lows[nh] = x;
              highs[nh] = x;
              nh++;
            }
          }
        }
      }
      for (auto i = 1u; i < nt; i++) {
        auto v = thresholds[i];
        auto j = i;
        while (j > 0u && thresholds[j - 1u] > v) {
          thresholds[j] = thresholds[j - 1u];
          j--;
        }
        thresholds[j] = v;
      }
      auto t = 0u;
      for (auto x = left; x <= right; x++) {
        auto cx = x - left;
        while (t < nt && thresholds[t] <= float(cx)) {
          t++;
        }
        auto inside = ((nt - t) & 1u) != 0u;
        for (auto k = 0u; k < nh; k++) {
          inside = inside || (cx >= lows[k] && cx <= highs[k]);
        }

        if (inside) {
          auto value =
              sf(float(prediction[uint(y) * p(0u) + uint(x)]), correction);
          auto next = af(sum, value);
          correction = sf(sf(next, sum), value);
          sum = next;
          count++;
        }
      }
    }
    return select(0.0, sum / float(max(count, 1u)), count > 0u);
  }
  float2 offset_point(float2 q, float4 normal, float2 radius) {
    return float2(daway(da(ds(q.x), dm(normal.xy, radius))),
                  daway(da(ds(q.y), dm(normal.zw, radius))));
  }
  uint unclip(uint base, array<float2, 4> points) {
    auto area = ds(0.0);
    auto perimeter = ds(0.0);
    for (auto i = 0u; i < 4u; i++) {
      auto a = points[i];
      auto b = points[(i + 1u) % 4u];
      area = da(area, da(dm(ds(a.x), ds(b.y)), -dm(ds(b.x), ds(a.y))));
      auto d = b - a;
      perimeter = da(perimeter, ds(flength(d)));
    }
    if (dvalue(area) < 0.0) {
      area = -area;
    }
    auto delta =
        dd(dm(dm(area, ds(0.5)), ds(as_type<float>(p(3u)))), perimeter);
    if (!(dvalue(delta) > 0.0 && dvalue(delta) < 3.402823e38)) {
      return 0u;
    }
    array<float2, 4> pts{};
    auto n = 0u;
    for (auto i = 0u; i < 4u; i++) {
      // Integer polygon vertices follow the CPU's ties-to-even contract.
      auto q = rint(points[i]);
      if (n > 0u && all(pts[n - 1u] == q)) {
        continue;
      }
      pts[n] = q;
      n++;
    }
    if (n >= 2u && all(pts[0] == pts[n - 1u])) {
      n--;
    }
    if (n < 3u) {
      return 0u;
    }
    auto winding = ds(0.0);
    for (auto i = 0u; i < n; i++) {
      auto a = pts[i];
      auto b = pts[(i + 1u) % n];
      winding = da(winding, da(dm(ds(a.x), ds(b.y)), -dm(ds(b.x), ds(a.y))));
    }
    if (dvalue(winding) < 0.0) {
      for (auto i = 0u; i < n / 2u; i++) {
        auto q = pts[i];
        pts[i] = pts[n - 1u - i];
        pts[n - 1u - i] = q;
      }
    }
    array<float4, 4> normals{};
    for (auto i = 0u; i < n; i++) {
      auto d = pts[(i + 1u) % n] - pts[i];
      auto len = dq(da(dm(ds(d.x), ds(d.x)), dm(ds(d.y), ds(d.y))));
      normals[i] = float4(dd(ds(d.y), len), dd(ds(-d.x), len));
    }
    auto count = 0u;
    auto previous = n - 1u;
    for (auto i = 0u; i < n; i++) {
      auto q = pts[i];
      auto incoming = normals[previous];
      auto outgoing = normals[i];
      scratch[base + count] = offset_point(q, incoming, delta);
      count++;
      auto turn =
          da(dm(incoming.xy, outgoing.zw), -dm(incoming.zw, outgoing.xy));
      if (dvalue(turn) >= 0.0) {
        if (dvalue(da(dm(incoming.xy, outgoing.xy),
                      dm(incoming.zw, outgoing.zw))) < 0.985) {
          auto start = datan2(incoming.zw, incoming.xy);
          auto end = datan2(outgoing.zw, outgoing.xy);
          if (dvalue(end) < dvalue(start)) {
            end = da(end, dm(dpi(), ds(2.0)));
          }
          auto angle = da(end, -start);
          auto fraction = dd(angle, dm(dpi(), ds(2.0)));
          auto chord = ds(min(0.25, dvalue(delta)));
          auto steps =
              max(2u, uint(dvalue(dd(dm(fraction, dpi()),
                                     dacos(da(ds(1.0), -dd(chord, delta)))))));
          steps = min(steps, max(2u, uint(dvalue(dm(ds(222.0), fraction)))));
          auto step = dd(angle, ds(float(steps)));
          auto sc = dsincos(step);
          auto initial = dsincos(start);
          auto vx = initial.zw;
          auto vy = initial.xy;
          for (auto j = 0u; j <= steps; j++) {
            scratch[base + count] = offset_point(q, float4(vx, vy), delta);
            count++;
            auto next = da(dm(vx, sc.zw), -dm(vy, sc.xy));
            vy = da(dm(vy, sc.zw), dm(vx, sc.xy));
            vx = next;
          }
        }
      } else {
        scratch[base + count] = q;
        count++;
      }
      scratch[base + count] = offset_point(q, outgoing, delta);
      count++;
      previous = i;
    }
    return count;
  }
  void run(uint3 id) {
    auto index = id.x;
    if (index >= metadata[1]) {
      return;
    }
    auto m = 16u + index * 8u;
    auto height = metadata[m + 4u] - metadata[m + 2u] + 1u;
    auto base = metadata[m + 5u] * 4u + index * 1024u;
    auto out = index * 16u;
    boxes[out + 9u] = 0u;
    auto n = hull_rows(base, metadata[m + 5u], metadata[m + 2u], height);
    auto r = rectangle(base, n);
    if (r.valid == 0u || min(r.side.x, r.side.y) < 3.0) {
      return;
    }
    auto confidence = score(r.points);
    if (!(confidence >= as_type<float>(p(2u)))) {
      return;
    }
    auto expanded = unclip(base, r.points);
    auto hull = hull_points(base, expanded);
    r = rectangle(base + 512u, hull);
    if (r.valid == 0u || min(r.side.x, r.side.y) < 5.0) {
      return;
    }
    for (auto i = 0u; i < 4u; i++) {
      scratch[base + i] = clamp(
          rint(float2(mf(divide(r.points[i].x, float(p(0u))), float(p(6u))),
                       mf(divide(r.points[i].y, float(p(1u))), float(p(7u))))),
          float2(0.0), float2(float(p(4u)), float(p(5u))));
    }
    auto mapped = hull_points(base, 4u);
    r = rectangle(base + 512u, mapped);
    if (r.valid == 0u || distance(r.points[0], r.points[1]) <= 4.0 ||
        distance(r.points[0], r.points[3]) <= 4.0) {
      return;
    }
    for (auto i = 0u; i < 4u; i++) {
      boxes[out + i * 2u] = as_type<uint>(r.points[i].x);
      boxes[out + i * 2u + 1u] = as_type<uint>(r.points[i].y);
    }
    boxes[out + 8u] = as_type<uint>(confidence);
    boxes[out + 9u] = 1u;
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
