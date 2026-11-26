// Cross-platform Vulkan DB geometry, using the same mask and integer offset
// conventions as src/db.rs. OpenCV calipers attribution is in NOTICE.
layout(local_size_x = 1) in;
#include "storage.glsl"

struct Rect {
    vec2 points[4];
    vec2 side;
    uint valid;
};
layout(std430, set = 0, binding = 0) readonly buffer Predictions { STORAGE_0 predictions[]; };
layout(std430, set = 0, binding = 1) readonly buffer Metadata { uint metadata[]; };
layout(std430, set = 0, binding = 2) readonly buffer Rows { uint rows[]; };
layout(std430, set = 0, binding = 3) buffer Hull { vec2 hull[]; };
layout(std430, set = 0, binding = 4) buffer Boxes { uint boxes[]; };
layout(std140, set = 0, binding = 5) uniform Params { uvec4 words[64]; };

// Double-single arithmetic for subpixel decisions on Vulkan devices without
// shaderFloat64. Keep the residual of each operation through round-to-integer.
float rf(float x) { return uintBitsToFloat(floatBitsToUint(x) & words[63].w); }
float af(float x, float y) { return rf(x + y); }
float sf(float x, float y) { return rf(x - y); }
float mf(float x, float y) { return rf(x * y); }
vec2 ds(float x) { return vec2(x, 0.); }
float dvalue(vec2 a) { return a.x + a.y; }
vec2 normalized(float a, float b) {
    float s = af(a, b);
    float t = sf(b, sf(s, a));
    return vec2(s, t);
}
vec2 da(vec2 a, vec2 b) {
    float s = af(a.x, b.x), t = sf(s, a.x);
    float e = af(af(sf(a.x, sf(s, t)), sf(b.x, t)), af(a.y, b.y));
    return normalized(s, e);
}
vec2 dm(vec2 a, vec2 b) {
    float product = mf(a.x, b.x);
    float e = fma(a.x, b.x, -product) + (a.x * b.y + a.y * b.x);
    return normalized(product, e);
}
vec2 dd(vec2 a, vec2 b) {
    float q = rf(a.x / b.x);
    float r = af(rf(fma(-q, b.x, a.x)), sf(a.y, mf(q, b.y)));
    vec2 result = normalized(q, rf(r / b.x));
    if (a.y == 0. && b.y == 0. && rf(fma(-result.x, b.x, a.x)) == 0.)
        return ds(result.x);
    return result;
}
vec2 dq(vec2 a) {
    if (a.x <= 0.)
        return ds(0.);
    float s = rf(sqrt(a.x));
    float r = fma(-s, s, a.x) + a.y;
    vec2 result = normalized(s, rf(r / (2. * s)));
    if (a.y == 0. && rf(fma(-result.x, result.x, a.x)) == 0.)
        return ds(result.x);
    return result;
}
vec2 dlength(vec2 a, vec2 b) {
    vec2 x = da(ds(a.x), ds(-b.x)), y = da(ds(a.y), ds(-b.y));
    return dq(da(dm(x, x), dm(y, y)));
}
vec2 dpi() { return vec2(3.1415927410125732, -8.742277657347586e-8); }
#ifdef DB_NATIVE_TRIG
vec2 datan2_(vec2 y, vec2 x) { return ds(atan(dvalue(y), dvalue(x))); }
vec4 dsincos(vec2 theta) { return vec4(ds(sin(dvalue(theta))), ds(cos(dvalue(theta)))); }
#else
vec2 datan2_(vec2 y, vec2 x) {
    if (x.x == 0.)
        return dm(dpi(), ds(y.x < 0. ? -0.5 : 0.5));
    bool swap = abs(y.x) > abs(x.x);
    vec2 z = swap ? dd(x, y) : dd(y, x);
    if (z.x < 0.)
        z = -z;
    bool diagonal = z.x > 0.41421356;
    if (diagonal)
        z = dd(da(z, ds(-1.)), da(z, ds(1.)));
    vec2 term = z, sum = z, square = -dm(z, z);
    for (int k = 3; k <= 41; k += 2) {
        term = dm(term, square);
        sum = da(sum, dd(term, ds(float(k))));
    }
    if (diagonal)
        sum = da(dm(dpi(), ds(0.25)), sum);
    if (swap)
        sum = da(dm(dpi(), ds(0.5)), -sum);
    if (x.x < 0.)
        sum = da(dpi(), -sum);
    return y.x < 0. ? -sum : sum;
}
vec4 dsincos(vec2 theta) {
    vec2 half_pi = dm(dpi(), ds(0.5));
    int n = int(roundEven(dvalue(dd(theta, half_pi))));
    vec2 x = da(theta, -dm(ds(float(n)), half_pi));
    vec2 square = -dm(x, x), a = x, b = ds(1.), s = a, c = b;
    for (int k = 2; k <= 18; k += 2) {
        a = dd(dm(a, square), ds(float(k * (k + 1))));
        s = da(s, a);
        b = dd(dm(b, square), ds(float(k * (k - 1))));
        c = da(c, b);
    }
    int quadrant = n & 3;
    if (quadrant == 1)
        return vec4(c, -s);
    if (quadrant == 2)
        return vec4(-s, -c);
    if (quadrant == 3)
        return vec4(-c, s);
    return vec4(s, c);
}
#endif
vec2 dacos(vec2 x) { return datan2_(dq(dm(da(ds(1.), -x), da(ds(1.), x))), x); }
float daway(vec2 x) {
    bool neg = x.x < 0.;
    vec2 v = da(neg ? -x : x, ds(0.5));
    float base = floor(v.x);
    // Clipper first rounds the coordinate plus 0.5 to binary64. Discard
    // residuals inside its half-ULP below an integer (including powers of two).
    if (base == v.x && base >= 1. && v.y < 0.) {
        uint bits = floatBitsToUint(base);
        int exponent = int(bits >> 23u) - 127 - int((bits & 0x7fffffu) == 0u);
        if (-v.y <= exp2(float(exponent - 53)))
            v.y = 0.;
    }
    float rounded = base + floor((v.x - base) + v.y);
    return neg ? -rounded : rounded;
}

uint p(uint i_2) {
    uint _e9 = words[(i_2 / 4u)][(i_2 % 4u)];
    return _e9;
}

float cross_(vec2 o, vec2 a, vec2 b) {
    vec2 u = a - o, v = b - o;
    float d = u.x * v.y - u.y * v.x;
    return d;
}
// Pick the nearest f32 quotient using the exact product residual. Merely
// adding a rounded correction double-rounds 1 / (1 - 2^-24) down to one.
float divide(float a, float b) {
    if (a == 0.) return a;
    float magnitude = abs(a), q = rf(magnitude / b);
    float residual = rf(fma(-q, b, magnitude));
    if (residual != 0.) {
        uint bits = floatBitsToUint(q);
        float neighbor = uintBitsToFloat(residual > 0. ? bits + 1u : bits - 1u);
        float other = abs(rf(fma(-neighbor, b, magnitude)));
        if (other < abs(residual) || (other == abs(residual) && (bits & 1u) != 0u))
            q = neighbor;
    }
    return a < 0. ? -q : q;
}

vec2[4] ordered(vec2 points[4]) {
    for (int i = 1; i < 4; ++i) {
        vec2 v = points[i];
        int j = i;
        while (j > 0 && points[j - 1].x > v.x) {
            points[j] = points[j - 1];
            --j;
        }
        points[j] = v;
    }
    vec2 a = points[0], b = points[1], c = points[2], d = points[3];
    return vec2[4](a.y <= b.y ? a : b, c.y <= d.y ? c : d, c.y <= d.y ? d : c, a.y <= b.y ? b : a);
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
        vec2 a = hull[base + i], b = hull[base + top];
        if (a.y < b.y || (a.y == b.y && (kind == 3u ? a.x > b.x : a.x < b.x)))
            top = i;
    }
    uint start = (top + (kind == 2u ? count - 1u : 1u)) % count;
    uvec4 support = uvec4(start);
    for (uint k = 1u; k < count; ++k) {
        uint i = (start + k) % count;
        vec2 v = hull[base + i];
        if (v.y < hull[base + support.x].y)
            support.x = i;
        if (v.x > hull[base + support.y].x)
            support.y = i;
        if (v.y > hull[base + support.z].y)
            support.z = i;
        if (v.x < hull[base + support.w].x)
            support.w = i;
    }
    float best = 3.402823e38;
    vec2 origin = vec2(0), u = vec2(0), v = vec2(0);
    for (uint i = 0u; i < count; ++i) {
        vec2 dirs[4];
        for (int k = 0; k < 4; ++k) {
            vec2 d = hull[base + (support[k] + 1u) % count] - hull[base + support[k]];
            dirs[k] = k == 0 ? d : k == 1 ? vec2(d.y, -d.x) : k == 2 ? -d : vec2(-d.y, d.x);
        }
        int side = 0;
        for (int k = 1; k < 4; ++k)
            if (dirs[k].y * dirs[side].x < dirs[k].x * dirs[side].y)
                side = k;
        vec2 d = dirs[side];
        float inv = dvalue(dd(ds(1.), dlength(d, vec2(0))));
        float a = d.x * inv, b = d.y * inv;
        support[side] = (support[side] + 1u) % count;
        vec2 left = hull[base + support.w], bottom = hull[base + support.x];
        vec2 dx = hull[base + support.y] - left, dy = hull[base + support.z] - bottom;
        float w = af(mf(dx.x, a), mf(dx.y, b)), h = af(mf(-dy.x, b), mf(dy.y, a));
        float area = w * h;
        if (area <= best) {
            best = area;
            float l = af(mf(a, left.x), mf(b, left.y)), t = af(mf(-b, bottom.x), mf(a, bottom.y));
            float inv_det = divide(1., af(mf(a, a), mf(b, b)));
            origin = vec2(sf(mf(l, a), mf(t, b)), af(mf(a, t), mf(b, l))) * inv_det;
            u = vec2(a, b) * w;
            v = vec2(-b, a) * h;
        }
    }
    vec2 center = vec2(af(origin.x, mf(af(u.x, v.x), 0.5)), af(origin.y, mf(af(u.y, v.y), 0.5)));
    float w = dvalue(dlength(u, vec2(0))), h = dvalue(dlength(v, vec2(0)));
    float radians = dvalue(datan2_(ds(u.y), ds(u.x)));
    float degrees = dvalue(dd(ds(radians * 180.), dpi()));
    vec4 sc = dsincos(dm(ds(degrees), dd(dpi(), ds(180.))));
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
    vec2 first = vec2(sf(sf(center.x, mf(a, h)), mf(b, w)), sf(af(center.y, mf(b, h)), mf(a, w)));
    vec2 second = vec2(sf(af(center.x, mf(a, h)), mf(b, w)), sf(sf(center.y, mf(b, h)), mf(a, w)));
    r.points = ordered(vec2[4](first, second, 2. * center - first, 2. * center - second));
    r.side = vec2(w, h);
    r.valid = 1u;
    return r;
}

uint push_hull(uint base, uint count, uint lower, vec2 point) {
    while (count >= lower + 2u &&
           cross_(hull[base + count - 2u], hull[base + count - 1u], point) <= 0.)
        --count;
    hull[base + count] = point;
    return count + 1u;
}

// A hole's contour follows the four adjacent foreground pixels. Its convex
// hull is therefore the hull of its row extrema expanded by one cardinal step.
uvec2 row_span(uint rowbase, int y) {
    return uvec2(rows[(rowbase + uint(y)) * 2u], rows[(rowbase + uint(y)) * 2u + 1u]);
}
uvec2 contour_span(uint rowbase, int y, int height, bool hole) {
    uvec2 span = uvec2(0xffffffffu, 0u);
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
            uvec2 adjacent = row_span(rowbase, y + k);
            span = uvec2(min(span.x, adjacent.x), max(span.y, adjacent.y));
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
            uvec2 span = contour_span(rowbase, y, int(height), hole);
            float yy = float(int(ybase) + y);
            if (pass == 0 || k != first)
                n = push_hull(base, n, lower, vec2(pass == 0 ? span.x : span.y, yy));
            if (span.y != span.x)
                n = push_hull(base, n, lower, vec2(pass == 0 ? span.y : span.x, yy));
        }
        if (n < 2u)
            return 0u;
        lower = n - 1u;
    }
    return n - 1u;
}

uint hull_points(uint base, uint count) {
    for (uint i = 1u; i < count; ++i) {
        vec2 q = hull[base + i];
        uint j = i;
        while (j > 0u) {
            vec2 prev = hull[base + j - 1u];
            if (prev.x < q.x || (prev.x == q.x && prev.y <= q.y))
                break;
            hull[base + j] = prev;
            --j;
        }
        hull[base + j] = q;
    }
    uint unique = 0u;
    for (uint i = 0u; i < count; ++i) {
        vec2 q = hull[base + i];
        if (unique == 0u || q != hull[base + unique - 1u])
            hull[base + unique++] = q;
    }
    if (unique < 3u)
        return 0u;
    uint n = 0u, dest = base + 512u;
    for (uint i = 0u; i < unique; ++i)
        n = push_hull(dest, n, 0u, hull[base + i]);
    uint lower = n - 1u;
    for (uint i = unique - 1u; i > 0u; --i)
        n = push_hull(dest, n, lower, hull[base + i - 1u]);
    return n - 1u;
}

// Integer LINE_8 mask, including both Bresenham edge pixels on shallow
// sides. For each row invert the edge's integer error recurrence.
float score(vec2 points[4]) {
    ivec2 extent = ivec2(p(0u), p(1u));
    vec2 lo = min(min(points[0], points[1]), min(points[2], points[3]));
    vec2 hi = max(max(points[0], points[1]), max(points[2], points[3]));
    ivec2 origin = clamp(ivec2(floor(lo)), ivec2(0), extent - 1);
    int bottom = clamp(int(ceil(hi.y)), 0, extent.y - 1);
    ivec2 pts[4];
    for (int i = 0; i < 4; ++i)
        pts[i] = ivec2(points[i] - vec2(origin)) + origin;
    vec2 sum = ds(0.);
    int count = 0;
    for (int y = origin.y; y <= bottom; ++y) {
        int left = extent.x, right = -1;
        for (int i = 0; i < 4; ++i) {
            ivec2 a = pts[i], b = pts[(i + 1) % 4];
            if (a.x > b.x) {
                ivec2 tmp = a;
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
            sum = da(sum, ds(float(predictions[y * extent.x + x])));
        count += max(0, right - left + 1);
    }
    return count == 0 ? 0. : dvalue(dd(sum, ds(float(count))));
}

vec2 offset_point(vec2 q, vec4 normal, vec2 radius) {
    return vec2(daway(da(ds(q.x), dm(normal.xy, radius))),
                daway(da(ds(q.y), dm(normal.zw, radius))));
}

uint unclip(uint base, vec2 points[4]) {
    vec2 area = ds(0.), perimeter = ds(0.);
    vec2 q[4];
    int n = 0;
    for (int i = 0; i < 4; ++i) {
        vec2 a = points[i], b = points[(i + 1) % 4];
        area = da(area, da(dm(ds(a.x), ds(b.y)), -dm(ds(b.x), ds(a.y))));
        perimeter = da(perimeter, dlength(a, b));
        if (n == 0 || trunc(a) != q[n - 1])
            q[n++] = trunc(a);
    }
    if (n > 1 && q[n - 1] == q[0])
        --n;
    if (n < 3)
        return 0u;
    float configured = uintBitsToFloat(p(3u));
    // Keep the default rational factor intact until the division. Computing
    // 8/5 first can push an exact 4.5-pixel offset across Clipper's tie.
    vec2 delta = configured == 1.6 ? dd(dm(area, ds(4.)), dm(perimeter, ds(5.)))
                                   : dd(dm(dm(area, ds(0.5)), ds(configured)), perimeter);
    if (dvalue(delta) < 0.)
        delta = -delta;
    if (!(dvalue(delta) > 0.))
        return 0u;
    float winding = 0.;
    for (int i = 0; i < n; ++i) {
        vec2 a = q[i], b = q[(i + 1) % n];
        winding += a.x * b.y - b.x * a.y;
    }
    if (winding < 0.) {
        for (int i = 0; i < n / 2; ++i) {
            vec2 t = q[i];
            q[i] = q[n - 1 - i];
            q[n - 1 - i] = t;
        }
    }
    vec4 normals[4];
    for (int i = 0; i < n; ++i) {
        vec2 d = q[(i + 1) % n] - q[i], len = dlength(d, vec2(0));
        normals[i] = vec4(dd(ds(d.y), len), dd(ds(-d.x), len));
    }
    vec2 tolerance = dvalue(delta) < 1. ? dm(delta, ds(0.25)) : ds(0.25);
    vec2 steps = dd(dpi(), dacos(da(ds(1.), -dd(tolerance, delta))));
    vec2 limit = dm(delta, dpi());
    if (dvalue(steps) > dvalue(limit))
        steps = limit;
    vec2 angle = dd(dm(ds(2.), dpi()), steps);
    vec4 sc = dsincos(angle);
    uint written = 0u;
    int prev = n - 1;
    for (int i = 0; i < n; ++i) {
        vec4 normal = normals[prev], end = normals[i];
        vec2 cross_val = da(dm(normal.xy, end.zw), -dm(normal.zw, end.xy));
        vec2 dot_val = da(dm(normal.xy, end.xy), dm(normal.zw, end.zw));
        if (abs(dvalue(dm(cross_val, delta))) < 1. && dvalue(dot_val) > 0.) {
            hull[base + written++] = offset_point(q[i], normal, delta);
            continue;
        }
        if (dvalue(cross_val) < 0.) {
            hull[base + written++] = offset_point(q[i], normal, delta);
            hull[base + written++] = q[i];
        } else {
            vec2 arc = datan2_(cross_val, dot_val);
            uint count = max(1u, uint(floor(abs(dvalue(dd(arc, angle))) + 0.5)));
            // The temporary offset hull occupies the first 512 slots.
            uint stride = max(1u, (count + 119u) / 120u);
            if (stride > 1u)
                sc = dsincos(dm(angle, ds(float(stride))));
            for (uint j = 0u; j < count; j += stride) {
                hull[base + written++] = offset_point(q[i], normal, delta);
                normal = vec4(da(dm(normal.xy, sc.zw), -dm(normal.zw, sc.xy)),
                              da(dm(normal.xy, sc.xy), dm(normal.zw, sc.zw)));
            }
            if (stride > 1u)
                sc = dsincos(angle);
        }
        hull[base + written++] = offset_point(q[i], end, delta);
        prev = i;
    }
    return written;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
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
    if (!(confidence >= uintBitsToFloat(p(2u))))
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
        vec2 mapped =
            vec2(divide(r.points[i].x, float(width)), divide(r.points[i].y, float(height_limit))) *
            vec2(p(6u), p(7u));
        r.points[i] = clamp(roundEven(mapped), vec2(0), vec2(p(4u) - 1u, p(5u) - 1u));
    }
    r.points = ordered(r.points);
    if (p(10u) != 0u) {
        for (uint i = 0u; i < 4u; ++i) {
            vec2 a = r.points[(i + 1u) % 4u] - r.points[i];
            vec2 b = r.points[(i + 2u) % 4u] - r.points[(i + 1u) % 4u];
            if (a.x * b.y - a.y * b.x <= 0.)
                return;
        }
    }
    if (distance(r.points[0], r.points[1]) < 4. || distance(r.points[0], r.points[3]) < 4.)
        return;
    for (uint i = 0u; i < 4u; ++i) {
        boxes[outbox + i * 2u] = floatBitsToUint(r.points[i].x);
        boxes[outbox + i * 2u + 1u] = floatBitsToUint(r.points[i].y);
    }
    boxes[outbox + 8u] = floatBitsToUint(confidence);
    boxes[outbox + 9u] = 1u;
}
