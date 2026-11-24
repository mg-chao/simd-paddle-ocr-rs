// Native Metal buffer and parameter contract for crop.
struct Resident {
  device const uint *pixels;
  device const uint *transforms;
  device uint *output;
  constant Params &params;
  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  uint t(uint i) { return transforms[p(5u) * 32u + i]; }

  int3 bgr(int2 at) {
    auto x = uint(clamp(at.x, 0, int(p(0u)) - 1));
    auto y = uint(clamp(at.y, 0, int(p(1u)) - 1));
    auto offset = y * p(2u) + x * p(3u);
    int3 color;
    for (uint c = 0; c < 3u; ++c) {
      uint byte = offset + c;
      color[c] = int((pixels[byte / 4u] >> ((byte % 4u) * 8u)) & 255u);
    }
    return p(4u) != 0u ? color.zyx : color;
  }

  uint pack_bgr(int3 color) {
    auto c = uint3(clamp(color, int3(0), int3(255)));
    return 0xff000000u | c.x | (c.y << 8u) | (c.z << 16u);
  }

  // Keep the 15-bit coefficient construction and correction in the same
  // order as the CPU and Vulkan cubic samplers.
  float4 axis_weights(int phase) {
    float x = float(phase) / 32.0;
    float y = 1.0 - x;
    float a = -0.75;
    float first = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) *
                    (x + 1.0) - 4.0 * a;
    float second = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    float third = ((a + 2.0) * y - (a + 3.0)) * y * y + 1.0;
    return float4(first, second, third, 1.0 - first - second - third);
  }

  void run(uint3 id, uint3 grid) {
    uint index = id.x + id.y * grid.x * 64u;
    bool from_grid = p(6u) != 0u;
    uint width = from_grid ? p(7u) : t(0u);
    uint height = from_grid ? p(8u) : t(1u);
    if (index >= width * height) {
      return;
    }

    if (!from_grid) {
      // The host uses the compact transform for integer translations.
      uint ox = index % width, oy = index / width;
      bool vertical = t(4u) != 0u;
      int x = int(vertical ? t(2u) - 1u - oy : ox);
      int y = int(vertical ? ox : oy);
      int2 origin = int2(as_type<float>(t(12u)), as_type<float>(t(18u)));
      output[index] = pack_bgr(bgr(origin + int2(x, y)));
      return;
    }

    int2 fixed_xy = as_type<int2>(
        uint2(transforms[index * 2u], transforms[index * 2u + 1u]));
    int2 base = fixed_xy >> 5;
    int2 phase = fixed_xy & 31;
    if (all(phase == int2(0))) {
      output[index] = pack_bgr(bgr(base));
      return;
    }

    float4 wx = axis_weights(phase.x), wy = axis_weights(phase.y);
    int3 sum = int3(1 << 14);
    int total = 0, largest = -32769, smallest = 32768;
    int2 high = int2(0), low = int2(0);
    for (int ky = 0; ky < 4; ++ky) {
      int4 weights = clamp(int4(rint(wx * wy[ky] * 32768.0)), int4(-32768),
                           int4(32767));
      for (int kx = 0; kx < 4; ++kx) {
        int2 at = base + int2(kx - 1, ky - 1);
        int weight = weights[kx];
        sum += weight * bgr(at);
        total += weight;
        if (kx >= 2 && ky >= 2) {
          if (weight > largest) {
            largest = weight;
            high = at;
          }
          if (weight < smallest) {
            smallest = weight;
            low = at;
          }
        }
      }
    }
    int delta = 32768 - total;
    if (delta != 0) {
      sum += delta * bgr(delta > 0 ? high : low);
    }
    output[index] = pack_bgr(sum >> 15);
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
