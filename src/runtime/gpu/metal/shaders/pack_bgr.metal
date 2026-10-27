// Native Metal buffer and parameter contract for pack_bgr.
struct Resident {
  device const uint *input;
  device uint *output;
  constant Params &params;

  void run(uint3 id, uint3 grid) {
    auto word = id.x + id.y * grid.x * 64u;
    auto bytes = params.words[0].x;
    if (word * 4u >= bytes) {
      return;
    }
    auto value = 0u;
    for (auto b = 0u; b < 4u; b++) {
      auto index = word * 4u + b;
      if (index < bytes) {
        value |= ((input[index / 3u] >> ((index % 3u) * 8u)) & 255u)
                 << (b * 8u);
      }
    }
    output[word] = value;
  }
};
kernel void pack_bgr(device const uint *input [[buffer(0)]],
                     device uint *output [[buffer(1)]],
                     constant Params &params [[buffer(2)]],
                     uint3 id [[thread_position_in_grid]],
                     uint3 grid [[threadgroups_per_grid]]) {

  Resident state{input, output, params};
  state.run(id, grid);
}
