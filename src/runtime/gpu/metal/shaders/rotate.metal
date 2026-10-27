// Native Metal buffer and parameter contract for rotate.
struct Resident {
  device const uint *pixels;
  device const uint *winner;
  device uint *output;
  constant Params &params;

  void run(uint3 id, uint3 grid) {
    auto i = id.x + id.y * grid.x * 64u;
    auto n = params.words[0].x;
    if (i >= n) {
      return;
    }
    auto row = params.words[0].z * 4u;
    auto rotate = winner[row] == 1u && as_type<float>(winner[row + 1u]) >
                                           as_type<float>(params.words[0].y);
    output[i] = pixels[select(i, n - 1u - i, rotate)];
  }
};
kernel void rotate(device const uint *pixels [[buffer(0)]],
                   device const uint *winner [[buffer(1)]],
                   device uint *output [[buffer(2)]],
                   constant Params &params [[buffer(3)]],
                   uint3 id [[thread_position_in_grid]],
                   uint3 grid [[threadgroups_per_grid]]) {

  Resident state{pixels, winner, output, params};
  state.run(id, grid);
}
