// Native Metal buffer and parameter contract for head_reduce.
struct Resident {
  device const uint4 *partial;
  device uint4 *winners;
  constant Params &params;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  void run(uint3 id) {
    auto row = id.x;
    if (row >= p(0u)) {
      return;
    }
    auto best = 0xffffffffu;
    auto maximum = -3.402823e38;
    auto invalid = 0u;
    for (auto i = 0u; i < p(3u); i++) {
      auto item = partial[row * p(3u) + i];
      auto value = as_type<float>(item.y);
      invalid |= item.w;
      if (value > maximum || (value == maximum && item.x < best)) {
        maximum = value;
        best = item.x;
      }
    }
    auto score = maximum;
    if (p(4u) != 0u) {
      auto sum = 0.0;
      for (auto i = 0u; i < p(3u); i++) {
        auto item = partial[row * p(3u) + i];
        sum += as_type<float>(item.z) * exp(as_type<float>(item.y) - maximum);
      }
      score = 1.0 / sum;
    }
    winners[row] = uint4(best, as_type<uint>(score), 0u, invalid);
  }
};
kernel void head_reduce(device const uint4 *partial [[buffer(0)]],
                        device uint4 *winners [[buffer(1)]],
                        constant Params &params [[buffer(2)]],
                        uint3 id [[thread_position_in_grid]]) {

  Resident state{partial, winners, params};
  state.run(id);
}
