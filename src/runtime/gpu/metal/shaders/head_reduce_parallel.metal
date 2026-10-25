// Native Metal buffer and parameter contract for head_reduce_parallel.
struct Resident {
  device const uint4 *partial;
  device uint4 *winners;
  constant Params &params;
  threadgroup uint4 *selected;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }

  bool better(uint4 a, uint4 b) {
    auto x = as_type<float>(a.y);
    auto y = as_type<float>(b.y);
    return x > y || (x == y && (a.x < b.x || (a.x == b.x && a.z < b.z)));
  }
  void run(uint3 group, uint lane) {
    auto row = group.x;
    if (row >= p(0u) || p(3u) > 512u) {
      return;
    }
    auto maximum = -3.402823e38;
    auto local = uint4(0xffffffffu, as_type<uint>(maximum), 0xffffffffu, 0u);
    for (auto i = lane; i < p(3u); i += 64u) {
      auto item = partial[row * p(3u) + i];
      local.w |= item.w;
      auto candidate = uint4(item.x, item.y, i, 0u);
      if (better(candidate, local)) {
        local.x = candidate.x;
        local.y = candidate.y;
        local.z = candidate.z;
      }
    }
    selected[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (auto stride = 32u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        auto a = selected[lane];
        auto b = selected[lane + stride];
        auto choice = a;
        if (better(b, a)) {
          choice.x = b.x;
          choice.y = b.y;
          choice.z = b.z;
        }
        choice.w = a.w | b.w;
        selected[lane] = choice;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
      auto winner = selected[0];
      winners[row] = uint4(winner.x, winner.y, 0u, winner.w);
    }
  }
};
kernel void head_reduce_parallel(device const uint4 *partial [[buffer(0)]],
                                 device uint4 *winners [[buffer(1)]],
                                 constant Params &params [[buffer(2)]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint4 selected[64];
  Resident state{partial, winners, params, selected};
  state.run(group, lane);
}
