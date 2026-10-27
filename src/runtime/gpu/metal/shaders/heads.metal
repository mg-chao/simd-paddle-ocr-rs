// Native Metal buffer and parameter contract for heads.
struct Resident {
  device const INPUT *features;
  device const float *weights;
  device const float *bias;
  device uint4 *partial;
  constant Params &params;
  threadgroup float *scores;
  threadgroup uint *indices;
  threadgroup float *sums;
  threadgroup uint *invalid;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }

  void run(uint lane, uint3 group) {
    auto column = group.x * 64u + lane;
    auto row = group.y;
    auto value = -3.402823e38;
    if (column < p(2u)) {
      value = bias[column];
      for (auto k = 0u; k < p(1u); k++) {
        value = value +
                float(features[row * p(1u) + k]) * weights[k * p(2u) + column];
      }
    }
    scores[lane] = value;
    indices[lane] = column;
    invalid[lane] = uint(!(value <= 3.402823e38 && value >= -3.402823e38));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (auto stride = 32u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        auto other = scores[lane + stride];
        if (other > scores[lane] ||
            (other == scores[lane] && indices[lane + stride] < indices[lane])) {
          scores[lane] = other;
          indices[lane] = indices[lane + stride];
        }
        invalid[lane] = invalid[lane] | invalid[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p(4u) == 0u) {
      if (lane == 0u) {
        partial[row * p(3u) + group.x] =
            uint4(indices[0], as_type<uint>(scores[0]), 0u, invalid[0]);
      }
      return;
    }
    sums[lane] =
        select(0.0, exp(value - scores[0]), column < p(2u) && p(4u) != 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (auto stride = 32u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        sums[lane] += sums[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
      partial[row * p(3u) + group.x] =
          uint4(indices[0], as_type<uint>(scores[0]), as_type<uint>(sums[0]),
                invalid[0]);
    }
  }
};
kernel void heads(device const INPUT *features [[buffer(0)]],
                  device const float *weights [[buffer(1)]],
                  device const float *bias [[buffer(2)]],
                  device uint4 *partial [[buffer(3)]],
                  constant Params &params [[buffer(4)]],
                  uint lane [[thread_index_in_threadgroup]],
                  uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup float scores[64];
  threadgroup uint indices[64];
  threadgroup float sums[64];
  threadgroup uint invalid[64];
  Resident state{features, weights, bias, partial, params,
                 scores,   indices, sums, invalid};
  state.run(lane, group);
}
