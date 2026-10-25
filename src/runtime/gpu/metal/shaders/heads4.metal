// Native Metal buffer and parameter contract for heads4.
struct Resident {
  device const INPUT *features;
  device const float *weights;
  device const float *bias;
  device uint4 *partial;
  constant Params &params;
  threadgroup float4 *scores;
  threadgroup uint4 *indices;
  threadgroup float4 *sums;
  threadgroup uint4 *invalid;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }

  void run(uint lane, uint3 group) {
    auto column = group.x * 64u + lane;
    auto row = group.y * 4u;
    auto value = float4(-3.402823e38);
    if (column < p(2u)) {
      value = float4(bias[column]);
      for (auto k = 0u; k < p(1u); k++) {
        auto weight = weights[k * p(2u) + column];
        value.x += float(features[row * p(1u) + k]) * weight;
        if (row + 1u < p(0u)) {
          value.y += float(features[(row + 1u) * p(1u) + k]) * weight;
        }
        if (row + 2u < p(0u)) {
          value.z += float(features[(row + 2u) * p(1u) + k]) * weight;
        }
        if (row + 3u < p(0u)) {
          value.w += float(features[(row + 3u) * p(1u) + k]) * weight;
        }
      }
    }
    scores[lane] = value;
    indices[lane] = uint4(column);
    invalid[lane] = select(
        uint4(0u), uint4(1u),
        !((value <= float4(3.402823e38)) & (value >= float4(-3.402823e38))));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (auto stride = 32u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        auto other = scores[lane + stride];
        auto index = indices[lane + stride];
        auto replace = (other > scores[lane]) |
                       ((other == scores[lane]) & (index < indices[lane]));
        scores[lane] = select(scores[lane], other, replace);
        indices[lane] = select(indices[lane], index, replace);
        invalid[lane] |= invalid[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p(4u) == 0u) {
      if (lane == 0u) {
        for (auto r = 0u; r < 4u; r++) {
          if (row + r < p(0u)) {
            partial[(row + r) * p(3u) + group.x] = uint4(
                indices[0][r], as_type<uint>(scores[0][r]), 0u, invalid[0][r]);
          }
        }
      }
      return;
    }
    sums[lane] = float4(0.0);
    if (column < p(2u) && p(4u) != 0u) {
      sums[lane] = exp(value - scores[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (auto stride = 32u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        sums[lane] += sums[lane + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
      for (auto r = 0u; r < 4u; r++) {
        if (row + r < p(0u)) {
          partial[(row + r) * p(3u) + group.x] =
              uint4(indices[0][r], as_type<uint>(scores[0][r]),
                    as_type<uint>(sums[0][r]), invalid[0][r]);
        }
      }
    }
  }
};
kernel void heads4(device const INPUT *features [[buffer(0)]],
                   device const float *weights [[buffer(1)]],
                   device const float *bias [[buffer(2)]],
                   device uint4 *partial [[buffer(3)]],
                   constant Params &params [[buffer(4)]],
                   uint lane [[thread_index_in_threadgroup]],
                   uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup float4 scores[64];
  threadgroup uint4 indices[64];
  threadgroup float4 sums[64];
  threadgroup uint4 invalid[64];
  Resident state{features, weights, bias, partial, params,
                 scores,   indices, sums, invalid};
  state.run(lane, group);
}
