#define SCRATCH_FLOATS 1228

uint ap(uint i) const {
  return p(i + select(0u, 1u, i >= 60u) + select(0u, 4u, i >= 63u));
}
uint address(uint index, uint operand) const {
  if (!VIEWS) {
    return index;
  }
  auto value = index;
  auto start = p(9u + operand * 2u);
  auto count = p(10u + operand * 2u);
  for (auto transform = 0u; transform < count; transform++) {
    auto remaining = value;
    auto mapped = ap(start + 1u);
    auto axis = ap(start);
    while (true) {
      if (axis == 0u) {
        break;
      }
      axis--;
      auto c = remaining % ap(start + 2u + axis);
      remaining /= ap(start + 2u + axis);
      mapped += c * ap(start + 7u + axis);
    }
    value = mapped;
    start += 12u;
  }
  return value;
}

void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  threadgroup float *queries =
      reinterpret_cast<threadgroup float *>(scratch + 0);
  threadgroup float *keys =
      reinterpret_cast<threadgroup float *>(scratch + 128);
  threadgroup float *values =
      reinterpret_cast<threadgroup float *>(scratch + 640);
  threadgroup float *weights =
      reinterpret_cast<threadgroup float *>(scratch + 1152);
  threadgroup float *maxima =
      reinterpret_cast<threadgroup float *>(scratch + 1216);
  threadgroup float *denominators =
      reinterpret_cast<threadgroup float *>(scratch + 1220);
  threadgroup float *corrections =
      reinterpret_cast<threadgroup float *>(scratch + 1224);
  auto block = group.x + group.y * p(1u);
  if (block >= p(0u)) {
    return;
  }
  auto batch = block / p(8u);
  auto first = (block % p(8u)) * 4u;
  auto query = lane / 16u;
  auto channel = lane % 16u;
  auto row = first + query;
  auto rows = p(3u);
  auto count = p(4u);
  auto d = p(5u);
  auto dv = p(6u);
  auto packed = VIEWS && (p(2u) & 2u) != 0u;
  auto packed_stride = 0u;
  auto packed_base = 0u;
  if (packed) {
    packed_stride = 3u * p(15u) * d;
    packed_base =
        (batch / p(15u)) * rows * packed_stride + (batch % p(15u)) * d;
  }
  for (auto i = lane; i < 128u; i += 64u) {
    auto r = first + i / 32u;
    auto c = i % 32u;
    queries[i] = 0.0;
    if (r < rows && c < d) {
      auto value = 0.0;
      if (packed) {
        value = load_a(packed_base + r * packed_stride + c);
      } else {
        value = load_a(address((batch * rows + r) * d + c, 0u));
      }
      if (VIEWS && (p(2u) & 1u) != 0u) {
        value = output_round(value * load_d(0u));
      }
      queries[i] = value * as_type<float>(p(7u));
    }
  }
  if (lane < 4u) {
    maxima[lane] = -3.402823466e38;
    denominators[lane] = 0.0;
  }
  auto acc0 = 0.0;
  auto acc1 = 0.0;
  for (auto base = 0u; base < count; base += 16u) {
    for (auto i = lane; i < 512u; i += 64u) {
      auto k = base + i / 32u;
      auto c = i % 32u;
      keys[i] = 0.0;
      values[i] = 0.0;
      if (k < count && c < d) {
        if (packed) {
          keys[i] = load_b(packed_base + k * packed_stride + p(15u) * d + c);
        } else {
          keys[i] = load_b(address((batch * d + c) * count + k, 1u));
        }
      }
      if (k < count && c < dv) {
        if (packed) {
          values[i] =
              load_c(packed_base + k * packed_stride + 2u * p(15u) * d + c);
        } else {
          values[i] = load_c(address((batch * count + k) * dv + c, 2u));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto score = 0.0;
    for (auto i = 0u; i < 32u; i++) {
      score += queries[query * 32u + i] * keys[channel * 32u + i];
    }
    weights[lane] = score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto valid = min(16u, count - base);
    if (channel == 0u) {
      auto next = maxima[query];
      for (auto k = 0u; k < valid; k++) {
        next = max(next, weights[query * 16u + k]);
      }
      auto correction = exp(maxima[query] - next);
      auto total = 0.0;
      for (auto k = 0u; k < valid; k++) {
        auto weight = exp(weights[query * 16u + k] - next);
        weights[query * 16u + k] = weight;
        total += weight;
      }
      corrections[query] = correction;
      maxima[query] = next;
      denominators[query] = denominators[query] * correction + total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc0 *= corrections[query];
    acc1 *= corrections[query];
    for (auto k = 0u; k < valid; k++) {
      auto weight = weights[query * 16u + k];
      acc0 += weight * values[k * 32u + channel];
      acc1 += weight * values[k * 32u + channel + 16u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < rows) {
    auto output_base = (batch * rows + row) * dv;
    if (packed && (p(2u) & 4u) != 0u) {
      output_base =
          ((batch / p(15u)) * rows * p(15u) + row * p(15u) + batch % p(15u)) *
          dv;
    }
    if (channel < dv) {
      store_output(output_base + channel, acc0 / denominators[query]);
    }
    if (channel + 16u < dv) {
      store_output(output_base + channel + 16u, acc1 / denominators[query]);
    }
  }
}
