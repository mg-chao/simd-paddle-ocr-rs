#define SCRATCH_FLOATS ((MODE == 0) ? 1024 : 64)
float reduce(float value, uint lane, threadgroup float *values,
             bool maximum) const {
  values[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32; stride > 0; stride /= 2) {
    if (lane < stride)
      values[lane] = maximum ? max(values[lane], values[lane + stride])
                             : values[lane] + values[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float result = values[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}
// Keep the eight ARM ReduceMean chains in registers. Each shuffle supplies
// the next element of a chain from a coalesced 32-float load.
float packet_sum(uint base, uint length, uint lane) const {
#pragma clang fp reassociate(off)
  float sum = 0.f;
  uint start = 0u;
  for (; length - start >= 256u; start += 256u) {
    float values[8];
#pragma unroll
    for (uint tile = 0u; tile < 8u; ++tile)
      values[tile] = load_a(base + start + tile * 32u + lane);
#pragma unroll
    for (uint tile = 0u; tile < 8u; ++tile) {
#pragma unroll
      for (uint packet = 0u; packet < 4u; ++packet)
        sum += simd_shuffle(values[tile], ushort(packet * 8u + lane % 8u));
    }
  }
  if (lane < 8u)
    for (uint i = start + lane; i < length; i += 8u)
      sum += load_a(base + i);
  return sum;
}

// Each threadgroup owns one plane. Short spatial means and GlobalAveragePool
// use a 64-thread staged load. Long spatial means use a 32-thread SIMD group.
void spatial_mean(uint row, uint lane, uint plane, threadgroup float *partial) const {
#pragma clang fp reassociate(off)
  const bool pool = MODE == 0 && p(6) == 2u;
  uint base = row * plane, prefix = pool ? 0u : (0u - base) % 4u;
  uint packets = (plane - prefix) / 4u;
  uint stride = pool ? 4u : 8u;
  uint length = (plane - prefix) / stride * stride;
  float sum = 0.f;
  if (MODE == 0) {
    for (uint start = 0u; start < length; start += 1024u) {
      uint count = min(1024u, length - start);
      for (uint j = lane; j < count; j += 64u)
        partial[j] = load_a(base + prefix + start + j);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane < stride)
        for (uint j = lane; j < count; j += stride)
          sum += partial[j];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    sum = packet_sum(base + prefix, length, lane);
  }
  if (lane < stride)
    partial[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 4u && !pool) {
    float v = partial[lane] + partial[lane + 4u];
    if (packets % 2u)
      v += load_a(base + prefix + length + lane);
    partial[lane] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    // The original Metal kernel folded short ReduceMean planes from left to
    // right. Preserve that ordering for threshold-sensitive recognition scores.
    float lo = partial[0] + partial[1];
    float total = MODE == 0 && !pool ? (lo + partial[2]) + partial[3]
                                     : lo + (partial[2] + partial[3]);
    for (uint j = 0u; j < prefix; ++j)
      total += load_a(base + j);
    for (uint j = prefix + 4u * packets; j < plane; ++j)
      total += load_a(base + j);
    float divisor = float(plane), quotient = total / divisor;
    if (isfinite(quotient))
      quotient = fma(fma(-quotient, divisor, total), 1.f / divisor, quotient);
    store_output(row, quotient);
  }
}
void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  uint row = group.x + group.y * p(1);
  if (row >= p(0))
    return;
  uint columns = p(3), base = row * columns;
  if (MODE == 3 || (MODE == 0 && p(6) != 0u)) {
    spatial_mean(row, lane, columns, scratch);
    return;
  }
  float value = MODE == 1 ? -FLT_MAX : 0.0f;
  for (uint col = lane; col < columns; col += 64) {
    float x = load_a(base + col);
    value = MODE == 1 ? max(value, x) : value + x;
  }
  float mean = reduce(value, lane, scratch, MODE == 1);
  if (MODE == 0) {
    if (lane == 0)
      store_output(row, mean / float(columns));
    return;
  }
  if (MODE == 2)
    mean /= float(columns);
  value = 0.0f;
  for (uint col = lane; col < columns; col += 64) {
    float centered = load_a(base + col) - mean;
    value += MODE == 1 ? exp(centered) : centered * centered;
  }
  float scale = reduce(value, lane, scratch, false);
  if (MODE == 2)
    scale = sqrt(scale / float(columns) + as_type<float>(p(4)));
  for (uint col = lane; col < columns; col += 64) {
    float v = MODE == 1 ? exp(load_a(base + col) - mean) / scale
                        : ((load_a(base + col) - mean) / scale) * load_b(col) +
                              load_c(col);
    store_output(base + col, v);
  }
}
