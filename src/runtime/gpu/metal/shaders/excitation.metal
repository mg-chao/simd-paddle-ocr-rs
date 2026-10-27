#define SCRATCH_FLOATS 65
void run(uint3 id, uint3 group, uint lane, threadgroup float *partial) const {
  uint row = group.x + group.y * p(1);
  if (row >= p(0))
    return;
  uint inner = p(3), channels = p(4), plane = p(5), channel = row % channels,
       batch = row / channels;
  float value = 0.0f;
  if (SMALL_PROJECTION && lane == 0 && p(19))
    value = load_c(channel);
  if (!SMALL_PROJECTION && p(12) == 1 && p(13) == 0) {
    if (lane == 0)
      for (uint i = 0; i < inner; i++)
        value += load_a(batch * inner + i) * load_b(channel * inner + i);
  } else {
    for (uint i = lane; i < inner; i += 64)
      value += load_a(batch * inner + i) * load_b(channel * inner + i);
    partial[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride /= 2) {
      if (lane < stride)
        partial[lane] += partial[lane + stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    value = partial[0];
  }
  if (SMALL_PROJECTION) {
    if (lane == 0)
      store_output(row, epilogue(value, row));
    return;
  }
  if (lane == 0) {
    if (p(12) == 1)
      value += load_c(p(11) ? 0 : channel);
    else if (p(7))
      value += load_c(channel);
    float gate =
        p(8) == 1 ? clamp(value * as_type<float>(p(9)) + as_type<float>(p(10)),
                          0.0f, 1.0f)
                  : (p(12) == 1 ? sigmoid(value) : 1.0f / (1.0f + exp(-value)));
    partial[64] = p(12) == 1 ? output_round(gate) : gate;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < plane; i += 64)
    store_output(row * plane + i, load_d(row * plane + i) * partial[64]);
}
