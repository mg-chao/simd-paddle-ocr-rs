#define SCRATCH_FLOATS 64
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
void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  uint row = group.x + group.y * p(1);
  if (row >= p(0))
    return;
  uint columns = p(3), base = row * columns;
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
