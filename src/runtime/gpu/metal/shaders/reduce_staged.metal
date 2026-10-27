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
  uint flat = group.x + group.y * p(1), columns = p(3), chunks = p(4),
       row = FINISH ? flat : flat / chunks, chunk = FINISH ? 0 : flat % chunks;
  if (row >= p(0))
    return;
  float value = SOFTMAX ? -FLT_MAX : 0.0f;
  if (FINISH) {
    for (uint part = lane; part < chunks; part += 64) {
      float x = load_b((row * chunks + part) * 2);
      value = SOFTMAX ? max(value, x) : value + x;
    }
    float maximum = reduce(value, lane, scratch, SOFTMAX);
    if (!SOFTMAX) {
      if (lane == 0)
        store_output(row, maximum / float(columns));
      return;
    }
    value = 0.0f;
    for (uint part = lane; part < chunks; part += 64) {
      uint offset = (row * chunks + part) * 2;
      value += load_b(offset + 1) * exp(load_b(offset) - maximum);
    }
    float denominator = reduce(value, lane, scratch, false);
    for (uint col = lane; col < columns; col += 64)
      store_output(row * columns + col,
                   exp(load_a(row * columns + col) - maximum) / denominator);
  } else {
    uint end = min(columns, (chunk + 1) * 1024), start = chunk * 1024;
    for (uint col = start + lane; col < end; col += 64) {
      float x = load_a(row * columns + col);
      value = SOFTMAX ? max(value, x) : value + x;
    }
    float maximum = reduce(value, lane, scratch, SOFTMAX);
    if (lane == 0)
      store_output(flat * 2, maximum);
    if (SOFTMAX) {
      value = 0.0f;
      for (uint col = start + lane; col < end; col += 64)
        value += exp(load_a(row * columns + col) - maximum);
      float sum = reduce(value, lane, scratch, false);
      if (lane == 0)
        store_output(flat * 2 + 1, sum);
    }
  }
}
