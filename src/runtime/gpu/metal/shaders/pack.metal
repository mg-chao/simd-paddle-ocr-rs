void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  uint index = id.x + id.y * p(1);
  if (index >= p(0))
    return;
  uint channels = p(3), plane = p(4), physical = p(5),
       channel = index % physical, pixel = (index / physical) % plane,
       batch = index / (physical * plane);
  float value = 0.0f;
  if (channel < channels) {
    value = load_a((batch * channels + channel) * plane + pixel);
    if (p(6))
      value *= load_b(batch * channels + channel);
  }
  store_output(index, value);
}
