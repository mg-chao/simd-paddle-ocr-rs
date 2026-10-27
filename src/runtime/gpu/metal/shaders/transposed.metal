#define SCRATCH_FLOATS 640

void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  threadgroup float4 *weights =
      reinterpret_cast<threadgroup float4 *>(scratch + 0);
  threadgroup float *pixels =
      reinterpret_cast<threadgroup float *>(scratch + 512);
  auto block = group.x + group.y * p(35u);
  if (block >= p(34u)) {
    return;
  }
  auto batch = block / (p(32u) * p(33u));
  auto co = (block / p(33u) % p(32u)) * 8u + lane / 8u;
  auto pixel = (block % p(33u)) * 8u + lane % 8u;
  auto cin = p(3u);
  auto cout = p(4u);
  auto h = p(5u);
  auto w = p(6u);
  auto plane = h * w;
  auto sum = float4(0.0);
  if (co < cout && p(7u) != 0u) {
    sum = float4(load_c(co));
  }
  for (auto base = 0u; base < cin; base += 16u) {
    for (auto i = lane; i < 128u; i += 64u) {
      auto row = (block / p(33u) % p(32u)) * 8u + i / 16u;
      auto ci = base + i % 16u;
      weights[i] = float4(0.0);
      if (row < cout && ci < cin) {
        auto index = (ci * cout + row) * 4u;
        weights[i] = float4(load_b(index), load_b(index + 1u),
                            load_b(index + 2u), load_b(index + 3u));
      }
      auto source = (block % p(33u)) * 8u + i % 8u;
      auto channel = base + i / 8u;
      pixels[i] = 0.0;
      if (source < plane && channel < cin) {
        pixels[i] = load_a((batch * cin + channel) * plane + source);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (auto k = 0u; k < 16u; k++) {
      sum += weights[(lane / 8u) * 16u + k] * pixels[k * 8u + lane % 8u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (co < cout && pixel < plane) {
    auto index = (batch * cout + co) * plane * 4u + (pixel / w) * w * 4u +
                 (pixel % w) * 2u;
    store_output(index, epilogue<true>(sum.x, index, co));
    store_output(index + 1u, epilogue<true>(sum.y, index + 1u, co));
    store_output(index + w * 2u, epilogue<true>(sum.z, index + w * 2u, co));
    store_output(index + w * 2u + 1u, epilogue<true>(sum.w, index + w * 2u + 1u, co));
  }
}
