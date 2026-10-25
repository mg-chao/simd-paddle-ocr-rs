// Native Metal buffer and parameter contract for ctc.
struct Resident {
  device const uint4 *winners;
  device const uint *lengths;
  device uint *output;
  constant Params &params;
  threadgroup uint *scan;
  threadgroup uint &emitted;
  threadgroup uint &bytes;
  threadgroup atomic_uint &error;
  threadgroup float &sum;
  threadgroup float &correction;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }

  float score_round(float x) {
    return as_type<float>(as_type<uint>(x) & p(255u));
  }

  void run(uint lane, uint3 group) {
    auto seq = group.x;
    auto steps = p(1u);
    auto base = seq * (steps + 1u) * 6u;
    if (lane == 0u) {
      emitted = 0u;
      bytes = 0u;
      sum = 0.0;
      correction = 0.0;
      atomic_store_explicit(&error, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (auto tile = 0u; tile < steps; tile += 64u) {
      auto step = tile + lane;
      auto item = uint4(0u);
      auto marked = false;
      if (step < steps) {
        item = winners[seq * steps + step];
        if (item.x >= p(2u) || item.w != 0u) {
          atomic_fetch_or_explicit(&error, 1u, memory_order_relaxed);
        }
        auto previous = 0u;
        if (step > 0u) {
          previous = winners[seq * steps + step - 1u].x;
        }
        marked = item.x != 0u && item.x != previous && item.x < p(2u);
      }
      scan[lane] = uint(marked);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (auto delta = 1u; delta < 64u; delta *= 2u) {
        auto value = 0u;
        if (lane >= delta) {
          value = scan[lane - delta];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[lane] += value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (marked) {
        auto record = base + (emitted + scan[lane]) * 6u;
        auto end = step + 1u;
        while (end < steps && winners[seq * steps + end].x == item.x) {
          end++;
        }
        output[record] = item.x;
        output[record + 1u] = item.y;
        output[record + 2u] = step;
        output[record + 3u] = end;
        output[record + 4u] = lengths[item.x];
        output[record + 5u] = 0u;
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
      if (lane == 0u) {
        for (auto i = 0u; i < scan[63]; i++) {
          auto record = base + (emitted + i + 1u) * 6u;
          auto add = output[record + 4u];
          if (bytes > 0xffffffffu - add) {
            atomic_fetch_or_explicit(&error, 2u, memory_order_relaxed);
          } else {
            bytes += add;
          }
          auto value =
              score_round(as_type<float>(output[record + 1u]) - correction);
          auto next = score_round(sum + value);
          correction = score_round(score_round(next - sum) - value);
          sum = next;
        }
        emitted += scan[63];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
      auto span_bytes = select(0u, emitted * p(5u), p(3u) != 0u);
      if (bytes > p(4u) || span_bytes > p(4u) - min(bytes, p(4u))) {
        atomic_fetch_or_explicit(&error, 2u, memory_order_relaxed);
      }
      output[base] = emitted;
      output[base + 1u] = bytes;
      output[base + 2u] = as_type<uint>(
          select(0.0, sum / float(max(emitted, 1u)), emitted > 0u));
      output[base + 3u] = atomic_load_explicit(&error, memory_order_relaxed);
      output[base + 4u] = steps;
      output[base + 5u] = 0u;
    }
  }
};
kernel void ctc(device const uint4 *winners [[buffer(0)]],
                device const uint *lengths [[buffer(1)]],
                device uint *output [[buffer(2)]],
                constant Params &params [[buffer(3)]],
                uint lane [[thread_index_in_threadgroup]],
                uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup uint scan[64];
  threadgroup uint emitted;
  threadgroup uint bytes;
  threadgroup atomic_uint error;
  threadgroup float sum;
  threadgroup float correction;
  Resident state{winners, lengths, output, params, scan,
                 emitted, bytes,   error,  sum,    correction};
  state.run(lane, group);
}
