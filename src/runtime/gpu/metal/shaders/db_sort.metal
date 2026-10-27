// Native Metal buffer and parameter contract for db_sort.
struct Resident {
  device const uint *boxes;
  device uint *output;
  constant Params &params;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  void swap(uint a, uint b) {
    for (auto k = 0u; k < 16u; k++) {
      auto v = output[(a + 1u) * 16u + k];
      output[(a + 1u) * 16u + k] = output[(b + 1u) * 16u + k];
      output[(b + 1u) * 16u + k] = v;
    }
  }
  float2 point(uint i) {
    return float2(as_type<float>(output[(i + 1u) * 16u]),
                  as_type<float>(output[(i + 1u) * 16u + 1u]));
  }
  bool less(float2 a, float2 b) {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
  }
  void swap_greater(uint a, uint b) {
    if (less(point(b), point(a))) {
      swap(a, b);
    }
  }
  void down_heap(uint base, uint start, uint n) {
    auto i = start;
    while (true) {
      if (i > n / 2u) {
        break;
      }
      auto child = 2u * i;
      if (child < n && less(point(base + child - 1u), point(base + child))) {
        child++;
      }
      if (!less(point(base + i - 1u), point(base + child - 1u))) {
        break;
      }
      swap(base + i - 1u, base + child - 1u);
      i = child;
    }
  }
  void canonical_sort(uint count) {
    if (count < 2u) {
      return;
    }
    array<uint3, 64> stack{};
    auto top = 1u;
    stack[0] = uint3(0u, count, 2u * (32u - clz(count)));
    while (true) {
      if (top == 0u) {
        break;
      }
      top--;
      auto entry = stack[top];
      auto base = entry.x;
      auto n = entry.y;
      auto depth = entry.z;
      if (n < 2u) {
        continue;
      }
      if (n <= 16u) {
        if (n == 2u) {
          swap_greater(base, base + 1u);
        } else if (n == 3u) {
          swap_greater(base, base + 1u);
          swap_greater(base, base + 2u);
          swap_greater(base + 1u, base + 2u);
        } else {
          for (auto i = 1u; i < n; i++) {
            auto j = i;
            while (j > 0u && less(point(base + j), point(base + j - 1u))) {
              swap(base + j, base + j - 1u);
              j--;
            }
          }
        }
        continue;
      }
      if (depth == 0u) {
        for (auto i = n / 2u; i > 0u; i--) {
          down_heap(base, i, n);
        }
        for (auto i = n; i > 1u; i--) {
          swap(base, base + i - 1u);
          down_heap(base, 1u, i - 1u);
        }
        continue;
      }
      auto hi = n - 1u;
      auto middle = hi >> 1u;
      swap_greater(base, base + middle);
      swap_greater(base, base + hi);
      swap_greater(base + middle, base + hi);
      auto pivot = point(base + middle);
      swap(base + middle, base + hi - 1u);
      auto left = 0u;
      auto right = hi - 1u;
      while (true) {
        left++;
        while (less(point(base + left), pivot)) {
          left++;
        }
        right--;
        while (less(pivot, point(base + right))) {
          right--;
        }
        if (left >= right) {
          break;
        }
        swap(base + left, base + right);
      }
      if (left != hi - 1u) {
        swap(base + left, base + hi - 1u);
      }
      stack[top] = uint3(base, left, depth - 1u);
      top++;
      stack[top] = uint3(base + left + 1u, n - left - 1u, depth - 1u);
      top++;
    }
  }
  void run() {
    auto count = 0u;
    for (auto i = 0u; i < p(0u); i++) {
      if (boxes[i * 16u + 9u] == 0u) {
        continue;
      }
      for (auto k = 0u; k < 16u; k++) {
        output[(count + 1u) * 16u + k] = boxes[i * 16u + k];
      }
      count++;
    }
    canonical_sort(count);
    for (auto i = 1u; i < count; i++) {
      auto j = i;
      while (j > 0u) {
        auto a = point(j - 1u);
        auto b = point(j);
        if (abs(b.y - a.y) >= 10.0 || b.x >= a.x) {
          break;
        }
        swap(j, j - 1u);
        j--;
      }
    }
    output[0] = count;
  }
};
kernel void db_sort(device const uint *boxes [[buffer(0)]],
                    device uint *output [[buffer(1)]],
                    constant Params &params [[buffer(2)]]) {

  Resident state{boxes, output, params};
  state.run();
}
