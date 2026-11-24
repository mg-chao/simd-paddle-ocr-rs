// Native Metal buffer and parameter contract for db_sort.
struct Resident {
  device const uint *boxes;
  device uint *output;
  constant Params &params;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  // Break equal-coordinate ties by the reverse raster contour index.
  uint slot(uint i) { return (i + 1u) * 16u + 15u; }
  float coord(uint id, uint axis) {
    return as_type<float>(boxes[id * 16u + axis]);
  }
  bool before(uint a, uint b, bool row) {
    if (row && coord(a, 0u) != coord(b, 0u))
      return coord(a, 0u) < coord(b, 0u);
    return coord(a, 1u) != coord(b, 1u) ? coord(a, 1u) < coord(b, 1u) : a < b;
  }
  void sift(uint start, uint n, uint i, bool row) {
    uint value = output[slot(start + i)];
    while (2u * i + 1u < n) {
      uint child = 2u * i + 1u;
      if (child + 1u < n && before(output[slot(start + child)],
                                   output[slot(start + child + 1u)], row))
        ++child;
      uint next = output[slot(start + child)];
      if (!before(value, next, row))
        break;
      output[slot(start + i)] = next;
      i = child;
    }
    output[slot(start + i)] = value;
  }
  void sort_indices(uint start, uint n, bool row) {
    for (uint k = n / 2u; k > 0u; --k)
      sift(start, n, k - 1u, row);
    for (uint k = n; k > 1u; --k) {
      uint last = slot(start + k - 1u), first = slot(start);
      uint v = output[first];
      output[first] = output[last];
      output[last] = v;
      sift(start, k - 1u, 0u, row);
    }
  }
  void run() {
    uint n = 0u;
    for (uint i = 0u; i < p(0u); ++i)
      if (boxes[i * 16u + 9u] != 0u)
        output[slot(n++)] = i;
    sort_indices(0u, n, false);
    for (uint start = 0u; start < n;) {
      uint end = start + 1u;
      while (end < n &&
             coord(output[slot(end)], 1u) - coord(output[slot(end - 1u)], 1u) <
                 10.)
        ++end;
      sort_indices(start, end - start, true);
      start = end;
    }
    for (uint i = 0u; i < n; ++i) {
      uint id = output[slot(i)];
      for (uint k = 0u; k < 16u; ++k)
        output[(i + 1u) * 16u + k] = boxes[id * 16u + k];
    }
    output[0] = n;
  }
};
kernel void db_sort(device const uint *boxes [[buffer(0)]],
                    device uint *output [[buffer(1)]],
                    constant Params &params [[buffer(2)]]) {

  Resident state{boxes, output, params};
  state.run();
}
