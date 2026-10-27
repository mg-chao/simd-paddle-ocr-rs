// Native Metal buffer and parameter contract for db_rows.
struct Resident {
  device const uint *parents;
  device const uint *ranks;
  device const uint *metadata;
  device atomic_uint *rows;
  constant Params &params;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  void run(uint3 id, uint3 grid) {
    auto i = id.x + id.y * grid.x * 64u;
    if (p(2u) == 0u) {
      if (i < metadata[3]) {
        atomic_store_explicit(&rows[i * 2u], p(0u), memory_order_relaxed);
        atomic_store_explicit(&rows[i * 2u + 1u], 0u, memory_order_relaxed);
      }
      return;
    }
    if (i >= p(0u) * p(1u)) {
      return;
    }
    auto rank = ranks[parents[i]];
    if (rank == 0xffffffffu) {
      return;
    }
    auto base = 16u + rank * 8u;
    auto row = metadata[base + 5u] + i / p(0u) - metadata[base + 2u];
    atomic_fetch_min_explicit(&rows[row * 2u], i % p(0u), memory_order_relaxed);
    atomic_fetch_max_explicit(&rows[row * 2u + 1u], i % p(0u),
                              memory_order_relaxed);
  }
};
kernel void db_rows(device const uint *parents [[buffer(0)]],
                    device const uint *ranks [[buffer(1)]],
                    device const uint *metadata [[buffer(2)]],
                    device atomic_uint *rows [[buffer(3)]],
                    constant Params &params [[buffer(4)]],
                    uint3 id [[thread_position_in_grid]],
                    uint3 grid [[threadgroups_per_grid]]) {

  Resident state{parents, ranks, metadata, rows, params};
  state.run(id, grid);
}
