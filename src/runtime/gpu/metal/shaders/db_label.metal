// Native Metal buffer and parameter contract for db_label.
struct Resident {
  device const INPUT *prediction;
  device uint *mask;
  device atomic_uint *parents;
  device atomic_uint *counts;
  device uint *ranks;
  device uint *blocks;
  device atomic_uint *metadata;
  constant Params &params;
  threadgroup atomic_uint *tile_parents;
  threadgroup uint *scan_fg;
  threadgroup uint *scan_bg;

  uint p(uint i) { return params.words[i / 4u][i % 4u]; }
  uint root(uint value) {
    auto v = value;
    while (true) {
      auto next = atomic_load_explicit(&parents[v], memory_order_relaxed);
      if (next == v) {
        break;
      }
      v = next;
    }
    return v;
  }
  void join(uint a, uint b) {
    auto left = root(a);
    auto right = root(b);
    while (true) {
      if (left == right) {
        break;
      }
      auto hi = max(left, right);
      auto lo = min(left, right);
      auto previous =
          atomic_fetch_min_explicit(&parents[hi], lo, memory_order_relaxed);
      if (previous == hi) {
        break;
      }
      left = root(previous);
      right = root(lo);
    }
  }
  bool is_foreground(uint i) {
    return float(prediction[i]) > as_type<float>(p(2u));
  }
  uint qualifier(uint i) {
    if (atomic_load_explicit(&parents[i], memory_order_relaxed) != i) {
      return 0u;
    }
    auto count = atomic_load_explicit(&counts[i], memory_order_relaxed);
    if (count < 3u || (count & 0x80000000u) != 0u) {
      return 0u;
    }
    return select(2u, 1u, mask[i] != 0u);
  }

  uint tile_root(uint i) {
    auto r = i;
    while (true) {
      auto next = atomic_load_explicit(&tile_parents[r], memory_order_relaxed);
      if (next == r) {
        break;
      }
      r = next;
    }
    return r;
  }
  void tile_join(uint a, uint b) {
    auto x = tile_root(a);
    auto y = tile_root(b);
    while (true) {
      if (x == y) {
        break;
      }
      auto hi = max(x, y);
      auto lo = min(x, y);
      auto previous = atomic_fetch_min_explicit(&tile_parents[hi], lo,
                                                memory_order_relaxed);
      if (previous == hi) {
        break;
      }
      x = tile_root(previous);
      y = tile_root(lo);
    }
  }

  void run(uint lane, uint3 group, uint3 grid) {
    auto block = group.x + group.y * grid.x;
    auto i = block * 64u + lane;
    auto w = p(0u);
    auto h = p(1u);
    auto n = w * h;
    auto mode = p(4u);
    if (mode == 1u) {
      auto columns = (w + 7u) / 8u;
      auto bx = (block % columns) * 8u;
      auto by = (block / columns) * 8u;
      atomic_store_explicit(&tile_parents[lane], lane, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto x = bx + lane % 8u;
      auto y = by + lane / 8u;
      if (x < w && y < h) {
        auto pixel = y * w + x;
        auto fg = mask[pixel];
        if (lane % 8u < 7u && x + 1u < w && mask[pixel + 1u] == fg) {
          tile_join(lane, lane + 1u);
        }
        if (lane / 8u < 7u && y + 1u < h) {
          if (mask[pixel + w] == fg) {
            tile_join(lane, lane + 8u);
          }
          if (fg != 0u) {
            if (lane % 8u > 0u && mask[pixel + w - 1u] != 0u) {
              tile_join(lane, lane + 7u);
            }
            if (lane % 8u < 7u && x + 1u < w && mask[pixel + w + 1u] != 0u) {
              tile_join(lane, lane + 9u);
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (x < w && y < h) {
        auto r = tile_root(lane);
        atomic_store_explicit(&parents[y * w + x],
                              (by + r / 8u) * w + bx + r % 8u,
                              memory_order_relaxed);
      }
      return;
    }
    if (mode == 4u) {
      if (block >= (n + 63u) / 64u) {
        return;
      }
      auto kind = 0u;
      if (i < n) {
        kind = qualifier(i);
      }
      scan_fg[lane] = uint(kind == 1u);
      scan_bg[lane] = uint(kind == 2u);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (auto d = 1u; d < 64u; d *= 2u) {
        auto a = 0u;
        auto b = 0u;
        if (lane >= d) {
          a = scan_fg[lane - d];
          b = scan_bg[lane - d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan_fg[lane] += a;
        scan_bg[lane] += b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (i < n) {
        ranks[i] = select(scan_bg[lane], scan_fg[lane], kind == 1u);
      }
      if (lane == 63u) {
        blocks[block * 4u] = scan_fg[63];
        blocks[block * 4u + 1u] = scan_bg[63];
      }
      return;
    }

    if (mode == 9u) {
      auto block_count = (n + 63u) / 64u;
      if (block >= (block_count + 63u) / 64u) {
        return;
      }
      auto fg = 0u;
      auto bg = 0u;
      if (i < block_count) {
        fg = blocks[i * 4u];
        bg = blocks[i * 4u + 1u];
      }
      scan_fg[lane] = fg;
      scan_bg[lane] = bg;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (auto d = 1u; d < 64u; d *= 2u) {
        auto a = 0u;
        auto b = 0u;
        if (lane >= d) {
          a = scan_fg[lane - d];
          b = scan_bg[lane - d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan_fg[lane] += a;
        scan_bg[lane] += b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (i < block_count) {
        blocks[i * 4u + 2u] = scan_fg[lane] - fg;
        blocks[i * 4u + 3u] = scan_bg[lane] - bg;
      }
      if (lane == 63u) {
        auto base = (block_count + block) * 4u;
        blocks[base] = scan_fg[63];
        blocks[base + 1u] = scan_bg[63];
      }
      return;
    }
    if (mode == 5u) {
      if (i != 0u) {
        return;
      }
      auto fg = 0u;
      auto bg = 0u;
      auto block_count = (n + 63u) / 64u;
      for (auto b = 0u; b < (block_count + 63u) / 64u; b++) {
        auto base = (block_count + b) * 4u;
        blocks[base + 2u] = fg;
        blocks[base + 3u] = bg;
        fg += blocks[base];
        bg += blocks[base + 1u];
      }
      atomic_store_explicit(&metadata[0], min(fg, p(3u)), memory_order_relaxed);
      atomic_store_explicit(&metadata[1], min(fg + bg, p(3u)),
                            memory_order_relaxed);
      for (auto c = 0u; c < min(fg + bg, p(3u)); c++) {
        auto base = 16u + c * 8u;
        atomic_store_explicit(&metadata[base + 1u], w, memory_order_relaxed);
        atomic_store_explicit(&metadata[base + 2u], h, memory_order_relaxed);
      }
      return;
    }
    if (mode == 8u) {
      if (i != 0u) {
        return;
      }
      auto rows = 0u;
      for (auto c = 0u;
           c < atomic_load_explicit(&metadata[1], memory_order_relaxed); c++) {
        auto base = 16u + c * 8u;
        atomic_store_explicit(&metadata[base + 5u], rows, memory_order_relaxed);
        rows +=
            atomic_load_explicit(&metadata[base + 4u], memory_order_relaxed) -
            atomic_load_explicit(&metadata[base + 2u], memory_order_relaxed) +
            1u;
      }
      atomic_store_explicit(&metadata[3], rows, memory_order_relaxed);
      return;
    }
    if (i >= n) {
      return;
    }
    auto x = i % w;
    auto y = i / w;
    if (mode == 0u) {
      auto value = float(prediction[i]);
      if (!(value <= 3.402823e38 && value >= -3.402823e38)) {
        atomic_fetch_or_explicit(&metadata[2], 1u, memory_order_relaxed);
      }
      auto yes = is_foreground(i);
      if (p(5u) != 0u) {
        if (x + 1u < w) {
          yes = yes || is_foreground(i + 1u);
        }
        if (y + 1u < h) {
          yes = yes || is_foreground(i + w);
          if (x + 1u < w) {
            yes = yes || is_foreground(i + w + 1u);
          }
        }
      }
      mask[i] = uint(yes);
      atomic_store_explicit(&parents[i], i, memory_order_relaxed);
      atomic_store_explicit(&counts[i], 0u, memory_order_relaxed);
      return;
    }
    if (mode == 2u) {
      auto fg = mask[i];
      if ((p(6u) == 0u || x % 8u == 7u) && x + 1u < w && mask[i + 1u] == fg) {
        join(i, i + 1u);
      }
      if (y + 1u < h) {
        if ((p(6u) == 0u || y % 8u == 7u) && mask[i + w] == fg) {
          join(i, i + w);
        }
        if (fg != 0u) {
          if ((p(6u) == 0u || y % 8u == 7u || x % 8u == 0u) && x > 0u &&
              mask[i + w - 1u] != 0u) {
            join(i, i + w - 1u);
          }
          if ((p(6u) == 0u || y % 8u == 7u || x % 8u == 7u) && x + 1u < w &&
              mask[i + w + 1u] != 0u) {
            join(i, i + w + 1u);
          }
        }
      }
      return;
    }
    if (mode == 3u) {
      auto r = root(i);
      atomic_store_explicit(&parents[i], r, memory_order_relaxed);
      auto boundary = 0u;
      if (mask[i] != 0u) {
        auto edge = x == 0u || y == 0u || x + 1u == w || y + 1u == h;
        for (auto dy = -1; dy <= 1; dy++) {
          for (auto dx = -1; dx <= 1; dx++) {
            auto nx = int(x) + dx;
            auto ny = int(y) + dy;
            if (nx >= 0 && ny >= 0 && nx < int(w) && ny < int(h)) {
              edge = edge || mask[uint(ny) * w + uint(nx)] == 0u;
            }
          }
        }
        boundary = uint(edge);
      } else {
        if (x == 0u || y == 0u || x + 1u == w || y + 1u == h) {
          atomic_fetch_or_explicit(&counts[r], 0x80000000u,
                                   memory_order_relaxed);
        }
        if (x > 0u) {
          boundary += mask[i - 1u];
        }
        if (x + 1u < w) {
          boundary += mask[i + 1u];
        }
        if (y > 0u) {
          boundary += mask[i - w];
        }
        if (y + 1u < h) {
          boundary += mask[i + w];
        }
      }
      if (boundary != 0u) {
        atomic_fetch_add_explicit(&counts[r], boundary, memory_order_relaxed);
      }
      return;
    }
    if (mode == 6u) {
      auto kind = qualifier(i);
      if (kind == 0u) {
        ranks[i] = 0xffffffffu;
        return;
      }
      auto upper = ((n + 63u) / 64u + block / 64u) * 4u;
      auto offset =
          select(blocks[block * 4u + 3u] + blocks[upper + 3u] +
                     atomic_load_explicit(&metadata[0], memory_order_relaxed),
                 blocks[block * 4u + 2u] + blocks[upper + 2u], kind == 1u);
      auto rank = ranks[i] + offset - 1u;
      ranks[i] = select(0xffffffffu, rank, rank < p(3u));
      if (rank < p(3u)) {
        auto base = 16u + rank * 8u;
        atomic_store_explicit(&metadata[base], i, memory_order_relaxed);
        atomic_store_explicit(&metadata[base + 6u], kind, memory_order_relaxed);
      }
      return;
    }
    if (mode == 7u) {
      auto rank =
          ranks[atomic_load_explicit(&parents[i], memory_order_relaxed)];
      if (rank == 0xffffffffu) {
        return;
      }
      auto base = 16u + rank * 8u;
      atomic_fetch_min_explicit(&metadata[base + 1u], x, memory_order_relaxed);
      atomic_fetch_min_explicit(&metadata[base + 2u], y, memory_order_relaxed);
      atomic_fetch_max_explicit(&metadata[base + 3u], x, memory_order_relaxed);
      atomic_fetch_max_explicit(&metadata[base + 4u], y, memory_order_relaxed);
    }
  }
};
kernel void db_label(device const INPUT *prediction [[buffer(0)]],
                     device uint *mask [[buffer(1)]],
                     device atomic_uint *parents [[buffer(2)]],
                     device atomic_uint *counts [[buffer(3)]],
                     device uint *ranks [[buffer(4)]],
                     device uint *blocks [[buffer(5)]],
                     device atomic_uint *metadata [[buffer(6)]],
                     constant Params &params [[buffer(7)]],
                     uint lane [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]],
                     uint3 grid [[threadgroups_per_grid]]) {
  threadgroup atomic_uint tile_parents[64];
  threadgroup uint scan_fg[64];
  threadgroup uint scan_bg[64];
  Resident state{prediction, mask,   parents,      counts,  ranks,  blocks,
                 metadata,   params, tile_parents, scan_fg, scan_bg};
  state.run(lane, group, grid);
}
