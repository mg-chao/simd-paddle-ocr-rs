void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  uint index = id.x + id.y * p(1);
  if (index >= p(0))
    return;
  uint cin = p(3), cout = p(4), h = p(5), w = p(6), oh = p(7), ow = p(8),
       cpg = p(9), opg = p(10);
  uint ox = index % ow, oy = (index / ow) % oh, co = (index / (oh * ow)) % cout,
       batch = index / (oh * ow * cout);
  float sum = p(19) ? load_c(co) : 0.0f;
  for (uint ci = 0; ci < cpg; ci++)
    for (uint ky = 0; ky < p(11); ky++) {
      int y = int(oy * p(13) + ky * p(15)) - int(p(17));
      if (y < 0 || y >= int(h))
        continue;
      for (uint kx = 0; kx < p(12); kx++) {
        int x = int(ox * p(14) + kx * p(16)) - int(p(18));
        if (x < 0 || x >= int(w))
          continue;
        uint source =
            ((batch * cin + (co / opg) * cpg + ci) * h + uint(y)) * w + uint(x);
        sum += load_a(source) *
               load_b(((co * cpg + ci) * p(11) + ky) * p(12) + kx);
      }
    }
  store_output(index, epilogue(sum, index));
}
