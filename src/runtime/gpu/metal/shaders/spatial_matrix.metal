// Compile-time dimensions describe the complete source image; signed offsets
// and tensor extents supply zero padding without out-of-range device loads.
#if STAGED
#define PATCH_WIDTH (7 * STRIDE_W + FILTER)
#define PATCH_HEIGHT (3 * STRIDE_H + FILTER)
#define SCRATCH_FLOATS ((INPUT_CHANNELS * PATCH_WIDTH * PATCH_HEIGHT + 1) / 2)
#endif
void run(uint3, uint3 group, uint lane, threadgroup float *scratch) const {
  uint block = group.x + group.y * p(34), batch = block / (p(32) * p(33));
  if (batch >= p(35))
    return;
  uint channel0 = ((block / p(33)) % p(32)) * 32;
  uint tile = block % p(33), x0 = (tile % ((p(8) + 7) / 8)) * 8,
       y0 = (tile / ((p(8) + 7) / 8)) * 4;
  constexpr auto descriptor = mpp::tensor_ops::convolution2d_descriptor(
      int4(32, 8, 4, 1),
#if STAGED
      int4(INPUT_CHANNELS, PATCH_WIDTH, PATCH_HEIGHT, 1),
#else
      int4(INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT, 1),
#endif
      int2(FILTER, FILTER),
      mpp::tensor_ops::convolution2d_activation_layout::nhwc,
      mpp::tensor_ops::convolution2d_weights_layout::hwio,
      int2(STRIDE_W, STRIDE_H), int2(1), 1, false,
      mpp::tensor_ops::convolution2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::convolution2d<descriptor, execution_simdgroups<4>> operation;
#if STAGED
  threadgroup half *patch = reinterpret_cast<threadgroup half *>(scratch);
  for (uint logical = lane;
       logical < INPUT_CHANNELS * PATCH_WIDTH * PATCH_HEIGHT; logical += 128) {
    uint channel = logical / (PATCH_WIDTH * PATCH_HEIGHT),
         pixel = logical % (PATCH_WIDTH * PATCH_HEIGHT);
    int y = int(y0 * STRIDE_H + pixel / PATCH_WIDTH) - int(p(17));
    int x = int(x0 * STRIDE_W + pixel % PATCH_WIDTH) - int(p(18));
    patch[pixel * INPUT_CHANNELS + channel] =
        (x >= 0 && y >= 0 && x < int(p(6)) && y < int(p(5)))
            ? half(load_a(((batch * p(3) + channel) * p(5) + uint(y)) * p(6) +
                          uint(x)))
            : half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  operation.set_offsets(int2(FILTER / 2));
  using Image = tensor<threadgroup half, dextents<int, 4>, tensor_inline>;
  Image images(patch,
               dextents<int, 4>(INPUT_CHANNELS, PATCH_WIDTH, PATCH_HEIGHT, 1),
               array<int, 4>{1, INPUT_CHANNELS, INPUT_CHANNELS * PATCH_WIDTH,
                             INPUT_CHANNELS * PATCH_WIDTH * PATCH_HEIGHT});
  auto image = images.slice(0, 0, 0, 0);
#else
  operation.set_offsets(int2(int(x0 * STRIDE_W) - int(p(18)) + FILTER / 2,
                             int(y0 * STRIDE_H) - int(p(17)) + FILTER / 2));
  using Image = tensor<device STORAGE_A, dextents<int, 4>, tensor_inline>;
  Image images(
      const_cast<device STORAGE_A *>(a),
      dextents<int, 4>(int(p(3)), int(p(6)), int(p(5)), int(p(35))),
      array<int, 4>{1, int(p(3)), int(p(3) * p(6)), int(p(3) * p(5) * p(6))});
  auto image = images.slice(0, 0, 0, int(batch));
#endif
  using Filter = tensor<device STORAGE_B, dextents<int, 4>, tensor_inline>;
  Filter weights(const_cast<device STORAGE_B *>(b),
                 dextents<int, 4>(int(p(36)), int(p(3)), FILTER, FILTER),
                 array<int, 4>{1, int(p(36)), int(p(36) * p(3)),
                               int(FILTER * p(36) * p(3))});
  auto filter = weights.slice(int(channel0), 0, 0, 0);
  auto result = operation.template get_destination_cooperative_tensor<
      decltype(image), decltype(filter), float>();
  for (uint i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i)) {
      uint channel = channel0 + uint(result.get_multidimensional_index(i)[0]);
      result[i] = p(19) && channel < p(4) ? load_c(channel) : 0.0f;
    }
  operation.run(image, filter, result);
  for (uint i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i)) {
      auto xyz = result.get_multidimensional_index(i);
      uint channel = channel0 + uint(xyz[0]), x = x0 + uint(xyz[1]),
           y = y0 + uint(xyz[2]);
      if (channel < p(4) && x < p(8) && y < p(7)) {
        uint index = ((batch * p(4) + channel) * p(7) + y) * p(8) + x;
        store_output(index, epilogue(result[i], index));
      }
    }
}
