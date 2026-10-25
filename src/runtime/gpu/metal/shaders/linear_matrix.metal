void run(uint3, uint3 group, uint, threadgroup float *) const {
  uint block = group.x + group.y * p(1), tiles = (p(5) + 63) / 64;
  uint row0 = (block / tiles) * 32, column0 = (block % tiles) * 64;
  if (row0 >= p(3) || column0 >= p(5))
    return;
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      32, 64, int(dynamic_extent), false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> multiply;
  using Left = tensor<device STORAGE_A, dextents<int, 2>, tensor_inline>;
  using Right = tensor<device STORAGE_B, dextents<int, 2>, tensor_inline>;
  Left left(const_cast<device STORAGE_A *>(a),
            dextents<int, 2>(int(p(4)), int(p(3))),
            array<int, 2>{1, int(p(4))});
  Right right(const_cast<device STORAGE_B *>(b),
              dextents<int, 2>(int(p(5)), int(p(4))),
              array<int, 2>{1, int(p(5))});
  auto x = left.slice(0, int(row0)), w = right.slice(int(column0), 0);
  auto result = multiply.template get_destination_cooperative_tensor<
      decltype(x), decltype(w), float>();
  for (uint i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i))
      result[i] = 0;
  multiply.run(x, w, result);
  for (uint i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i)) {
      auto xy = result.get_multidimensional_index(i);
      uint row = row0 + uint(xy[1]), column = column0 + uint(xy[0]);
      if (row >= p(3) || column >= p(5))
        continue;
      uint index = row * p(5) + column;
      float value = output_round(result[i]);
      if (p(19))
        value = output_round(value + load_c(column));
      if (p(21))
        value = output_round(value + load_d(index));
      if (p(20))
        value *= output_round(sigmoid(value));
      store_output(index, value);
    }
}
