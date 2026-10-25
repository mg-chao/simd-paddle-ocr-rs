// Four SIMD groups cooperate on a 64-channel by 64-position tile. The
// descriptor explicitly forbids reduced-precision arithmetic for F32 operands.
void run(uint3 id, uint3 group, uint lane, threadgroup float *) const {
  uint rows = p(4), inner = p(3), columns = p(7) * p(8);
  uint block = group.x + group.y * p(34), tiles = p(32) * p(33);
  uint batch = block / tiles;
  if (batch >= p(35))
    return;
  uint row0 = ((block / p(33)) % p(32)) * 64, column0 = (block % p(33)) * 64;
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      64, 64, int(dynamic_extent), PACKED, PACKED, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> multiply;
  auto compute = [&](auto weights, auto values) {
    auto result = multiply.template get_destination_cooperative_tensor<
        decltype(weights), decltype(values), float>();
    for (uint i = 0; i < result.get_capacity(); ++i)
      if (result.is_valid_element(i)) {
        uint row = row0 + uint(result.get_multidimensional_index(i)[1]);
        result[i] = p(19) && row < rows ? load_c(row) : 0.0f;
      }
    multiply.run(weights, values, result);
    for (uint i = 0; i < result.get_capacity(); ++i)
      if (result.is_valid_element(i)) {
        auto xy = result.get_multidimensional_index(i);
        uint row = row0 + uint(xy[1]), column = column0 + uint(xy[0]);
        if (row >= rows || column >= columns)
          continue;
        uint index = TOKEN_OUTPUT ? (batch * columns + column) * rows + row
                                  : (batch * rows + row) * columns + column;
        float value = result[i];
#if TOKEN_OUTPUT
        // The closed tail graph is Swish followed by an NTC residual add.
        value = value * sigmoid(value) + load_d(index);
#else
        value = epilogue(value, index);
#endif
        store_output(index, value);
      }
  };
  using Weights = tensor<device STORAGE_B, dextents<int, 2>, tensor_inline>;
  using Values = tensor<device STORAGE_A, dextents<int, 2>, tensor_inline>;
#if PACKED
  Weights weights(const_cast<device STORAGE_B *>(b),
                  dextents<int, 2>(int(rows), int(inner)),
                  array<int, 2>{1, int(p(36))});
  Values values(const_cast<device STORAGE_A *>(a) + batch * columns * inner,
                dextents<int, 2>(int(inner), int(columns)),
                array<int, 2>{1, int(inner)});
  compute(weights.slice(int(row0), 0), values.slice(0, int(column0)));
#else
  Weights weights(const_cast<device STORAGE_B *>(b),
                  dextents<int, 2>(int(inner), int(rows)),
                  array<int, 2>{1, int(inner)});
  Values values(const_cast<device STORAGE_A *>(a) + batch * inner * columns,
                dextents<int, 2>(int(columns), int(inner)),
                array<int, 2>{1, int(columns)});
  compute(weights.slice(0, int(row0)), values.slice(int(column0), 0));
#endif
}
