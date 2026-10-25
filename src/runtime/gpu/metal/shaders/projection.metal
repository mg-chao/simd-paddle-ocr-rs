#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>
// Vocabulary projection keeps logits in threadgroup memory and emits one
// winner/normalization partial per tile, avoiding a full logits allocation.
kernel void projection(device const MATRIX_STORAGE *features [[buffer(0)]],
                       device const MATRIX_STORAGE *weights [[buffer(1)]],
                       device const float *bias [[buffer(2)]],
                       device uint4 *partial [[buffer(3)]],
                       constant Params &params [[buffer(4)]],
                       uint2 group [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  uint rows = params.words[0].x, inner = params.words[0].y,
       classes = params.words[0].z, tiles = params.words[0].w;
  uint row0 = group.y * 32, column0 = group.x * 64;
  if (row0 >= rows || column0 >= classes)
    return;
  constexpr float limit = 3.402823e38f;
  threadgroup float scores[2048];
  for (uint i = tid; i < 2048; i += 128)
    scores[i] = -limit;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      32, 64, int(dynamic_extent), false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> multiply;
  using Matrix = tensor<device MATRIX_STORAGE, dextents<int, 2>, tensor_inline>;
  Matrix x(const_cast<device MATRIX_STORAGE *>(features),
           dextents<int, 2>(int(inner), int(rows)),
           array<int, 2>{1, int(inner)});
  Matrix w(const_cast<device MATRIX_STORAGE *>(weights),
           dextents<int, 2>(int(classes), int(inner)),
           array<int, 2>{1, int(classes)});
  auto left = x.slice(0, int(row0)), right = w.slice(int(column0), 0);
  auto result = multiply.template get_destination_cooperative_tensor<
      decltype(left), decltype(right), float>();
  for (uint i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i))
      result[i] = 0;
  multiply.run(left, right, result);
  for (uint i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i)) {
      auto xy = result.get_multidimensional_index(i);
      uint row = uint(xy[1]), column = uint(xy[0]);
      if (row0 + row < rows && column0 + column < classes)
        scores[row * 64 + column] = result[i] + bias[column0 + column];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = sg; row < 32; row += 4) {
    float lo = scores[row * 64 + lane], hi = scores[row * 64 + lane + 32];
    bool first = column0 + lane < classes,
         second = column0 + lane + 32 < classes;
    float maximum = simd_max(max(first ? lo : -limit, second ? hi : -limit));
    uint winner = simd_min(
        min(first && lo == maximum ? column0 + lane : 0xffffffffu,
            second && hi == maximum ? column0 + lane + 32 : 0xffffffffu));
    uint invalid = simd_or(uint(first && (as_type<uint>(lo) & 0x7fffffffu) >
                                             as_type<uint>(limit)) |
                           uint(second && (as_type<uint>(hi) & 0x7fffffffu) >
                                              as_type<uint>(limit)));
    float sum = 0;
    if (params.words[1].x) {
      sum = simd_sum((first ? exp(lo - maximum) : 0.0f) +
                     (second ? exp(hi - maximum) : 0.0f));
    }
    if (lane == 0 && row0 + row < rows)
      partial[(row0 + row) * tiles + group.x] =
          uint4(winner, as_type<uint>(maximum), as_type<uint>(sum), invalid);
  }
}
