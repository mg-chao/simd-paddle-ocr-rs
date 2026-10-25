// Native Metal buffer and parameter contract for heads_features_f32.
struct Resident {
  device const half *features;
  device float *output;
  constant Params &params;

  void run(uint3 position) {
    auto index = position.x + position.y * params.words[0].y;
    if (index < params.words[0].x) {
      output[index] = float(features[index]);
    }
  }
};
kernel void heads_features_f32(device const half *features [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               constant Params &params [[buffer(2)]],
                               uint3 position [[thread_position_in_grid]]) {

  Resident state{features, output, params};
  state.run(position);
}
