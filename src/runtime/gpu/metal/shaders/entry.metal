}
;
#ifndef SCRATCH_FLOATS
#define SCRATCH_FLOATS 1
#endif
kernel void neural(device const STORAGE_A *a [[buffer(0)]],
                   device const STORAGE_B *b [[buffer(1)]],
                   device const STORAGE_C *c [[buffer(2)]],
                   device const STORAGE_D *d [[buffer(3)]],
                   device const STORAGE_E *e [[buffer(4)]],
                   device STORAGE_O *output [[buffer(5)]],
                   constant NeuralParameters &params [[buffer(6)]],
                   uint3 id [[thread_position_in_grid]],
                   uint3 group [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[SCRATCH_FLOATS];
  Neural n{a, b, c, d, e, output, params};
  n.run(id, group, lane, scratch);
}
