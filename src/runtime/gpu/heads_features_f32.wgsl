// FP16 feature values widen exactly before the existing FP32 vocabulary head.
// Params and enable-f16 are supplied by the resident dispatch source prefix.
@group(0) @binding(0) var<storage, read> features: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) position: vec3<u32>) {
    let index = position.x + position.y * params.words[0].y;
    if index < params.words[0].x {
        output[index] = f32(features[index]);
    }
}
