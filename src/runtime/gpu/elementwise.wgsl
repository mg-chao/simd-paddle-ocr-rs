// F32 arithmetic with semantic trailing-axis broadcasting, maximum rank eight.
struct Parameters { words: array<vec4<u32>, 8>, }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Parameters;
fn p(i: u32) -> u32 { return params.words[i / 4u][i % 4u]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * p(1u);
    if index >= p(0u) { return; }
    let op = p(2u);
    if op == 2u {
        // Match the CPU comparison (including canonical +0 for negative zero).
        output[index] = select(0.0, a[index], a[index] > 0.0);
        return;
    }
    var remaining = index;
    var ai = 0u; var bi = 0u;
    var axis = p(3u);
    loop {
        if axis == 0u { break; }
        axis--;
        let dimension = p(4u + axis);
        let coordinate = remaining % dimension;
        remaining = remaining / dimension;
        ai += coordinate * p(12u + axis);
        bi += coordinate * p(20u + axis);
    }
    if op == 1u { output[index] = a[ai] * b[bi]; }
    else { output[index] = a[ai] + b[bi]; }
}
