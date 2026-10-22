// Direct F32 NCHW convolution. OIHW weights include only channels per group.
// Parameter words are packed vec4s to satisfy uniform-buffer alignment.
struct Parameters { words: array<vec4<u32>, 8>, }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Parameters;
fn p(i: u32) -> u32 { return params.words[i / 4u][i % 4u]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x + id.y * p(1u);
    if index >= p(0u) { return; }
    let cin = p(3u); let cout = p(4u);
    let height = p(5u); let width = p(6u);
    let oh = p(7u); let ow = p(8u);
    let cpg = p(9u); let opg = p(10u);
    let kh = p(11u); let kw = p(12u);
    let sh = p(13u); let sw = p(14u);
    let dh = p(15u); let dw = p(16u);
    let pt = p(17u); let pl = p(18u);
    let ox = index % ow;
    let oy = (index / ow) % oh;
    let co = (index / (ow * oh)) % cout;
    let batch = index / (ow * oh * cout);
    let first = (co / opg) * cpg;
    var sum = 0.0;
    if p(19u) != 0u { sum = bias[co]; }
    for (var ci = 0u; ci < cpg; ci++) {
        for (var ky = 0u; ky < kh; ky++) {
            let iy = i32(oy * sh + ky * dh) - i32(pt);
            if iy < 0 || iy >= i32(height) { continue; }
            for (var kx = 0u; kx < kw; kx++) {
                let ix = i32(ox * sw + kx * dw) - i32(pl);
                if ix < 0 || ix >= i32(width) { continue; }
                let source = ((batch * cin + first + ci) * height + u32(iy)) * width + u32(ix);
                let weight = ((co * cpg + ci) * kh + ky) * kw + kx;
                sum = sum + input[source] * weights[weight];
            }
        }
    }
    output[index] = sum;
}
