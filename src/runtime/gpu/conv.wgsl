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
    if p(19u) != 0u { sum = load_c(co); }
    for (var ci = 0u; ci < cpg; ci++) {
        for (var ky = 0u; ky < kh; ky++) {
            let iy = i32(oy * sh + ky * dh) - i32(pt);
            if iy < 0 || iy >= i32(height) { continue; }
            for (var kx = 0u; kx < kw; kx++) {
                let ix = i32(ox * sw + kx * dw) - i32(pl);
                if ix < 0 || ix >= i32(width) { continue; }
                let source = ((batch * cin + first + ci) * height + u32(iy)) * width + u32(ix);
                let weight = ((co * cpg + ci) * kh + ky) * kw + kx;
                sum = sum + load_a(source) * load_b(weight);
            }
        }
    }
    store_output(index, epilogue(sum, index));
}
