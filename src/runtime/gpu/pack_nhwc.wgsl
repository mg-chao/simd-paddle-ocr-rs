// Append to common.wgsl. Binding0 is canonical NCHW F32; binding1 is an
// optional F32 [batch, channel] scale, or an unused source/dummy binding.
// Binding5 uses the selected $O storage type. All arithmetic, including an
// optional channel scale, occurs in F32 before store_output converts once.
// Packet: 0=physical element count, 1=dispatch X thread count, 3=input Cin,
// 4=H*W, 5=physical Cin, 6=scale enabled. Host validates nonzero dimensions.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) position: vec3<u32>) {
    let index = position.x + position.y * p(1u);
    if index >= p(0u) {
        return;
    }
    let input_channels = p(3u);
    let plane = p(4u);
    let physical_channels = p(5u);
    let channel = index % physical_channels;
    if channel >= input_channels {
        store_output(index, 0.0);
        return;
    }
    let pixel = (index / physical_channels) % plane;
    let batch = index / (physical_channels * plane);
    var value = load_a((batch * input_channels + channel) * plane + pixel);
    if p(6u) != 0u {
        value = value * load_b(batch * input_channels + channel);
    }
    store_output(index, value);
}
