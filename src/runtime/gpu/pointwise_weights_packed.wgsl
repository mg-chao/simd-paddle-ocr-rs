// K-major vectors hold four adjacent output channels; the last vector is padded.
fn pointwise_weights(row:u32, k:u32, rows:u32, inner:u32)->vec4<f32> {
    if k<inner && row<rows {return raw_b[k*((rows+3u)/4u)+row/4u];}
    return vec4<f32>(0.0);
}
