fn pointwise_weights(row:u32, k:u32, rows:u32, inner:u32)->vec4<f32> {
    var value=vec4<f32>(0.0);
    if k<inner {
        if row<rows {value.x=load_b(row*inner+k);}
        if row+1u<rows {value.y=load_b((row+1u)*inner+k);}
        if row+2u<rows {value.z=load_b((row+2u)*inner+k);}
        if row+3u<rows {value.w=load_b((row+3u)*inner+k);}
    }
    return value;
}
