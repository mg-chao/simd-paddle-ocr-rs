@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<$OUT>;
@group(0) @binding(3) var<uniform> params: Params;
fn p(i:u32)->u32 { return params.words[i/4u][i%4u]; }
fn pixel(x:u32,y:u32,c:u32)->i32 {
    if x>=p(0u) || y>=p(1u) { return 0; }
    let channel=select(c,2u-c,p(4u)!=0u);
    let byte=y*p(2u)+x*p(3u)+channel;
    return i32((pixels[byte/4u]>>((byte%4u)*8u))&255u);
}
// Integer rational construction avoids precision-dependent half-coordinate rounding.
fn multiply_divide(a:u32,b:u32,d:u32)->vec2<u32>{
    if a==0u{return vec2<u32>(0u);}
    if b<=0xffffffffu/a{let product=a*b;return vec2<u32>(product/d,product%d);}
    let a0=a&65535u;let a1=a>>16u;let b0=b&65535u;let b1=b>>16u;
    let w0=a0*b0;let t=a1*b0+(w0>>16u);let w1=(t&65535u)+a0*b1;
    let high=a1*b1+(t>>16u)+(w1>>16u);let low=(w1<<16u)+(w0&65535u);
    var rem=high;var quotient=0u;
    for(var bit=32u;bit>0u;bit--){rem=(rem<<1u)|((low>>(bit-1u))&1u);if rem>=d{rem-=d;quotient|=1u<<(bit-1u);}}
    return vec2<u32>(quotient,rem);
}
fn coefficient(position:u32,source:u32,destination:u32)->vec2<i32> {
    let denominator=2u*destination;
    let numerator=multiply_divide(2u*position+1u,source,denominator);
    let base=i32(numerator.x)-select(1,0,numerator.y>=destination);
    let rem=(numerator.y+denominator-destination)%denominator;
    let scaled=multiply_divide(rem,2048u,denominator);
    let whole=scaled.x;
    let rest=scaled.y;
    let up=rest*2u>denominator || (rest*2u==denominator && (whole&1u)!=0u);
    return vec2<i32>(base,i32(whole+u32(up)));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>) {
    let pixel_index=id.x+id.y*grid.x*64u;
    let plane=p(8u)*p(9u);
    if pixel_index>=plane { return; }
    let y=pixel_index/p(9u); let x=pixel_index%p(9u);
    if x>=p(7u) {
        for(var c=0u;c<3u;c++){output[c*plane+pixel_index+p(12u)]=normalize_store(bitcast<f32>(p(11u)));}
        return;
    }
    var xc=coefficient(x,p(5u),p(7u));
    if xc.x<0 { xc=vec2<i32>(0,0); }
    if xc.x>=i32(p(5u))-1 { xc=vec2<i32>(i32(p(5u))-1,0); }
    let yc=coefficient(y,p(6u),p(8u));
    let x0=u32(xc.x); let x1=min(x0+1u,p(5u)-1u);
    let y0=u32(clamp(yc.x,0,i32(p(6u))-1)); let y1=u32(clamp(yc.x+1,0,i32(p(6u))-1));
    let a=2048-xc.y; let b=xc.y;
    for(var c=0u;c<3u;c++){
        let h0=pixel(x0,y0,c)*a+pixel(x1,y0,c)*b;
        let h1=pixel(x0,y1,c)*a+pixel(x1,y1,c)*b;
        let value=clamp((((h0>>4)*(2048-yc.y)>>16)+((h1>>4)*yc.y>>16)+2)>>2,0,255);
        output[c*plane+pixel_index+p(12u)]=normalize_store(lut[c*256u+u32(value)]);
    }
}
