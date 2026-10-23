@group(0) @binding(0) var<storage,read> pixels:array<u32>;
@group(0) @binding(1) var<storage,read> transforms:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
fn t(i:u32)->u32{return transforms[p(5u)*32u+i];}
fn coefficient(i:u32)->vec2<f32>{return vec2<f32>(bitcast<f32>(t(8u+i*2u)),bitcast<f32>(t(9u+i*2u)));}
fn pixel(x:i32,y:i32,c:u32)->f32{
    let px=u32(clamp(x,0,i32(p(0u))-1));let py=u32(clamp(y,0,i32(p(1u))-1));let channel=select(c,2u-c,p(4u)!=0u);
    let offset=py*p(2u)+px*p(3u)+channel;return f32((pixels[offset/4u]>>((offset%4u)*8u))&255u);
}
fn cubic(value:vec2<f32>)->vec2<f32>{
    var x=value;if dvalue(x)<0.0{x=-x;}
    if dvalue(x)<=1.0{return da(dm(da(dm(ds(1.25),x),ds(-2.25)),dm(x,x)),ds(1.0));}
    if dvalue(x)<2.0{return da(dm(da(dm(da(dm(ds(-0.75),x),ds(3.75)),x),ds(-6.0)),x),ds(3.0));}
    return ds(0.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>){
    let index=id.x+id.y*grid.x*64u;let width=t(0u);let height=t(1u);if index>=width*height{return;}
    let ox=index%width;let oy=index/width;let vertical=t(4u)!=0u;
    let x=select(ox,t(2u)-1u-oy,vertical);let y=select(oy,ox,vertical);
    let u=dd(ds(f32(x)),ds(f32(t(2u))));let v=dd(ds(f32(y)),ds(f32(t(3u))));
    let den=da(da(dm(coefficient(6u),u),dm(coefficient(7u),v)),ds(1.0));
    let sx=dd(da(da(dm(coefficient(0u),u),dm(coefficient(1u),v)),coefficient(2u)),den);
    let sy=dd(da(da(dm(coefficient(3u),u),dm(coefficient(4u),v)),coefficient(5u)),den);
    let xb=i32(dfloor(sx));let yb=i32(dfloor(sy));var result=0xff000000u;
    var wx:array<vec2<f32>,4>;var wy:array<vec2<f32>,4>;
    for(var k=0;k<4;k++){wx[k]=cubic(da(sx,ds(-f32(xb+k-1))));wy[k]=cubic(da(sy,ds(-f32(yb+k-1))));}
    for(var c=0u;c<3u;c++){
        var sum=ds(0.0);
        for(var ky=0;ky<4;ky++){for(var kx=0;kx<4;kx++){sum=da(sum,dm(dm(ds(pixel(xb+kx-1,yb+ky-1,c)),wx[kx]),wy[ky]));}}
        result|=u32(clamp(dfloor(da(sum,ds(0.5))),0.0,255.0))<<(c*8u);
    }output[index]=result;
}
