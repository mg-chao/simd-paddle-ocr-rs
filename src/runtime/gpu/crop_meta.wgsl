@group(0) @binding(0) var<storage,read> boxes:array<u32>;
@group(0) @binding(1) var<storage,read_write> transforms:array<u32>;
@group(0) @binding(2) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
fn store(base:u32,i:u32,value:vec2<f32>){transforms[base+8u+i*2u]=bitcast<u32>(value.x);transforms[base+9u+i*2u]=bitcast<u32>(value.y);}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    let index=id.x;if index>=p(0u){return;}
    // Full OCR can prepare the accepted prefix before its count reaches CPU.
    // Standalone callers supply their own count and do not depend on the header.
    if p(1u)!=0u && index>=boxes[0]{return;}
    let base=index*32u;let input=(index+1u)*16u;
    var q:array<vec2<f32>,4>;for(var i=0u;i<4u;i++){q[i]=vec2<f32>(bitcast<f32>(boxes[input+i*2u]),bitcast<f32>(boxes[input+i*2u+1u]));}
    let width=u32(max(dfloor(dlength(q[0],q[1])),dfloor(dlength(q[2],q[3]))));
    let height=u32(max(dfloor(dlength(q[0],q[3])),dfloor(dlength(q[1],q[2]))));
    let vertical=height*2u>=width*3u;
    transforms[base]=select(width,height,vertical);transforms[base+1u]=select(height,width,vertical);
    transforms[base+2u]=width;transforms[base+3u]=height;transforms[base+4u]=u32(vertical);transforms[base+5u]=0u;
    if width==0u || height==0u{transforms[base+5u]=1u;return;}
    let dx1=da(ds(q[1].x),ds(-q[2].x));let dx2=da(ds(q[3].x),ds(-q[2].x));let dx3=da(da(ds(q[0].x),ds(-q[1].x)),da(ds(q[2].x),ds(-q[3].x)));
    let dy1=da(ds(q[1].y),ds(-q[2].y));let dy2=da(ds(q[3].y),ds(-q[2].y));let dy3=da(da(ds(q[0].y),ds(-q[1].y)),da(ds(q[2].y),ds(-q[3].y)));
    var g=ds(0.0);var h=ds(0.0);
    if abs(dvalue(dx3))>1e-12 || abs(dvalue(dy3))>1e-12{
        let denominator=da(dm(dx1,dy2),dn(dm(dx2,dy1)));if abs(dvalue(denominator))<=1e-12{transforms[base+5u]=1u;return;}
        g=dd(da(dm(dx3,dy2),dn(dm(dx2,dy3))),denominator);h=dd(da(dm(dx1,dy3),dn(dm(dx3,dy1))),denominator);
    }
    let a=da(da(ds(q[1].x),ds(-q[0].x)),dm(g,ds(q[1].x)));let b=da(da(ds(q[3].x),ds(-q[0].x)),dm(h,ds(q[3].x)));
    let d=da(da(ds(q[1].y),ds(-q[0].y)),dm(g,ds(q[1].y)));let e=da(da(ds(q[3].y),ds(-q[0].y)),dm(h,ds(q[3].y)));
    store(base,0u,a);store(base,1u,b);store(base,2u,ds(q[0].x));store(base,3u,d);store(base,4u,e);store(base,5u,ds(q[0].y));store(base,6u,g);store(base,7u,h);
    let gu=dm(g,dd(ds(f32(width-1u)),ds(f32(width))));let hv=dm(h,dd(ds(f32(height-1u)),ds(f32(height))));
    if dvalue(da(gu,ds(1.0)))<=1e-12 || dvalue(da(hv,ds(1.0)))<=1e-12 || dvalue(da(da(gu,hv),ds(1.0)))<=1e-12{transforms[base+5u]=1u;}
}
