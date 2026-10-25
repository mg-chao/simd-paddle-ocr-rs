// Unevaluated sums provide extra precision without native shader-f64.
// The runtime supplies an all-ones mask. Keeping rounding observable through
// integer operations prevents legal WGSL reassociation from erasing error terms.
fn rf(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)&params.words[63].w);}
fn af(a:f32,b:f32)->f32{return rf(a+b);}
fn sf(a:f32,b:f32)->f32{return rf(a-b);}
fn mf(a:f32,b:f32)->f32{return rf(a*b);}
fn ds(x:f32)->vec2<f32>{return vec2<f32>(x,0.0);}
fn da(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
    let s=af(a.x,b.x);let v=sf(s,a.x);
    let e=af(af(af(sf(a.x,sf(s,v)),sf(b.x,v)),a.y),b.y);let hi=af(s,e);
    return vec2<f32>(hi,sf(e,sf(hi,s)));
}
fn dn(a:vec2<f32>)->vec2<f32>{return -a;}
fn dm(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
    let hi=mf(a.x,b.x);
    let ah=bitcast<f32>(bitcast<u32>(a.x)&0xfffff000u);let al=sf(a.x,ah);
    let bh=bitcast<f32>(bitcast<u32>(b.x)&0xfffff000u);let bl=sf(b.x,bh);
    let error=af(af(af(sf(mf(ah,bh),hi),mf(ah,bl)),mf(al,bh)),mf(al,bl));
    let low=af(af(error,mf(a.x,b.y)),mf(a.y,b.x));let s=af(hi,low);
    return vec2<f32>(s,sf(low,sf(s,hi)));
}
fn dd(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{let q=rf(a.x/b.x);let remainder=da(a,dn(dm(b,ds(q))));return da(ds(q),ds((remainder.x+remainder.y)/b.x));}
fn dq(a:vec2<f32>)->vec2<f32>{let q=rf(sqrt(a.x));if q==0.0{return ds(0.0);}return da(ds(q),dd(da(a,dn(dm(ds(q),ds(q)))),ds(2.0*q)));}
// Subtract the integral high part before adding the low part. Reassociation
// would round away a negative low term at a boundary and move a crop by a pixel.
fn dfloor(a:vec2<f32>)->f32{let base=floor(a.x);let tail=af(sf(a.x,base),a.y);return af(base,floor(tail));}
fn dvalue(a:vec2<f32>)->f32{return a.x+a.y;}
fn dround(a:vec2<f32>)->f32{let lo=dfloor(a);let tail=dvalue(da(a,ds(-lo)));if tail>0.5 || (tail==0.5 && (i32(lo)&1)!=0){return lo+1.0;}return lo;}
fn dlength(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{let x=da(ds(a.x),ds(-b.x));let y=da(ds(a.y),ds(-b.y));return dq(da(dm(x,x),dm(y,y)));}
fn dpi()->vec2<f32>{return vec2<f32>(3.1415927410125732,-8.742278000372485e-8);}
fn datan(value:vec2<f32>)->vec2<f32>{return ds(rf(atan(dvalue(value))));}
fn datan2(y:vec2<f32>,x:vec2<f32>)->vec2<f32>{if dvalue(x)==0.0{return dm(dpi(),ds(select(-0.5,0.5,dvalue(y)>=0.0)));}return ds(rf(atan2(dvalue(y),dvalue(x))));}
fn dsincos(angle:vec2<f32>)->vec4<f32>{let x=dvalue(angle);return vec4<f32>(ds(rf(sin(x))),ds(rf(cos(x))));}
fn dacos(x:vec2<f32>)->vec2<f32>{return ds(rf(acos(dvalue(x))));}
fn daway(x:vec2<f32>)->f32{if dvalue(x)<0.0{return -dfloor(da(-x,ds(0.5)));}return dfloor(da(x,ds(0.5)));}
