$ENABLE
// Portable F32 correctness kernels. Each dispatch uses checked, packed U32 parameters.
struct Parameters { words: array<vec4<u32>, 32>, }
@group(0) @binding(0) var<storage, read> raw_a: array<$A>;
@group(0) @binding(1) var<storage, read> raw_b: array<$B>;
@group(0) @binding(2) var<storage, read> raw_c: array<$C>;
@group(0) @binding(3) var<storage, read> raw_d: array<$D>;
@group(0) @binding(4) var<storage, read> raw_e: array<$E>;
@group(0) @binding(5) var<storage, read_write> raw_output: array<$O>;
@group(0) @binding(6) var<uniform> params: Parameters;
fn p(i:u32)->u32 { return params.words[i/4u][i%4u]; }
fn erf_approx(x:f32)->f32 {
    if abs(x)>=6.0 {return select(1.0,-1.0,x<0.0);}
    let t=1.0/(1.0+0.3275911*abs(x));
    let poly=(((((1.0614054*t-1.4531521)*t)+1.4214138)*t-0.28449672)*t+0.2548296)*t;
    return select(1.0,-1.0,x<0.0)*(1.0-poly*exp(-x*x));
}

$ROUND_F16

fn epilogue(initial: f32, index: u32) -> f32 {
    var value = initial;
    for (var i=0u; i<p(64u); i++) {
        let base=68u+i*3u; let op=p(base);
        if op<=2u {
            var offset=index;
            if p(base+1u)==0u { offset=0u; }
            if p(base+1u)==1u { offset=(index/p(66u))%p(65u); }
            if op==1u { value=value+load_d(offset); } else { value=value+load_e(offset); }
        } else if op==3u { value=select(0.0,value,value>0.0); }
        else if op==4u { if value>=0.0 {value=1.0/(1.0+exp(-value));} else {let x=exp(value);value=x/(1.0+x);} }
        else if op==5u { value=clamp(value*bitcast<f32>(p(base+1u))+bitcast<f32>(p(base+2u)),0.0,1.0); }
        else if op==6u { value=value*clamp(value*0.16666667+0.5,0.0,1.0); }
        else if op==7u { value=((erf_approx(value/1.4142135623730951)+1.0)*value)*0.5; }
        else if op==8u { var gate=0.0;if value>=0.0 {gate=1.0/(1.0+exp(-value));} else {let x=exp(value);gate=x/(1.0+x);}value=value*gate; }
    }
    return value;
}

fn load_a(i:u32)->f32 {return f32(raw_a[i]);}
fn load_b(i:u32)->f32 {return f32(raw_b[i]);}
fn load_c(i:u32)->f32 {return f32(raw_c[i]);}
fn load_d(i:u32)->f32 {return f32(raw_d[i]);}
fn load_e(i:u32)->f32 {return f32(raw_e[i]);}
fn load_output(i:u32)->f32 {return f32(raw_output[i+p(60u)]);}
fn store_output(i:u32,value:f32) {raw_output[i+p(60u)]=$O(value);}
