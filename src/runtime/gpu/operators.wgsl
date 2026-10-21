// Portable F32 correctness kernels. Each dispatch uses checked, packed U32 parameters.
struct Parameters { words: array<vec4<u32>, 32>, }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read> d: array<f32>;
@group(0) @binding(4) var<storage, read> e: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Parameters;
fn p(i:u32)->u32 { return params.words[i/4u][i%4u]; }
fn erf_approx(x:f32)->f32 {
    if abs(x)>=6.0 {return select(1.0,-1.0,x<0.0);}
    let t=1.0/(1.0+0.3275911*abs(x));
    let poly=(((((1.0614054*t-1.4531521)*t)+1.4214138)*t-0.28449672)*t+0.2548296)*t;
    return select(1.0,-1.0,x<0.0)*(1.0-poly*exp(-x*x));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x+id.y*p(1u);
    if index>=p(0u) { return; }
    let op=p(2u);
    if op<=3u {
        var remaining=index; var ai=0u; var bi=0u; var axis=p(3u);
        loop { if axis==0u { break; } axis--;
            let coordinate=remaining%p(4u+axis); remaining/=p(4u+axis);
            ai+=coordinate*p(12u+axis); bi+=coordinate*p(20u+axis);
        }
        if op==0u { output[index]=a[ai]+b[bi]; }
        else if op==1u { output[index]=a[ai]*b[bi]; }
        else if op==2u { output[index]=a[ai]/b[bi]; }
        else { output[index]=a[ai]-b[bi]; }
        return;
    }
    if op==4u { output[index]=a[index]*a[index]; return; }
    if op==5u { output[index]=sqrt(a[index]); return; }
    if op==6u { output[index]=select(0.0,a[index],a[index]>0.0); return; }
    if op==7u { output[index]=erf_approx(a[index]); return; }
    if op==8u {
        let value=a[index];
        if value>=0.0 {output[index]=1.0/(1.0+exp(-value));}
        else {let exponent=exp(value);output[index]=exponent/(1.0+exponent);}
        return;
    }
    if op==9u { output[index]=clamp(a[index]*bitcast<f32>(p(3u))+bitcast<f32>(p(4u)),0.0,1.0); return; }
    if op==11u {
        let cin=p(3u); let cout=p(4u); let h=p(5u); let w=p(6u);
        let ow=w*2u; let oh=h*2u;
        let x=index%ow; let y=(index/ow)%oh; let co=(index/(ow*oh))%cout;
        let batch=index/(ow*oh*cout); var sum=0.0;
        if p(7u)!=0u { sum=c[co]; }
        for(var ci=0u;ci<cin;ci++) {
            let source=((batch*cin+ci)*h+y/2u)*w+x/2u;
            let weight=(ci*cout+co)*4u+(y%2u)*2u+x%2u;
            sum+=a[source]*b[weight];
        }
        output[index]=sum; return;
    }
    if op==12u {
        let h=p(3u);let w=p(4u);let x=index%w;let y=(index/w)%h;
        let plane=index/(h*w)*h*w; var value=a[index];
        if x+1u<w {value=max(value,a[index+1u]);}
        if y+1u<h {value=max(value,a[plane+(y+1u)*w+x]);
            if x+1u<w {value=max(value,a[plane+(y+1u)*w+x+1u]);}}
        output[index]=value;return;
    }
    if op==13u {
        let h=p(3u);let w=p(4u);let oh=p(5u);let ow=p(6u);
        let x=index%ow;let y=(index/ow)%oh;let plane=index/(oh*ow)*h*w;
        var sum=0.0;
        for(var ky=0u;ky<3u;ky++){for(var kx=0u;kx<2u;kx++) {sum+=a[plane+(y*3u+ky)*w+x*2u+kx];}}
        output[index]=sum/6.0;return;
    }
    if op==14u {
        let channel=(index/p(4u))%p(3u);
        let multiplier=b[channel]/sqrt(e[channel]+bitcast<f32>(p(5u)));
        output[index]=(a[index]-d[channel])*multiplier+c[channel];return;
    }
    if op==15u {
        var remaining=index;var source=p(4u);var axis=p(3u);
        loop{if axis==0u{break;}axis--;let coordinate=remaining%p(8u+axis);remaining/=p(8u+axis);source+=coordinate*p(16u+axis);}
        output[index]=a[source];return;
    }
    if op==17u {
        let h=p(3u);let w=p(4u);let factor=p(5u);let oh=h*factor;let ow=w*factor;
        let x=index%ow;let y=(index/ow)%oh;let plane=index/(oh*ow)*h*w;
        output[index]=a[plane+(y/factor)*w+x/factor];return;
    }
    if op==18u {
        let rank=p(3u);let reduction=p(4u);var base=0u;var input_stride=1u;
        var axis=rank;
        loop{if axis==0u{break;}axis--;let dimension=p(8u+axis);let stride=p(16u+axis);
            if stride!=0u {base+=((index/stride)%dimension)*input_stride;}input_stride*=dimension;}
        var sum=0.0;
        for(var r=0u;r<reduction;r++) {var remaining=r;var offset=base;var stride=1u;var ax=rank;
            loop{if ax==0u{break;}ax--;let dimension=p(8u+ax);
                if p(16u+ax)==0u {offset+=(remaining%dimension)*stride;remaining/=dimension;}stride*=dimension;}
            sum+=a[offset];}
        output[index]=sum/f32(reduction);return;
    }
    if op==20u {
        let rows=p(3u);let inner=p(4u);let columns=p(5u);let matrix=rows*columns;
        let row=(index/columns)%rows;let column=index%columns;var batch=index/matrix;
        var ai=0u;var bi=0u;var axis=p(6u);
        loop{if axis==0u{break;}axis--;let coordinate=batch%p(8u+axis);batch/=p(8u+axis);ai+=coordinate*p(16u+axis);bi+=coordinate*p(24u+axis);}
        var sum=0.0;for(var k=0u;k<inner;k++){sum+=a[ai+row*inner+k]*b[bi+k*columns+column];}
        output[index]=sum;return;
    }
    if op==21u {
        let columns=p(3u);let start=index*columns;var maximum=a[start];
        for(var col=1u;col<columns;col++){maximum=max(maximum,a[start+col]);}
        var sum=0.0;for(var col=0u;col<columns;col++){let value=exp(a[start+col]-maximum);output[start+col]=value;sum+=value;}
        for(var col=0u;col<columns;col++){output[start+col]=output[start+col]/sum;}return;
    }
}
