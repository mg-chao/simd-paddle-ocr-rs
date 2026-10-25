@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x+id.y*p(1u);
    if index>=p(0u) { return; }
    let op=p(2u);
    if op==26u {
        if index>=p(6u){return;}
        let h=p(3u);let w=p(4u);let row=index/p(5u);let x=(index%p(5u))*4u;
        let y=row%h;let base=row*w+x;
        if x+4u<w {
            let current=vec4<f32>(load_a(base),load_a(base+1u),load_a(base+2u),load_a(base+3u));
            var pooled=max(current,vec4<f32>(current.y,current.z,current.w,load_a(base+4u)));
            if y+1u<h {
                let next=vec4<f32>(load_a(base+w),load_a(base+w+1u),load_a(base+w+2u),load_a(base+w+3u));
                pooled=max(pooled,next);
                pooled=max(pooled,vec4<f32>(next.y,next.z,next.w,load_a(base+w+4u)));
            }
            for(var component=0u;component<4u;component++){store_output(base+component,pooled[component]);}
        } else {
            // Keep the original max sequence at every right edge, including
            // NaNs and signed zero; no duplicate sentinel or out-of-row read.
            for(var component=0u;component<4u;component++){
                let column=x+component;if column>=w{break;}
                let element=base+component;var value=load_a(element);
                if column+1u<w {value=max(value,load_a(element+1u));}
                if y+1u<h {value=max(value,load_a(element+w));
                    if column+1u<w {value=max(value,load_a(element+w+1u));}}
                store_output(element,value);
            }
        }
        return;
    }
    if op==25u {
        // Each original intermediate was stored with output precision. Keep
        // those conversions while eliminating its allocation and dispatch.
        if index>=p(0u)/4u+select(0u,1u,p(0u)%4u!=0u){return;}
        for(var component=0u;component<4u;component++) {
            let element=index*4u+component;if element>=p(0u){break;}
            let value=load_a(element);var result=0.0;
            if p(3u)==1u {
                let scaled=output_round(value/load_b(0u));
                let error=output_round(erf_approx(scaled));
                let shifted=output_round(error+1.0);
                let product=output_round(shifted*value);
                result=product*0.5;
            } else if p(3u)==2u {
                let gate=output_round(clamp(value*0.16666667+0.5,0.0,1.0));
                result=value*gate;
            } else {
                var gate=0.0;
                if value>=0.0 {gate=1.0/(1.0+exp(-value));}
                else {let exponent=exp(value);gate=exponent/(1.0+exponent);}
                result=value*output_round(gate);
            }
            store_output(element,result);
        }
        return;
    }
    if op==24u {
        let channels=p(3u);let plane=p(4u);let padded=p(5u);
        let channel=index%padded;let pixel=(index/padded)%plane;let batch=index/(padded*plane);
        var value=0.0;if channel<channels{value=load_a((batch*channels+channel)*plane+pixel);}
        store_output(index,value);return;
    }
    if op==22u {store_output(index+p(4u),load_a(index+p(3u)));return;}
    if op==23u {
        let h=p(3u);let w=p(4u);let factor=p(5u);let oh=h*factor;let ow=w*factor;
        let x=index%ow;let y=(index/ow)%oh;let plane=index/(oh*ow)*h*w;
        store_output(index,load_a(plane+(y/factor)*w+x/factor)+load_b(index));return;
    }
    if op<=3u {
        var remaining=index; var ai=0u; var bi=0u; var axis=p(3u);
        loop { if axis==0u { break; } axis--;
            let coordinate=remaining%p(4u+axis); remaining/=p(4u+axis);
            ai+=coordinate*p(12u+axis); bi+=coordinate*p(20u+axis);
        }
        if op==0u { store_output(index,load_a(ai)+load_b(bi)); }
        else if op==1u { store_output(index,load_a(ai)*load_b(bi)); }
        else if op==2u { store_output(index,load_a(ai)/load_b(bi)); }
        else { store_output(index,load_a(ai)-load_b(bi)); }
        return;
    }
    if op==4u { store_output(index,load_a(index)*load_a(index)); return; }
    if op==5u { store_output(index,sqrt(load_a(index))); return; }
    if op==6u { store_output(index,select(0.0,load_a(index),load_a(index)>0.0)); return; }
    if op==7u { store_output(index,erf_approx(load_a(index))); return; }
    if op==8u {
        let value=load_a(index);
        if value>=0.0 {store_output(index,1.0/(1.0+exp(-value)));}
        else {let exponent=exp(value);store_output(index,exponent/(1.0+exponent));}
        return;
    }
    if op==9u { store_output(index,clamp(load_a(index)*bitcast<f32>(p(3u))+bitcast<f32>(p(4u)),0.0,1.0)); return; }
    if op==11u {
        let cin=p(3u); let cout=p(4u); let h=p(5u); let w=p(6u);
        let ow=w*2u; let oh=h*2u;
        let x=index%ow; let y=(index/ow)%oh; let co=(index/(ow*oh))%cout;
        let batch=index/(ow*oh*cout); var sum=0.0;
        if p(7u)!=0u { sum=load_c(co); }
        for(var ci=0u;ci<cin;ci++) {
            let source=((batch*cin+ci)*h+y/2u)*w+x/2u;
            let weight=(ci*cout+co)*4u+(y%2u)*2u+x%2u;
            sum+=load_a(source)*load_b(weight);
        }
        store_output(index,epilogue(sum,index)); return;
    }
    if op==12u {
        let h=p(3u);let w=p(4u);let x=index%w;let y=(index/w)%h;
        let plane=index/(h*w)*h*w; var value=load_a(index);
        if x+1u<w {value=max(value,load_a(index+1u));}
        if y+1u<h {value=max(value,load_a(plane+(y+1u)*w+x));
            if x+1u<w {value=max(value,load_a(plane+(y+1u)*w+x+1u));}}
        store_output(index,value);return;
    }
    if op==13u {
        let h=p(3u);let w=p(4u);let oh=p(5u);let ow=p(6u);
        let x=index%ow;let y=(index/ow)%oh;let plane=index/(oh*ow)*h*w;
        var sum=0.0;
        for(var ky=0u;ky<3u;ky++){for(var kx=0u;kx<2u;kx++) {sum+=load_a(plane+(y*3u+ky)*w+x*2u+kx);}}
        store_output(index,sum/6.0);return;
    }
    if op==14u {
        let channel=(index/p(4u))%p(3u);
        let multiplier=load_b(channel)/sqrt(load_e(channel)+bitcast<f32>(p(5u)));
        store_output(index,(load_a(index)-load_d(channel))*multiplier+load_c(channel));return;
    }
    if op==15u {
        var remaining=index;var source=p(4u);var axis=p(3u);
        loop{if axis==0u{break;}axis--;let coordinate=remaining%p(8u+axis);remaining/=p(8u+axis);source+=coordinate*p(16u+axis);}
        store_output(index,load_a(source));return;
    }
    if op==17u {
        let h=p(3u);let w=p(4u);let factor=p(5u);let oh=h*factor;let ow=w*factor;
        let x=index%ow;let y=(index/ow)%oh;let plane=index/(oh*ow)*h*w;
        store_output(index,load_a(plane+(y/factor)*w+x/factor));return;
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
            sum+=load_a(offset);}
        store_output(index,sum/f32(reduction));return;
    }
    if op==20u {
        let rows=p(3u);let inner=p(4u);let columns=p(5u);let matrix=rows*columns;
        let row=(index/columns)%rows;let column=index%columns;var batch=index/matrix;
        var ai=0u;var bi=0u;var axis=p(6u);
        loop{if axis==0u{break;}axis--;let coordinate=batch%p(8u+axis);batch/=p(8u+axis);ai+=coordinate*p(16u+axis);bi+=coordinate*p(24u+axis);}
        var sum=0.0;for(var k=0u;k<inner;k++){sum+=load_a(ai+row*inner+k)*load_b(bi+k*columns+column);}
        store_output(index,epilogue(sum,index));return;
    }
    if op==21u {
        let columns=p(3u);let start=index*columns;var maximum=load_a(start);
        for(var col=1u;col<columns;col++){maximum=max(maximum,load_a(start+col));}
        var sum=0.0;for(var col=0u;col<columns;col++){let value=exp(load_a(start+col)-maximum);store_output(start+col,value);sum+=value;}
        for(var col=0u;col<columns;col++){store_output(start+col,load_output(start+col)/sum);}return;
    }
}
