const SOFTMAX:bool=$SOFTMAX;
const FINISH:bool=$FINISH;
var<workgroup> values:array<f32,64>;
var<workgroup> maximum:f32;
var<workgroup> denominator:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let flat=group.x+group.y*p(1u);let columns=p(3u);let chunks=p(4u);
    var row=flat;var chunk=0u;if !FINISH {row=flat/chunks;chunk=flat%chunks;}
    if row>=p(0u){return;}
    if FINISH {
        var value=0.0;if SOFTMAX {value=-3.402823466e38;}
        for(var part=lane;part<chunks;part+=64u){let x=load_b((row*chunks+part)*2u);if SOFTMAX {value=max(value,x);}else{value+=x;}}
        values[lane]=value;workgroupBarrier();
        for(var stride=32u;stride>0u;stride/=2u){if lane<stride {if SOFTMAX {values[lane]=max(values[lane],values[lane+stride]);}else{values[lane]+=values[lane+stride];}}workgroupBarrier();}
        if !SOFTMAX {if lane==0u {store_output(row,values[0]/f32(columns));}return;}
        if lane==0u {maximum=values[0];}workgroupBarrier();value=0.0;
        for(var part=lane;part<chunks;part+=64u){let offset=(row*chunks+part)*2u;value+=load_b(offset+1u)*exp(load_b(offset)-maximum);}
        values[lane]=value;workgroupBarrier();
        for(var stride=32u;stride>0u;stride/=2u){if lane<stride {values[lane]+=values[lane+stride];}workgroupBarrier();}
        if lane==0u {denominator=values[0];}workgroupBarrier();
        for(var col=lane;col<columns;col+=64u){store_output(row*columns+col,exp(load_a(row*columns+col)-maximum)/denominator);}
    } else {
        let end=min(columns,(chunk+1u)*1024u);let start=chunk*1024u;
        var value=0.0;if SOFTMAX {value=-3.402823466e38;}
        for(var col=start+lane;col<end;col+=64u){let x=load_a(row*columns+col);if SOFTMAX {value=max(value,x);}else{value+=x;}}
        values[lane]=value;workgroupBarrier();
        for(var stride=32u;stride>0u;stride/=2u){if lane<stride {if SOFTMAX {values[lane]=max(values[lane],values[lane+stride]);}else{values[lane]+=values[lane+stride];}}workgroupBarrier();}
        if lane==0u {maximum=values[0];store_output(flat*2u,maximum);}workgroupBarrier();
        if SOFTMAX {
            value=0.0;for(var col=start+lane;col<end;col+=64u){value+=exp(load_a(row*columns+col)-maximum);}
            values[lane]=value;workgroupBarrier();
            for(var stride=32u;stride>0u;stride/=2u){if lane<stride {values[lane]+=values[lane+stride];}workgroupBarrier();}
            if lane==0u {store_output(flat*2u+1u,values[0]);}
        }
    }
}
