// Each row is reduced by 64 lanes. No activation is transferred to the host.
const MODE:u32=$MODEu; // 0 mean, 1 stable softmax, 2 centered LayerNorm
var<workgroup> partial:array<f32,64>;
var<workgroup> mean:f32;
var<workgroup> scale:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=group.x+group.y*p(1u);if row>=p(0u) {return;}
    let columns=p(3u);let base=row*columns;
    var value=0.0;if MODE==1u {value=-3.402823466e38;}
    for(var col=lane;col<columns;col+=64u) {if MODE==1u {value=max(value,load_a(base+col));} else {value+=load_a(base+col);}}
    partial[lane]=value;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){if lane<stride {if MODE==1u {partial[lane]=max(partial[lane],partial[lane+stride]);}else{partial[lane]+=partial[lane+stride];}}workgroupBarrier();}
    if MODE==0u {if lane==0u {store_output(row,partial[0]/f32(columns));}return;}
    if lane==0u {mean=partial[0];if MODE==2u {mean/=f32(columns);}}workgroupBarrier();
    value=0.0;
    for(var col=lane;col<columns;col+=64u) {
        if MODE==1u {value+=exp(load_a(base+col)-mean);} else {let centered=load_a(base+col)-mean;value+=centered*centered;}
    }
    partial[lane]=value;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){if lane<stride {partial[lane]+=partial[lane+stride];}workgroupBarrier();}
    if lane==0u {scale=partial[0];if MODE==2u {scale=sqrt(scale/f32(columns)+bitcast<f32>(p(4u)));}}workgroupBarrier();
    for(var col=lane;col<columns;col+=64u) {
        if MODE==1u {store_output(base+col,exp(load_a(base+col)-mean)/scale);}
        else {store_output(base+col,((load_a(base+col)-mean)/scale)*load_b(col)+load_c(col));}
    }
}
