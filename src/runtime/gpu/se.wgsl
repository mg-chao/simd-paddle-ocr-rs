// Last excitation projection and gate application: one workgroup per N,C.
var<workgroup> partial:array<f32,64>;
var<workgroup> gate:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let row=group.x+group.y*p(1u);if row>=p(0u){return;}
    let inner=p(3u);let channels=p(4u);let plane=p(5u);let channel=row%channels;let batch=row/channels;
    var sum=0.0;for(var i=lane;i<inner;i+=64u){sum+=load_a(batch*inner+i)*load_b(channel*inner+i);}
    partial[lane]=sum;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){if lane<stride{partial[lane]+=partial[lane+stride];}workgroupBarrier();}
    if lane==0u{var value=partial[0];if p(7u)!=0u{value+=load_c(channel);}
        if p(8u)==1u {gate=clamp(value*bitcast<f32>(p(9u))+bitcast<f32>(p(10u)),0.0,1.0);}
        else {gate=1.0/(1.0+exp(-value));}}
    workgroupBarrier();
    for(var i=lane;i<plane;i+=64u){store_output(row*plane+i,load_d(row*plane+i)*gate);}
}
