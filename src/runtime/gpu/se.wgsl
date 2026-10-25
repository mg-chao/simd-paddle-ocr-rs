// Last excitation projection and gate application: one workgroup per N,C.
var<workgroup> partial:array<f32,64>;
var<workgroup> gate:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let row=group.x+group.y*p(1u);if row>=p(0u){return;}
    let inner=p(3u);let channels=p(4u);let plane=p(5u);let channel=row%channels;let batch=row/channels;
    if p(12u)==1u {
        // The existing physical Conv epilogue evaluates bias and activation
        // in F32, then stores only the gate. Preserve that boundary; p13=0
        // also retains serial dot order without materializing the gate.
        var value=0.0;
        if p(13u)!=0u {
            // Explicit fast-matrix opt-in: all64 lanes contribute Float
            // products; no Half conversion is permitted inside this dot.
            var sum=0.0;
            for(var i=lane;i<inner;i+=64u){sum+=load_a(batch*inner+i)*load_b(channel*inner+i);}
            partial[lane]=sum;workgroupBarrier();
            for(var stride=32u;stride>1u;stride/=2u){
                if lane<stride{partial[lane]+=partial[lane+stride];}
                workgroupBarrier();
            }
            // The stride2 barrier also publishes partial[1] to lane0.
            // Finish the exact same tree without an eighth group barrier.
            if lane==0u{value=partial[0]+partial[1];}
        } else if lane==0u {
            for(var i=0u;i<inner;i++){value+=load_a(batch*inner+i)*load_b(channel*inner+i);}
        }
        if lane==0u {
            value+=load_c(select(channel,0u,p(11u)!=0u));
            if p(8u)==1u {gate=output_round(clamp(value*bitcast<f32>(p(9u))+bitcast<f32>(p(10u)),0.0,1.0));}
            else {
                var activated=0.0;
                if value>=0.0 {activated=1.0/(1.0+exp(-value));}
                else {let exponent=exp(value);activated=exponent/(1.0+exponent);}
                gate=output_round(activated);
            }
        }
        workgroupBarrier();
        for(var i=lane;i<plane;i+=64u){store_output(row*plane+i,load_d(row*plane+i)*gate);}
        return;
    }
    var sum=0.0;for(var i=lane;i<inner;i+=64u){sum+=load_a(batch*inner+i)*load_b(channel*inner+i);}
    partial[lane]=sum;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){if lane<stride{partial[lane]+=partial[lane+stride];}workgroupBarrier();}
    if lane==0u{var value=partial[0];if p(7u)!=0u{value+=load_c(channel);}
        if p(8u)==1u {gate=clamp(value*bitcast<f32>(p(9u))+bitcast<f32>(p(10u)),0.0,1.0);}
        else {gate=1.0/(1.0+exp(-value));}}
    workgroupBarrier();
    for(var i=lane;i<plane;i+=64u){store_output(row*plane+i,load_d(row*plane+i)*gate);}
}
