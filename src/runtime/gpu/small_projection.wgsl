// One ordinary64-thread workgroup owns one canonical [N,Cout,1,1] output.
// Half storage does not narrow products, partial sums, bias or the epilogue.
var<workgroup> partial:array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=group.x+group.y*p(1u);if row>=p(0u){return;}
    let inner=p(3u);let channels=p(4u);let batch=row/channels;let channel=row%channels;
    var sum=0.0;
    // Preserve the original Conv's bias initialization before the dot. Only
    // lane0 owns it, so reduction includes it exactly once without a store.
    if lane==0u && p(19u)!=0u {sum=load_c(channel);}
    for(var k=lane;k<inner;k+=64u){sum+=load_a(batch*inner+k)*load_b(channel*inner+k);}
    partial[lane]=sum;workgroupBarrier();
    for(var stride=32u;stride>1u;stride/=2u){
        if lane<stride{partial[lane]+=partial[lane+stride];}
        workgroupBarrier();
    }
    if lane==0u{store_output(row,epilogue(partial[0]+partial[1],row));}
}
