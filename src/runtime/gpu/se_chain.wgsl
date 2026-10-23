// Global pooling is a preceding dispatch. No cross-workgroup synchronization.
var<workgroup> hidden:array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let batch=group.x+group.y*p(1u);if batch>=p(0u){return;}
 let cin=p(3u);let middle=p(4u);let cout=p(5u);
 if lane<middle {var sum=load_c(lane);for(var k=0u;k<cin;k++){sum+=load_a(batch*cin+k)*load_b(lane*cin+k);}hidden[lane]=f32($O(max(0.0,sum)));}
 workgroupBarrier();
 for(var c=lane;c<cout;c+=64u){var sum=load_e(c);for(var k=0u;k<middle;k++){sum+=hidden[k]*load_d(c*middle+k);}store_output(batch*cout+c,clamp(sum*bitcast<f32>(p(9u))+bitcast<f32>(p(10u)),0.0,1.0));}
}
