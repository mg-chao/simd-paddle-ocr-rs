// Optional subgroup variant. No assumption about subgroup width or count.
// wgpu 30.0.1 gates these builtins with Features::SUBGROUP; Naga does not
// accept the WGSL enable-subgroups directive in this pinned release.
const MODE:u32=$MODEu;
var<workgroup> partial:array<f32,64>;
var<workgroup> mean:f32;
var<workgroup> scale:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32,
        @builtin(subgroup_invocation_id) sublane:u32,@builtin(subgroup_id) subgroup:u32,
        @builtin(num_subgroups) groups:u32) {
    let row=group.x+group.y*p(1u);if row>=p(0u){return;}
    let columns=p(3u);let base=row*columns;
    var value=0.0;if MODE==1u {value=-3.402823466e38;}
    for(var col=lane;col<columns;col+=64u){if MODE==1u {value=max(value,load_a(base+col));}else{value+=load_a(base+col);}}
    var reduced=subgroupAdd(value);if MODE==1u {reduced=subgroupMax(value);}
    if sublane==0u {partial[subgroup]=reduced;}workgroupBarrier();
    if lane==0u {var total=partial[0];for(var i=1u;i<groups;i++){if MODE==1u{total=max(total,partial[i]);}else{total+=partial[i];}}
        mean=total;if MODE!=1u {mean/=f32(columns);}if MODE==0u {store_output(row,mean);}}
    workgroupBarrier();if MODE==0u{return;}
    value=0.0;for(var col=lane;col<columns;col+=64u){if MODE==1u{value+=exp(load_a(base+col)-mean);}else{let x=load_a(base+col)-mean;value+=x*x;}}
    reduced=subgroupAdd(value);if sublane==0u{partial[subgroup]=reduced;}workgroupBarrier();
    if lane==0u {scale=0.0;for(var i=0u;i<groups;i++){scale+=partial[i];}if MODE==2u{scale=sqrt(scale/f32(columns)+bitcast<f32>(p(4u)));}}
    workgroupBarrier();
    for(var col=lane;col<columns;col+=64u){if MODE==1u{store_output(base+col,exp(load_a(base+col)-mean)/scale);}
        else{store_output(base+col,((load_a(base+col)-mean)/scale)*load_b(col)+load_c(col));}}
}
