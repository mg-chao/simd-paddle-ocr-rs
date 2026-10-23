@group(0) @binding(0) var<storage,read> parents:array<u32>;
@group(0) @binding(1) var<storage,read> ranks:array<u32>;
@group(0) @binding(2) var<storage,read> metadata:array<u32>;
@group(0) @binding(3) var<storage,read_write> rows:array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>){
    let i=id.x+id.y*grid.x*64u;
    if p(2u)==0u{if i<metadata[3]{atomicStore(&rows[i*2u],p(0u));atomicStore(&rows[i*2u+1u],0u);}return;}
    if i>=p(0u)*p(1u){return;}
    let rank=ranks[parents[i]];if rank==0xffffffffu{return;}
    let base=16u+rank*8u;let row=metadata[base+5u]+i/p(0u)-metadata[base+2u];
    atomicMin(&rows[row*2u],i%p(0u));atomicMax(&rows[row*2u+1u],i%p(0u));
}
