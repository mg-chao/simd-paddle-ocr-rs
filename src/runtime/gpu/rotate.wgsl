@group(0) @binding(0) var<storage,read> pixels:array<u32>;
@group(0) @binding(1) var<storage,read> winner:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>){
    let i=id.x+id.y*grid.x*64u;let n=params.words[0].x;if i>=n{return;}
    let row=params.words[0].z*4u;
    let rotate=winner[row]==1u && bitcast<f32>(winner[row+1u])>bitcast<f32>(params.words[0].y);
    output[i]=pixels[select(i,n-1u-i,rotate)];
}
