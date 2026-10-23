@group(0) @binding(0) var<storage,read> input:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@group(0) @binding(2) var<uniform> params:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>){
    let word=id.x+id.y*grid.x*64u;let bytes=params.words[0].x;if word*4u>=bytes{return;}var value=0u;
    for(var b=0u;b<4u;b++){let index=word*4u+b;if index<bytes{value|=((input[index/3u]>>((index%3u)*8u))&255u)<<(b*8u);}}
    output[word]=value;
}
