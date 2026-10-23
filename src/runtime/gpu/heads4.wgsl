@group(0) @binding(0) var<storage,read> features:array<$INPUT>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read> bias:array<f32>;
@group(0) @binding(3) var<storage,read_write> partial:array<vec4<u32>>;
@group(0) @binding(4) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
var<workgroup> scores:array<vec4<f32>,64>;
var<workgroup> indices:array<vec4<u32>,64>;
var<workgroup> sums:array<vec4<f32>,64>;
var<workgroup> invalid:array<vec4<u32>,64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
    let column=group.x*64u+lane;let row=group.y*4u;var value=vec4<f32>(-3.402823e38);
    if column<p(2u){
        value=vec4<f32>(bias[column]);
        for(var k=0u;k<p(1u);k++){
            let weight=weights[k*p(2u)+column];
            value.x+=f32(features[row*p(1u)+k])*weight;
            if row+1u<p(0u){value.y+=f32(features[(row+1u)*p(1u)+k])*weight;}
            if row+2u<p(0u){value.z+=f32(features[(row+2u)*p(1u)+k])*weight;}
            if row+3u<p(0u){value.w+=f32(features[(row+3u)*p(1u)+k])*weight;}
        }
    }
    scores[lane]=value;indices[lane]=vec4<u32>(column);
    invalid[lane]=select(vec4<u32>(0u),vec4<u32>(1u),!((value<=vec4<f32>(3.402823e38))&(value>=vec4<f32>(-3.402823e38))));
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){
        if lane<stride{
            let other=scores[lane+stride];let index=indices[lane+stride];
            let replace=(other>scores[lane])|((other==scores[lane])&(index<indices[lane]));
            scores[lane]=select(scores[lane],other,replace);indices[lane]=select(indices[lane],index,replace);invalid[lane]|=invalid[lane+stride];
        }workgroupBarrier();
    }
    sums[lane]=vec4<f32>(0.0);if column<p(2u) && p(4u)!=0u{sums[lane]=exp(value-scores[0]);}workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){if lane<stride{sums[lane]+=sums[lane+stride];}workgroupBarrier();}
    if lane==0u{for(var r=0u;r<4u;r++){if row+r<p(0u){partial[(row+r)*p(3u)+group.x]=vec4<u32>(indices[0][r],bitcast<u32>(scores[0][r]),bitcast<u32>(sums[0][r]),invalid[0][r]);}}}
}
