@group(0) @binding(0) var<storage,read> features:array<$INPUT>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read> bias:array<f32>;
@group(0) @binding(3) var<storage,read_write> partial:array<vec4<u32>>;
@group(0) @binding(4) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
var<workgroup> scores:array<f32,64>;
var<workgroup> indices:array<u32,64>;
var<workgroup> sums:array<f32,64>;
var<workgroup> invalid:array<u32,64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
    let column=group.x*64u+lane; let row=group.y;
    var value=-3.402823e38;
    if column<p(2u){
        value=bias[column];
        for(var k=0u;k<p(1u);k++){value=value+f32(features[row*p(1u)+k])*weights[k*p(2u)+column];}
    }
    scores[lane]=value;indices[lane]=column;
    invalid[lane]=u32(!(value<=3.402823e38 && value>=-3.402823e38));
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){
        if lane<stride{
            let other=scores[lane+stride];
            if other>scores[lane] || (other==scores[lane] && indices[lane+stride]<indices[lane]){scores[lane]=other;indices[lane]=indices[lane+stride];}
            invalid[lane]=invalid[lane]|invalid[lane+stride];
        }
        workgroupBarrier();
    }
    // Logit mode already has everything it needs after the winner reduction.
    // The parameter is uniform, so all lanes skip the probability barriers.
    if p(4u)==0u{
        if lane==0u{partial[row*p(3u)+group.x]=vec4<u32>(indices[0],bitcast<u32>(scores[0]),0u,invalid[0]);}
        return;
    }
    sums[lane]=select(0.0,exp(value-scores[0]),column<p(2u) && p(4u)!=0u);
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){if lane<stride{sums[lane]+=sums[lane+stride];}workgroupBarrier();}
    if lane==0u{partial[row*p(3u)+group.x]=vec4<u32>(indices[0],bitcast<u32>(scores[0]),bitcast<u32>(sums[0]),invalid[0]);}
}
