@group(0) @binding(0) var<storage,read> partial:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read_write> winners:array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    let row=id.x;if row>=p(0u){return;}
    var best=0xffffffffu;var maximum=-3.402823e38;var invalid=0u;
    for(var i=0u;i<p(3u);i++){
        let item=partial[row*p(3u)+i];let value=bitcast<f32>(item.y);invalid|=item.w;
        if value>maximum || (value==maximum && item.x<best){maximum=value;best=item.x;}
    }
    var score=maximum;
    if p(4u)!=0u{
        var sum=0.0;
        for(var i=0u;i<p(3u);i++){let item=partial[row*p(3u)+i];sum+=bitcast<f32>(item.z)*exp(bitcast<f32>(item.y)-maximum);}
        score=1.0/sum;
    }
    winners[row]=vec4<u32>(best,bitcast<u32>(score),0u,invalid);
}
