struct Params { words: array<vec4<u32>,64> };
@group(0) @binding(0) var<storage,read> partial:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read_write> winners:array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
var<workgroup> selected:array<vec4<u32>,64>;
// vec4(class,scoreBits,originalTile,invalidOR); first tile resolves duplicate
// class and signed-zero ties exactly as the original ascending scan.
fn better(a:vec4<u32>,b:vec4<u32>)->bool{
 let x=bitcast<f32>(a.y);let y=bitcast<f32>(b.y);
 return x>y || (x==y && (a.x<b.x || (a.x==b.x && a.z<b.z)));
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let row=group.x;if row>=p(0u) || p(3u)>512u{return;}
    var maximum=-3.402823e38;
    var local=vec4<u32>(0xffffffffu,bitcast<u32>(maximum),0xffffffffu,0u);
    for(var i=lane;i<p(3u);i+=64u){
        let item=partial[row*p(3u)+i];local.w|=item.w;
        let candidate=vec4<u32>(item.x,item.y,i,0u);
        if better(candidate,local){local.x=candidate.x;local.y=candidate.y;local.z=candidate.z;}
    }
    selected[lane]=local;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){
        if lane<stride{
            let a=selected[lane];let b=selected[lane+stride];var choice=a;
            if better(b,a){choice.x=b.x;choice.y=b.y;choice.z=b.z;}
            choice.w=a.w|b.w;selected[lane]=choice;
        }
        workgroupBarrier();
    }
    if lane==0u{let winner=selected[0];winners[row]=vec4<u32>(winner.x,winner.y,0u,winner.w);}
}
