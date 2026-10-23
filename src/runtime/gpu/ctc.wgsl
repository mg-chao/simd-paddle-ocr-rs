@group(0) @binding(0) var<storage,read> winners:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read> lengths:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
// Preserve compensated-score rounding across WGSL backend reassociation.
fn score_round(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)&p(255u));}
var<workgroup> scan:array<u32,64>;
var<workgroup> emitted:u32;
var<workgroup> bytes:u32;
var<workgroup> error:atomic<u32>;
var<workgroup> sum:f32;
var<workgroup> correction:f32;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
    let seq=group.x;let steps=p(1u);let base=seq*(steps+1u)*6u;
    if lane==0u{emitted=0u;bytes=0u;sum=0.0;correction=0.0;atomicStore(&error,0u);}workgroupBarrier();
    for(var tile=0u;tile<steps;tile+=64u){
        let step=tile+lane;var item=vec4<u32>(0u);var marked=false;
        if step<steps{
            item=winners[seq*steps+step];
            if item.x>=p(2u) || item.w!=0u{atomicOr(&error,1u);}
            var previous=0u;if step>0u{previous=winners[seq*steps+step-1u].x;}
            marked=item.x!=0u && item.x!=previous && item.x<p(2u);
        }
        scan[lane]=u32(marked);workgroupBarrier();
        for(var delta=1u;delta<64u;delta*=2u){
            var value=0u;if lane>=delta{value=scan[lane-delta];}workgroupBarrier();
            scan[lane]+=value;workgroupBarrier();
        }
        if marked{
            let record=base+(emitted+scan[lane])*6u;
            var end=step+1u;
            while end<steps && winners[seq*steps+end].x==item.x{end++;}
            output[record]=item.x;output[record+1u]=item.y;output[record+2u]=step;output[record+3u]=end;
            output[record+4u]=lengths[item.x];output[record+5u]=0u;
        }
        storageBarrier();workgroupBarrier();
        if lane==0u{
            for(var i=0u;i<scan[63];i++){
                let record=base+(emitted+i+1u)*6u;
                let add=output[record+4u];if bytes>0xffffffffu-add{atomicOr(&error,2u);}else{bytes+=add;}
                let value=score_round(bitcast<f32>(output[record+1u])-correction);let next=score_round(sum+value);correction=score_round(score_round(next-sum)-value);sum=next;
            }
            emitted+=scan[63];
        }
        workgroupBarrier();
    }
    if lane==0u{
        let span_bytes=select(0u,emitted*p(5u),p(3u)!=0u);
        if bytes>p(4u) || span_bytes>p(4u)-min(bytes,p(4u)){atomicOr(&error,2u);}
        output[base]=emitted;output[base+1u]=bytes;output[base+2u]=bitcast<u32>(select(0.0,sum/f32(max(emitted,1u)),emitted>0u));
        output[base+3u]=atomicLoad(&error);output[base+4u]=steps;output[base+5u]=0u;
    }
}
