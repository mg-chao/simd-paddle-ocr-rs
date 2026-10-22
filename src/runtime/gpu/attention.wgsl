// C# online-softmax algorithm, retiled across four queries and sixteen keys.
// Each lane owns one score and two output channels. Scalar accumulators avoid
// private dynamic arrays spilling to device memory on native WGSL compilers.
// Q [batch,queries,D], K [batch,D,keys], V [batch,keys,Dv], D,Dv <= 32.
var<workgroup> queries:array<f32,128>;
var<workgroup> keys:array<f32,512>;
var<workgroup> values:array<f32,512>;
var<workgroup> weights:array<f32,64>;
var<workgroup> maxima:array<f32,4>;
var<workgroup> denominators:array<f32,4>;
var<workgroup> corrections:array<f32,4>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let block=group.x+group.y*p(1u);if block>=p(0u){return;}
    let batch=block/p(8u);let first=(block%p(8u))*4u;
    let query=lane/16u;let channel=lane%16u;let row=first+query;
    let rows=p(3u);let count=p(4u);let d=p(5u);let dv=p(6u);
    for(var i=lane;i<128u;i+=64u){let r=first+i/32u;let c=i%32u;queries[i]=0.0;
        if r<rows && c<d{queries[i]=load_a((batch*rows+r)*d+c)*bitcast<f32>(p(7u));}}
    if lane<4u{maxima[lane]=-3.402823466e38;denominators[lane]=0.0;}
    var acc0=0.0;var acc1=0.0;
    for(var base=0u;base<count;base+=16u){
        for(var i=lane;i<512u;i+=64u){let k=base+i/32u;let c=i%32u;keys[i]=0.0;values[i]=0.0;
            if k<count && c<d{keys[i]=load_b((batch*d+c)*count+k);}
            if k<count && c<dv{values[i]=load_c((batch*count+k)*dv+c);}}
        workgroupBarrier();
        var score=0.0;
        for(var i=0u;i<32u;i++){score+=queries[query*32u+i]*keys[channel*32u+i];}
        weights[lane]=score;workgroupBarrier();
        let valid=min(16u,count-base);
        if channel==0u{
            var next=maxima[query];for(var k=0u;k<valid;k++){next=max(next,weights[query*16u+k]);}
            let correction=exp(maxima[query]-next);var total=0.0;
            for(var k=0u;k<valid;k++){let weight=exp(weights[query*16u+k]-next);weights[query*16u+k]=weight;total+=weight;}
            corrections[query]=correction;maxima[query]=next;denominators[query]=denominators[query]*correction+total;
        }
        workgroupBarrier();
        acc0*=corrections[query];acc1*=corrections[query];
        for(var k=0u;k<valid;k++){let weight=weights[query*16u+k];acc0+=weight*values[k*32u+channel];acc1+=weight*values[k*32u+channel+16u];}
        workgroupBarrier();
    }
    if row<rows{
        if channel<dv{store_output((batch*rows+row)*dv+channel,acc0/denominators[query]);}
        if channel+16u<dv{store_output((batch*rows+row)*dv+channel+16u,acc1/denominators[query]);}
    }
}
