// Folded Q/K/V affine movement chain; zero transforms keeps the canonical path.
fn ap(i:u32)->u32 {return p(i+select(0u,1u,i>=60u)+select(0u,4u,i>=63u));}
fn address(index:u32,operand:u32)->u32 {
    if !$VIEWS {return index;}
    var value=index;var start=p(9u+operand*2u);let count=p(10u+operand*2u);
    for(var transform=0u;transform<count;transform++){
        var remaining=value;var mapped=ap(start+1u);var axis=ap(start);
        loop{if axis==0u{break;}axis--;let c=remaining%ap(start+2u+axis);remaining/=ap(start+2u+axis);mapped+=c*ap(start+7u+axis);}
        value=mapped;start+=12u;
    }return value;
}
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
    let packed=$VIEWS && (p(2u)&2u)!=0u;
    var packed_stride=0u;var packed_base=0u;
    if packed {packed_stride=3u*p(15u)*d;packed_base=(batch/p(15u))*rows*packed_stride+(batch%p(15u))*d;}
    for(var i=lane;i<128u;i+=64u){let r=first+i/32u;let c=i%32u;queries[i]=0.0;
        if r<rows && c<d{
            var value=0.0;
            if packed {value=load_a(packed_base+r*packed_stride+c);}
            else {value=load_a(address((batch*rows+r)*d+c,0u));}
            if $VIEWS && (p(2u)&1u)!=0u {value=output_round(value*load_d(0u));}
            queries[i]=value*bitcast<f32>(p(7u));}}
    if lane<4u{maxima[lane]=-3.402823466e38;denominators[lane]=0.0;}
    var acc0=0.0;var acc1=0.0;
    for(var base=0u;base<count;base+=16u){
        for(var i=lane;i<512u;i+=64u){let k=base+i/32u;let c=i%32u;keys[i]=0.0;values[i]=0.0;
            if k<count && c<d{
                if packed {keys[i]=load_b(packed_base+k*packed_stride+p(15u)*d+c);}
                else {keys[i]=load_b(address((batch*d+c)*count+k,1u));}}
            if k<count && c<dv{
                if packed {values[i]=load_c(packed_base+k*packed_stride+2u*p(15u)*d+c);}
                else {values[i]=load_c(address((batch*count+k)*dv+c,2u));}}}
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
        var output_base=(batch*rows+row)*dv;
        if packed && (p(2u)&4u)!=0u {output_base=((batch/p(15u))*rows*p(15u)+row*p(15u)+batch%p(15u))*dv;}
        if channel<dv{store_output(output_base+channel,acc0/denominators[query]);}
        if channel+16u<dv{store_output(output_base+channel+16u,acc1/denominators[query]);}
    }
}
