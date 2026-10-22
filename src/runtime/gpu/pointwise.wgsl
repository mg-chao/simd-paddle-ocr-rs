// Padded NHWC input, canonical OI weights, canonical NCHW output.
// A 16x16 output tile; four F32 accumulators per lane.
var<workgroup> left:array<f32,256>;
var<workgroup> right:array<f32,256>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let block=group.x+group.y*p(34u);let channels=p(4u);let plane=p(7u)*p(8u);let cin=p(3u);let padded=p(36u);
    let tiles=p(32u)*p(33u);if block>=tiles*p(35u){return;}
    let batch=block/tiles;let row=(block%tiles/p(33u))*16u;let col=(block%p(33u))*16u;
    let lr=lane/16u;let lc=lane%16u;var sums:array<f32,4>;
    for(var j=0u;j<4u;j++){sums[j]=0.0;if p(19u)!=0u && col+lc<channels{sums[j]=load_c(col+lc);}}
    for(var base=0u;base<cin;base+=16u){
        for(var i=lane;i<256u;i+=64u){
            let pixel=row+i/16u;let k=base+i%16u;left[i]=0.0;
            if pixel<plane && k<cin{left[i]=load_a((batch*plane+pixel)*padded+k);}
            let channel=col+i%16u;let kk=base+i/16u;right[i]=0.0;
            if channel<channels && kk<cin{right[i]=load_b(channel*cin+kk);}
        }
        workgroupBarrier();
        for(var k=0u;k<16u;k++){for(var j=0u;j<4u;j++){sums[j]+=left[(lr+j*4u)*16u+k]*right[k*16u+lc];}}
        workgroupBarrier();
    }
    if col+lc<channels{for(var j=0u;j<4u;j++){let pixel=row+lr+j*4u;if pixel<plane{
        let index=(batch*channels+col+lc)*plane+pixel;store_output(index,epilogue(sums[j],index));}}}
}
