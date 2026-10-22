// Non-overlapping 2x2 stride-two deconvolution: four GEMMs share the input tile.
var<workgroup> weights:array<vec4<f32>,128>;
var<workgroup> pixels:array<f32,128>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let block=group.x+group.y*p(35u);if block>=p(34u){return;}
    let batch=block/(p(32u)*p(33u));let co=(block/p(33u)%p(32u))*8u+lane/8u;
    let pixel=(block%p(33u))*8u+lane%8u;
    let cin=p(3u);let cout=p(4u);let h=p(5u);let w=p(6u);let plane=h*w;
    var sum=vec4<f32>(0.0);if co<cout && p(7u)!=0u {sum=vec4<f32>(load_c(co));}
    for(var base=0u;base<cin;base+=16u){
        for(var i=lane;i<128u;i+=64u){
            let row=(block/p(33u)%p(32u))*8u+i/16u;let ci=base+i%16u;
            weights[i]=vec4<f32>(0.0);if row<cout && ci<cin{let index=(ci*cout+row)*4u;weights[i]=vec4<f32>(load_b(index),load_b(index+1u),load_b(index+2u),load_b(index+3u));}
            let source=(block%p(33u))*8u+i%8u;let channel=base+i/8u;
            pixels[i]=0.0;if source<plane && channel<cin{pixels[i]=load_a((batch*cin+channel)*plane+source);}
        }
        workgroupBarrier();
        for(var k=0u;k<16u;k++){sum+=weights[(lane/8u)*16u+k]*pixels[k*8u+lane%8u];}
        workgroupBarrier();
    }
    if co<cout && pixel<plane {
        let index=(batch*cout+co)*plane*4u+(pixel/w)*w*4u+(pixel%w)*2u;
        store_output(index,epilogue(sum.x,index));store_output(index+1u,epilogue(sum.y,index+1u));
        store_output(index+w*2u,epilogue(sum.z,index+w*2u));store_output(index+w*2u+1u,epilogue(sum.w,index+w*2u+1u));
    }
}
