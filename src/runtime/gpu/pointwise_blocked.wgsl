// FP32 NCHW 1x1 convolution: 32x32 outputs, K=16, 64 threads.
// Each thread owns four rows and four adjacent columns, reusing every
// shared-memory weight across four accumulators without a layout conversion.
var<workgroup> weights: array<f32, 512>;
var<workgroup> features: array<vec4<f32>, 128>;

fn finish_row(row:u32, column:u32, batch:u32, columns:u32, value:vec4<f32>) {
    if row>=p(4u) {return;}
    for(var i=0u;i<4u;i++) {
        if column+i<columns {
            let index=(batch*p(4u)+row)*columns+column+i;
            store_output(index,epilogue(value[i],index));
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let block=group.x+group.y*p(34u);
    let batch=block/(p(32u)*p(33u));
    if batch>=p(35u) {return;}
    let mr=(block/p(33u))%p(32u)*32u;
    let nc=block%p(33u)*32u;
    let inner=p(3u); let rows=p(4u); let columns=p(7u)*p(8u);
    let row=lane/8u*4u; let column=lane%8u;
    var sum0=vec4<f32>(0.0); var sum1=vec4<f32>(0.0);
    var sum2=vec4<f32>(0.0); var sum3=vec4<f32>(0.0);
    if p(19u)!=0u {
        if mr+row<rows {sum0=vec4<f32>(load_c(mr+row));}
        if mr+row+1u<rows {sum1=vec4<f32>(load_c(mr+row+1u));}
        if mr+row+2u<rows {sum2=vec4<f32>(load_c(mr+row+2u));}
        if mr+row+3u<rows {sum3=vec4<f32>(load_c(mr+row+3u));}
    }
    for(var base=0u;base<inner;base+=16u) {
        for(var i=lane;i<512u;i+=64u) {
            let r=mr+i/16u; let k=base+i%16u;
            var value=0.0;
            if r<rows && k<inner {value=load_b(r*inner+k);}
            weights[i]=value;
        }
        for(var i=lane;i<128u;i+=64u) {
            let k=base+i/8u; let c=nc+(i%8u)*4u;
            var value=vec4<f32>(0.0);
            if k<inner {
                let start=(batch*inner+k)*columns+c;
                if c<columns {value.x=load_a(start);}
                if c+1u<columns {value.y=load_a(start+1u);}
                if c+2u<columns {value.z=load_a(start+2u);}
                if c+3u<columns {value.w=load_a(start+3u);}
            }
            features[i]=value;
        }
        workgroupBarrier();
        for(var k=0u;k<min(16u,inner-base);k++) {
            let value=features[k*8u+column];
            sum0+=weights[row*16u+k]*value;
            sum1+=weights[(row+1u)*16u+k]*value;
            sum2+=weights[(row+2u)*16u+k]*value;
            sum3+=weights[(row+3u)*16u+k]*value;
        }
        workgroupBarrier();
    }
    finish_row(mr+row,nc+column*4u,batch,columns,sum0);
    finish_row(mr+row+1u,nc+column*4u,batch,columns,sum1);
    finish_row(mr+row+2u,nc+column*4u,batch,columns,sum2);
    finish_row(mr+row+3u,nc+column*4u,batch,columns,sum3);
}
