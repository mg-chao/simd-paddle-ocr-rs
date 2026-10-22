const K:u32=$KERNELu;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
        // Adjacent lanes read adjacent pixels, including the stride-two case.
        let index=id.x+id.y*p(1u); if index>=p(0u) {return;}
        let co=(index/(p(7u)*p(8u)))%p(4u); let batch=index/(p(7u)*p(8u)*p(4u));
        let ox=index%p(8u); let oy=(index/p(8u))%p(7u);
        var sum=0.0;if p(19u)!=0u {sum=load_c(co);}
        for(var ky=0u;ky<K;ky++){for(var kx=0u;kx<K;kx++){
            let iy=i32(oy*p(13u)+ky)-i32(p(17u));let ix=i32(ox*p(14u)+kx)-i32(p(18u));
            if iy>=0 && ix>=0 && iy<i32(p(5u)) && ix<i32(p(6u)) {
                sum=sum+load_a(((batch*p(3u)+co)*p(5u)+u32(iy))*p(6u)+u32(ix))*load_b((co*K+ky)*K+kx);
            }
        }}
        store_output(index,epilogue(sum,index));
}
