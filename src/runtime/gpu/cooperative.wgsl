// Experimental 8x8 FP32. Each subgroup owns a disjoint tile; no lane-width assumption.
alias MA = coop_mat8x8<f32, A>;
alias MB = coop_mat8x8<f32, B>;
alias MC = coop_mat8x8<f32, C>;
var<workgroup> left:array<f32,1024>;
var<workgroup> right:array<f32,1024>;
var<workgroup> result:array<f32,1024>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(subgroup_id) sg:u32,
        @builtin(subgroup_invocation_id) lane:u32, @builtin(subgroup_size) width:u32,
        @builtin(num_subgroups) groups:u32) {
    let block=group.x+group.y*p(34u);
    let tile=block*groups+sg;
    let batch=tile/(p(32u)*p(33u));
    let row=(tile/p(33u))%p(32u)*8u; let col=tile%p(33u)*8u;
    let m=p(3u);let k=p(4u);let n=p(5u);let offset=sg*64u;
    var ai=0u;var bi=0u;var remaining=batch;var axis=p(6u);
    loop {if axis==0u{break;}axis--;let c=remaining%p(8u+axis);remaining/=p(8u+axis);ai+=c*p(16u+axis);bi+=c*p(24u+axis);}
    for(var i=lane;i<64u;i+=width){result[offset+i]=0.0;}
    workgroupBarrier();
    var acc=coopLoadT<MC>(&result[offset],8u);
    for(var base=0u;base<k;base+=8u){
        for(var i=lane;i<64u;i+=width){
            let r=i/8u;let c=i%8u;var a=0.0;var b=0.0;
            if batch<p(35u) && row+r<m && base+c<k {a=load_a(ai+(row+r)*k+base+c);}
            if batch<p(35u) && base+r<k && col+c<n {b=load_b(bi+(base+r)*n+col+c);}
            left[offset+i]=a;right[offset+i]=b;
        }
        workgroupBarrier();
        let a=coopLoadT<MA>(&left[offset],8u);let b=coopLoadT<MB>(&right[offset],8u);
        acc=coopMultiplyAdd(a,b,acc);
        workgroupBarrier();
    }
    coopStoreT(acc,&result[offset],8u);workgroupBarrier();
    for(var i=lane;i<64u;i+=width){let r=row+i/8u;let c=col+i%8u;
        if batch<p(35u) && r<m && c<n {let index=(batch*m+r)*n+c;store_output(index,epilogue(result[offset+i],index));}}
}
