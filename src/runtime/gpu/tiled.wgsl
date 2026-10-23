// TILE outputs per matrix axis, K tile 16, 64 threads and register accumulators.
const TILE: u32 = $TILEu;
const CONV: bool = $CONV;
const POINTWISE: bool = $POINTWISE;
var<workgroup> left: array<f32, $SHARED>;
var<workgroup> right: array<f32, $SHARED>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let block=group.x+group.y*p(34u);
    let batch=block/(p(32u)*p(33u));
    if batch>=p(35u) { return; }
    let mr=(block/p(33u))%p(32u)*TILE; let nc=block%p(33u)*TILE;
    var rows=p(3u); var inner=p(4u); var columns=p(5u);
    var ai=0u; var bi=0u;
    if CONV { rows=p(4u); inner=p(9u)*p(11u)*p(12u); columns=p(7u)*p(8u); }
    else {
        var remaining=batch; var axis=p(6u);
        loop {if axis==0u {break;} axis--; let coordinate=remaining%p(8u+axis); remaining/=p(8u+axis);
            ai+=coordinate*p(16u+axis); bi+=coordinate*p(24u+axis); }
    }
    let local_row=lane/TILE; let local_column=lane%TILE;
    // Named scalar accumulators avoid dynamic private-array indexing/spills.
    $INIT
    for(var base=0u;base<inner;base+=16u) {
        for(var t=lane;t<TILE*16u;t+=64u) {
            let row=mr+t/16u; let k=base+t%16u; var value=0.0;
            if row<rows && k<inner {
                if CONV {value=load_b(row*inner+k);} else {value=load_a(ai+row*inner+k);}
            }
            left[t]=value;
            let col=nc+t%TILE; let rk=base+t/TILE; value=0.0;
            if col<columns && rk<inner {
                if CONV {
                  if POINTWISE {
                    value=load_a((batch*p(3u)+rk)*columns+col);
                    if $PRESCALE {if p(52u)==1u {value*=load_d(batch*p(53u)+rk);}else if p(52u)==2u {value*=load_e(batch*p(53u)+rk);}}
                  } else {
                    let kernel=p(11u)*p(12u); let ci=rk/kernel;
                    let iy=i32((col/p(8u))*p(13u)+(rk%kernel)/p(12u))-i32(p(17u));
                    let ix=i32((col%p(8u))*p(14u)+rk%p(12u))-i32(p(18u));
                    if iy>=0 && ix>=0 && iy<i32(p(5u)) && ix<i32(p(6u)) {
                        value=load_a(((batch*p(3u)+ci)*p(5u)+u32(iy))*p(6u)+u32(ix));
                        if $PRESCALE {if p(52u)==1u {value*=load_d(batch*p(53u)+ci);}else if p(52u)==2u {value*=load_e(batch*p(53u)+ci);}}
                    }
                  }
                } else { value=load_b(bi+rk*columns+col); }
            }
            right[t]=value;
        }
        workgroupBarrier();
        for(var k=0u;k<min(16u,inner-base);k++) {
            let right_value=right[k*TILE+local_column];
            $ACCUMULATE
        }
        workgroupBarrier();
    }
    $STORE
}
