// Preserve IEEE round-to-nearest, ties-to-even independently of driver conversion
// defaults and compiler folding across removed tensor stores.
fn round_to_half(value:f32)->f32 {
    let bits=bitcast<u32>(value);let sign=bits&0x80000000u;let magnitude=bits&0x7fffffffu;
    if magnitude>=0x7f800000u {return value;}
    if magnitude<0x38800000u {
        if magnitude<=0x33000000u {return bitcast<f32>(sign);}
        let shift=126u-(magnitude>>23u);
        let mantissa=(magnitude&0x007fffffu)|0x00800000u;
        let rounded=(mantissa+((1u<<(shift-1u))-1u)+((mantissa>>shift)&1u))>>shift;
        return bitcast<f32>(sign|bitcast<u32>(f32(rounded)*5.960464477539063e-8));
    }
    let rounded=magnitude+0x00000fffu+((magnitude>>13u)&1u);
    if rounded>=0x47800000u {return bitcast<f32>(sign|0x7f800000u);}
    return bitcast<f32>(sign|(rounded&0xffffe000u));
}
