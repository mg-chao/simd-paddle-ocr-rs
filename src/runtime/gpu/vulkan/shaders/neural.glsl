#include "storage.glsl"
layout(std430, set=0, binding=0) readonly buffer OperandA { STORAGE_0 a[]; };
layout(std430, set=0, binding=1) readonly buffer OperandB { STORAGE_1 b[]; };
layout(std430, set=0, binding=2) readonly buffer OperandC { STORAGE_2 c[]; };
layout(std430, set=0, binding=3) readonly buffer OperandD { STORAGE_3 d[]; };
layout(std430, set=0, binding=4) readonly buffer OperandE { STORAGE_4 e[]; };
layout(std430, set=0, binding=5) buffer Output { STORAGE_5 output_data[]; };
layout(std140, set=0, binding=6) uniform Parameters { uvec4 words[32]; };
uint p(uint index) { return words[index/4u][index%4u]; }
float load_a(uint index) { return float(a[index]); }
float load_b(uint index) { return float(b[index]); }
float load_c(uint index) { return float(c[index]); }
float load_d(uint index) { return float(d[index]); }
float load_e(uint index) { return float(e[index]); }
float load_output(uint index) { return float(output_data[index+p(60u)]); }
// Round before narrowing, independently of the driver's conversion mode.
float output_round(float value) {
#if !(PRECISION_MASK & 32)
    return value;
#else
    uint bits=floatBitsToUint(value), sign_bit=bits&0x80000000u, magnitude=bits&0x7fffffffu;
    if(magnitude>=0x7f800000u) return value;
    if(magnitude<0x38800000u) {
        if(magnitude<=0x33000000u) return uintBitsToFloat(sign_bit);
        uint shift=126u-(magnitude>>23u), mantissa=(magnitude&0x007fffffu)|0x00800000u;
        uint rounded=(mantissa+((1u<<(shift-1u))-1u)+((mantissa>>shift)&1u))>>shift;
        return uintBitsToFloat(sign_bit|floatBitsToUint(float(rounded)*5.960464477539063e-8));
    }
    uint rounded=magnitude+0xfffu+((magnitude>>13u)&1u);
    if(rounded>=0x47800000u) return uintBitsToFloat(sign_bit|0x7f800000u);
    return uintBitsToFloat(sign_bit|(rounded&0xffffe000u));
#endif
}
void store_output(uint index, float value) { output_data[index+p(60u)]=STORAGE_5(output_round(value)); }
float erf_approx(float x) {
    if(abs(x)>=6.0) return x<0.0 ? -1.0 : 1.0;
    float t=1.0/(1.0+0.3275911*abs(x));
    float poly=(((((1.0614054*t-1.4531521)*t)+1.4214138)*t-0.28449672)*t+0.2548296)*t;
    return (x<0.0 ? -1.0 : 1.0)*(1.0-poly*exp(-x*x));
}
float epilogue(float value, uint index) {
    for(uint i=0u;i<p(64u);i++) {
        uint base=68u+i*3u, op=p(base);
        if(op<=2u) {
            uint offset=index;
            if(p(base+1u)==0u) offset=0u;
            if(p(base+1u)==1u) offset=(index/p(66u))%p(65u);
            value += op==1u ? load_d(offset) : load_e(offset);
        } else if(op==3u) value=value>0.0 ? value : 0.0;
        else if(op==4u) { float x=exp(-abs(value)); value=value>=0.0 ? 1.0/(1.0+x) : x/(1.0+x); }
        else if(op==5u) value=clamp(value*uintBitsToFloat(p(base+1u))+uintBitsToFloat(p(base+2u)),0.0,1.0);
        else if(op==6u) value=value*clamp(value*0.16666667+0.5,0.0,1.0);
        else if(op==7u) value=((erf_approx(value/1.4142135623730951)+1.0)*value)*0.5;
        else if(op==8u) { float x=exp(-abs(value)); value*=value>=0.0 ? 1.0/(1.0+x) : x/(1.0+x); }
    }
    return value;
}
