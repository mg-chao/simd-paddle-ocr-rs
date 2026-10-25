// language: metal4.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
    uint size2;
    uint size3;
    uint size4;
    uint size5;
};

struct type_2 {
    metal::uint4 inner[32];
};
struct Parameters {
    type_2 words;
};
typedef half type_4[1];
struct type_6 {
    float inner[128];
};
struct type_7 {
    float inner[512];
};
struct type_8 {
    float inner[64];
};
struct type_9 {
    float inner[4];
};

uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint p(
    uint i_3,
    constant Parameters& params
) {
    uint _e9 = params.words.inner[metal::min(unsigned(naga_div(i_3, 4u)), 31u)][metal::min(unsigned(naga_mod(i_3, 4u)), 3u)];
    return _e9;
}

float erf_approx(
    float x
) {
    if (metal::abs(x) >= 6.0) {
        return (x < 0.0) ? -1.0 : 1.0;
    }
    float t = 1.0 / (1.0 + (0.3275911 * metal::abs(x)));
    float poly = ((((((((1.0614054 * t) - 1.4531521) * t) + 1.4214138) * t) - 0.28449672) * t) + 0.2548296) * t;
    return ((x < 0.0) ? -1.0 : 1.0) * (1.0 - (poly * metal::exp(-(x) * x)));
}

float output_round(
    float value_1
) {
    if (false) {
        return value_1;
    }
    uint bits = as_type<uint>(value_1);
    uint sign = bits & 2147483648u;
    uint magnitude = bits & 2147483647u;
    if (magnitude >= 2139095040u) {
        return value_1;
    }
    if (magnitude < 947912704u) {
        if (magnitude <= 855638016u) {
            return as_type<float>(sign);
        }
        uint shift = 126u - (magnitude >> 23u);
        uint mantissa = (magnitude & 8388607u) | 8388608u;
        uint rounded = ((mantissa + ((1u << (shift - 1u)) - 1u)) + ((mantissa >> shift) & 1u)) >> shift;
        return as_type<float>(sign | as_type<uint>(static_cast<float>(rounded) * 0.000000059604645));
    }
    uint rounded_1 = (magnitude + 4095u) + ((magnitude >> 13u) & 1u);
    if (rounded_1 >= 1199570944u) {
        return as_type<float>(sign | 2139095040u);
    }
    return as_type<float>(sign | (rounded_1 & 4294959104u));
}

float load_d(
    uint i_4,
    device type_4 const& raw_d,
    constant _mslBufferSizes& _buffer_sizes
) {
    half _e3 = raw_d[metal::min(unsigned(i_4), (_buffer_sizes.size3 - 0 - 2) / 2)];
    return static_cast<float>(_e3);
}

float load_e(
    uint i_5,
    device type_4 const& raw_e,
    constant _mslBufferSizes& _buffer_sizes
) {
    half _e3 = raw_e[metal::min(unsigned(i_5), (_buffer_sizes.size4 - 0 - 2) / 2)];
    return static_cast<float>(_e3);
}

float epilogue(
    float initial,
    uint index,
    device type_4 const& raw_d,
    device type_4 const& raw_e,
    constant Parameters& params,
    constant _mslBufferSizes& _buffer_sizes
) {
    float value_2 = {};
    uint i_6 = 0u;
    uint offset = {};
    float gate = {};
    value_2 = initial;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e133 = i_6;
            i_6 = _e133 + 1u;
        }
        loop_init = false;
        uint _e5 = i_6;
        uint _e7 = p(64u, params);
        if (_e5 < _e7) {
        } else {
            break;
        }
        {
            uint _e10 = i_6;
            uint base_1 = 68u + (_e10 * 3u);
            uint _e14 = p(base_1, params);
            if (_e14 <= 2u) {
                offset = index;
                uint _e20 = p(base_1 + 1u, params);
                if (_e20 == 0u) {
                    offset = 0u;
                }
                uint _e26 = p(base_1 + 1u, params);
                if (_e26 == 1u) {
                    uint _e30 = p(66u, params);
                    uint _e33 = p(65u, params);
                    offset = naga_mod(naga_div(index, _e30), _e33);
                }
                if (_e14 == 1u) {
                    float _e37 = value_2;
                    uint _e38 = offset;
                    float _e39 = load_d(_e38, raw_d, _buffer_sizes);
                    value_2 = _e37 + _e39;
                } else {
                    float _e41 = value_2;
                    uint _e42 = offset;
                    float _e43 = load_e(_e42, raw_e, _buffer_sizes);
                    value_2 = _e41 + _e43;
                }
            } else {
                if (_e14 == 3u) {
                    float _e47 = value_2;
                    float _e48 = value_2;
                    value_2 = (_e48 > 0.0) ? _e47 : 0.0;
                } else {
                    if (_e14 == 4u) {
                        float _e55 = value_2;
                        if (_e55 >= 0.0) {
                            float _e58 = value_2;
                            value_2 = 1.0 / (1.0 + metal::exp(-(_e58)));
                        } else {
                            float _e65 = value_2;
                            float x_1 = metal::exp(_e65);
                            value_2 = x_1 / (1.0 + x_1);
                        }
                    } else {
                        if (_e14 == 5u) {
                            float _e72 = value_2;
                            uint _e75 = p(base_1 + 1u, params);
                            uint _e80 = p(base_1 + 2u, params);
                            value_2 = metal::clamp((_e72 * as_type<float>(_e75)) + as_type<float>(_e80), 0.0, 1.0);
                        } else {
                            if (_e14 == 6u) {
                                float _e88 = value_2;
                                float _e89 = value_2;
                                value_2 = _e88 * metal::clamp((_e89 * 0.16666667) + 0.5, 0.0, 1.0);
                            } else {
                                if (_e14 == 7u) {
                                    float _e100 = value_2;
                                    float _e103 = erf_approx(_e100 / 1.4142135);
                                    float _e106 = value_2;
                                    value_2 = ((_e103 + 1.0) * _e106) * 0.5;
                                } else {
                                    if (_e14 == 8u) {
                                        gate = 0.0;
                                        float _e114 = value_2;
                                        if (_e114 >= 0.0) {
                                            float _e117 = value_2;
                                            gate = 1.0 / (1.0 + metal::exp(-(_e117)));
                                        } else {
                                            float _e124 = value_2;
                                            float x_2 = metal::exp(_e124);
                                            gate = x_2 / (1.0 + x_2);
                                        }
                                        float _e129 = value_2;
                                        float _e130 = gate;
                                        value_2 = _e129 * _e130;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    float _e135 = value_2;
    return _e135;
}

float load_a(
    uint i_7,
    device type_4 const& raw_a,
    constant _mslBufferSizes& _buffer_sizes
) {
    half _e3 = raw_a[metal::min(unsigned(i_7), (_buffer_sizes.size0 - 0 - 2) / 2)];
    return static_cast<float>(_e3);
}

float load_b(
    uint i_8,
    device type_4 const& raw_b,
    constant _mslBufferSizes& _buffer_sizes
) {
    half _e3 = raw_b[metal::min(unsigned(i_8), (_buffer_sizes.size1 - 0 - 2) / 2)];
    return static_cast<float>(_e3);
}

float load_c(
    uint i_9,
    device type_4 const& raw_c,
    constant _mslBufferSizes& _buffer_sizes
) {
    half _e3 = raw_c[metal::min(unsigned(i_9), (_buffer_sizes.size2 - 0 - 2) / 2)];
    return static_cast<float>(_e3);
}

float load_output(
    uint i_10,
    device type_4 const& raw_output,
    constant Parameters& params,
    constant _mslBufferSizes& _buffer_sizes
) {
    uint _e3 = p(60u, params);
    half _e6 = raw_output[metal::min(unsigned(i_10 + _e3), (_buffer_sizes.size5 - 0 - 2) / 2)];
    return static_cast<float>(_e6);
}

void store_output(
    uint i_11,
    float value_3,
    device type_4& raw_output,
    constant Parameters& params,
    constant _mslBufferSizes& _buffer_sizes
) {
    uint _e4 = p(60u, params);
    raw_output[metal::min(unsigned(i_11 + _e4), (_buffer_sizes.size5 - 0 - 2) / 2)] = static_cast<half>(value_3);
    return;
}

uint ap(
    uint i_12,
    constant Parameters& params
) {
    uint _e13 = p((i_12 + ((i_12 >= 60u) ? 1u : 0u)) + ((i_12 >= 63u) ? 4u : 0u), params);
    return _e13;
}

uint address(
    uint index_1,
    uint operand,
    constant Parameters& params
) {
    uint value_4 = {};
    uint start = {};
    uint transform = 0u;
    uint remaining = {};
    uint mapped = {};
    uint axis = {};
    if (false) {
        return index_1;
    }
    value_4 = index_1;
    uint _e8 = p(9u + (operand * 2u), params);
    start = _e8;
    uint _e14 = p(10u + (operand * 2u), params);
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e65 = transform;
            transform = _e65 + 1u;
        }
        loop_init_1 = false;
        uint _e17 = transform;
        if (_e17 < _e14) {
        } else {
            break;
        }
        {
            uint _e19 = value_4;
            remaining = _e19;
            uint _e21 = start;
            uint _e24 = ap(_e21 + 1u, params);
            mapped = _e24;
            uint _e26 = start;
            uint _e27 = ap(_e26, params);
            axis = _e27;
            uint2 loop_bound_2 = uint2(4294967295u);
            while(true) {
                if (metal::all(loop_bound_2 == uint2(0u))) { break; }
                loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
                uint _e29 = axis;
                if (_e29 == 0u) {
                    break;
                }
                uint _e33 = axis;
                axis = _e33 - 1u;
                uint _e35 = remaining;
                uint _e36 = start;
                uint _e39 = axis;
                uint _e41 = ap((_e36 + 2u) + _e39, params);
                uint c = naga_mod(_e35, _e41);
                uint _e43 = remaining;
                uint _e44 = start;
                uint _e47 = axis;
                uint _e49 = ap((_e44 + 2u) + _e47, params);
                remaining = naga_div(_e43, _e49);
                uint _e51 = mapped;
                uint _e52 = start;
                uint _e55 = axis;
                uint _e57 = ap((_e52 + 7u) + _e55, params);
                mapped = _e51 + (c * _e57);
            }
            uint _e60 = mapped;
            value_4 = _e60;
            uint _e61 = start;
            start = _e61 + 12u;
        }
    }
    uint _e67 = value_4;
    return _e67;
}

struct main_Input {
};
#define ATTENTION_FORCE_FALLBACK 0
void attention_original(
  metal::uint3 group
, uint lane
, device type_4 const& raw_a
, device type_4 const& raw_b
, device type_4 const& raw_c
, device type_4 const& raw_d
, device type_4& raw_output
, constant Parameters& params
, threadgroup type_6& queries
, threadgroup type_7& keys
, threadgroup type_7& values
, threadgroup type_8& weights
, threadgroup type_9& maxima
, threadgroup type_9& denominators
, threadgroup type_9& corrections
, constant _mslBufferSizes& _buffer_sizes
) {
    if (lane == 0u) {
        queries = {};
        keys = {};
        values = {};
        weights = {};
        maxima = {};
        denominators = {};
        corrections = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    bool local = {};
    uint packed_stride = 0u;
    uint packed_base = 0u;
    uint i = {};
    bool local_1 = {};
    float value = {};
    bool local_2 = {};
    float acc0_ = 0.0;
    float acc1_ = 0.0;
    uint base = 0u;
    uint i_1 = {};
    bool local_3 = {};
    bool local_4 = {};
    float score = {};
    uint i_2 = {};
    float next = {};
    uint k = {};
    float total = {};
    uint k_1 = {};
    uint k_2 = {};
    uint output_base = {};
    bool local_5 = {};
    uint _e5 = p(1u, params);
    uint block = group.x + (group.y * _e5);
    uint _e9 = p(0u, params);
    if (block >= _e9) {
        return;
    }
    uint _e12 = p(8u, params);
    uint batch = naga_div(block, _e12);
    uint _e15 = p(8u, params);
    uint first = naga_mod(block, _e15) * 4u;
    uint query = naga_div(lane, 16u);
    uint channel = naga_mod(lane, 16u);
    uint row = first + query;
    uint _e25 = p(3u, params);
    uint _e27 = p(4u, params);
    uint _e29 = p(5u, params);
    uint _e31 = p(6u, params);
    if (true) {
        uint _e36 = p(2u, params);
        local = (_e36 & 2u) != 0u;
    } else {
        local = false;
    }
    bool packed = local;
    if (packed) {
        uint _e49 = p(15u, params);
        packed_stride = (3u * _e49) * _e29;
        uint _e53 = p(15u, params);
        uint _e56 = packed_stride;
        uint _e59 = p(15u, params);
        packed_base = ((naga_div(batch, _e53) * _e25) * _e56) + (naga_mod(batch, _e59) * _e29);
    }
    i = lane;
    uint2 loop_bound_3 = uint2(4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (metal::all(loop_bound_3 == uint2(0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        if (!loop_init_2) {
            uint _e123 = i;
            i = _e123 + 64u;
        }
        loop_init_2 = false;
        uint _e64 = i;
        if (_e64 < 128u) {
        } else {
            break;
        }
        {
            uint _e67 = i;
            uint r = first + naga_div(_e67, 32u);
            uint _e71 = i;
            uint c_1 = naga_mod(_e71, 32u);
            uint _e75 = i;
            queries.inner[metal::min(unsigned(_e75), 127u)] = 0.0;
            if (r < _e25) {
                local_1 = c_1 < _e29;
            } else {
                local_1 = false;
            }
            bool _e83 = local_1;
            if (_e83) {
                value = 0.0;
                if (packed) {
                    uint _e86 = packed_base;
                    uint _e87 = packed_stride;
                    float _e91 = load_a((_e86 + (r * _e87)) + c_1, raw_a, _buffer_sizes);
                    value = _e91;
                } else {
                    uint _e97 = address((((batch * _e25) + r) * _e29) + c_1, 0u, params);
                    float _e98 = load_a(_e97, raw_a, _buffer_sizes);
                    value = _e98;
                }
                if (true) {
                    uint _e103 = p(2u, params);
                    local_2 = (_e103 & 1u) != 0u;
                } else {
                    local_2 = false;
                }
                bool _e109 = local_2;
                if (_e109) {
                    float _e110 = value;
                    float _e112 = load_d(0u, raw_d, _buffer_sizes);
                    float _e114 = output_round(_e110 * _e112);
                    value = _e114;
                }
                uint _e116 = i;
                float _e118 = value;
                uint _e120 = p(7u, params);
                queries.inner[metal::min(unsigned(_e116), 127u)] = _e118 * as_type<float>(_e120);
            }
        }
    }
    if (lane < 4u) {
        maxima.inner[metal::min(unsigned(lane), 3u)] = -340282350000000000000000000000000000000.0;
        denominators.inner[metal::min(unsigned(lane), 3u)] = 0.0;
    }
    uint2 loop_bound_4 = uint2(4294967295u);
    bool loop_init_3 = true;
    while(true) {
        if (metal::all(loop_bound_4 == uint2(0u))) { break; }
        loop_bound_4 -= uint2(loop_bound_4.y == 0u, 1u);
        if (!loop_init_3) {
            uint _e373 = base;
            base = _e373 + 16u;
        }
        loop_init_3 = false;
        uint _e140 = base;
        if (_e140 < _e27) {
        } else {
            break;
        }
        {
            i_1 = lane;
            uint2 loop_bound_5 = uint2(4294967295u);
            bool loop_init_4 = true;
            while(true) {
                if (metal::all(loop_bound_5 == uint2(0u))) { break; }
                loop_bound_5 -= uint2(loop_bound_5.y == 0u, 1u);
                if (!loop_init_4) {
                    uint _e222 = i_1;
                    i_1 = _e222 + 64u;
                }
                loop_init_4 = false;
                uint _e143 = i_1;
                if (_e143 < 512u) {
                } else {
                    break;
                }
                {
                    uint _e146 = base;
                    uint _e147 = i_1;
                    uint k_3 = _e146 + naga_div(_e147, 32u);
                    uint _e151 = i_1;
                    uint c_2 = naga_mod(_e151, 32u);
                    uint _e155 = i_1;
                    keys.inner[metal::min(unsigned(_e155), 511u)] = 0.0;
                    uint _e159 = i_1;
                    values.inner[metal::min(unsigned(_e159), 511u)] = 0.0;
                    if (k_3 < _e27) {
                        local_3 = c_2 < _e29;
                    } else {
                        local_3 = false;
                    }
                    bool _e167 = local_3;
                    if (_e167) {
                        if (packed) {
                            uint _e169 = i_1;
                            uint _e171 = packed_base;
                            uint _e172 = packed_stride;
                            uint _e176 = p(15u, params);
                            float _e180 = load_b(((_e171 + (k_3 * _e172)) + (_e176 * _e29)) + c_2, raw_b, _buffer_sizes);
                            keys.inner[metal::min(unsigned(_e169), 511u)] = _e180;
                        } else {
                            uint _e182 = i_1;
                            uint _e189 = address((((batch * _e29) + c_2) * _e27) + k_3, 1u, params);
                            float _e190 = load_b(_e189, raw_b, _buffer_sizes);
                            keys.inner[metal::min(unsigned(_e182), 511u)] = _e190;
                        }
                    }
                    if (k_3 < _e27) {
                        local_4 = c_2 < _e31;
                    } else {
                        local_4 = false;
                    }
                    bool _e196 = local_4;
                    if (_e196) {
                        if (packed) {
                            uint _e198 = i_1;
                            uint _e200 = packed_base;
                            uint _e201 = packed_stride;
                            uint _e206 = p(15u, params);
                            float _e211 = load_c(((_e200 + (k_3 * _e201)) + ((2u * _e206) * _e29)) + c_2, raw_c, _buffer_sizes);
                            values.inner[metal::min(unsigned(_e198), 511u)] = _e211;
                        } else {
                            uint _e213 = i_1;
                            uint _e220 = address((((batch * _e27) + k_3) * _e31) + c_2, 2u, params);
                            float _e221 = load_c(_e220, raw_c, _buffer_sizes);
                            values.inner[metal::min(unsigned(_e213), 511u)] = _e221;
                        }
                    }
                }
            }
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            score = 0.0;
            i_2 = 0u;
            uint2 loop_bound_6 = uint2(4294967295u);
            bool loop_init_5 = true;
            while(true) {
                if (metal::all(loop_bound_6 == uint2(0u))) { break; }
                loop_bound_6 -= uint2(loop_bound_6.y == 0u, 1u);
                if (!loop_init_5) {
                    uint _e250 = i_2;
                    i_2 = _e250 + 1u;
                }
                loop_init_5 = false;
                uint _e229 = i_2;
                if (_e229 < 32u) {
                } else {
                    break;
                }
                {
                    float _e232 = score;
                    uint _e236 = i_2;
                    float _e239 = queries.inner[metal::min(unsigned((query * 32u) + _e236), 127u)];
                    uint _e243 = i_2;
                    float _e246 = keys.inner[metal::min(unsigned((channel * 32u) + _e243), 511u)];
                    score = _e232 + (_e239 * _e246);
                }
            }
            float _e254 = score;
            weights.inner[metal::min(unsigned(lane), 63u)] = _e254;
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            uint _e256 = base;
            uint valid = metal::min(16u, _e27 - _e256);
            if (channel == 0u) {
                float _e263 = maxima.inner[metal::min(unsigned(query), 3u)];
                next = _e263;
                k = 0u;
                uint2 loop_bound_7 = uint2(4294967295u);
                bool loop_init_6 = true;
                while(true) {
                    if (metal::all(loop_bound_7 == uint2(0u))) { break; }
                    loop_bound_7 -= uint2(loop_bound_7.y == 0u, 1u);
                    if (!loop_init_6) {
                        uint _e279 = k;
                        k = _e279 + 1u;
                    }
                    loop_init_6 = false;
                    uint _e267 = k;
                    if (_e267 < valid) {
                    } else {
                        break;
                    }
                    {
                        float _e269 = next;
                        uint _e273 = k;
                        float _e276 = weights.inner[metal::min(unsigned((query * 16u) + _e273), 63u)];
                        next = metal::max(_e269, _e276);
                    }
                }
                float _e283 = maxima.inner[metal::min(unsigned(query), 3u)];
                float _e284 = next;
                float correction = metal::exp(_e283 - _e284);
                total = 0.0;
                k_1 = 0u;
                uint2 loop_bound_8 = uint2(4294967295u);
                bool loop_init_7 = true;
                while(true) {
                    if (metal::all(loop_bound_8 == uint2(0u))) { break; }
                    loop_bound_8 -= uint2(loop_bound_8.y == 0u, 1u);
                    if (!loop_init_7) {
                        uint _e312 = k_1;
                        k_1 = _e312 + 1u;
                    }
                    loop_init_7 = false;
                    uint _e291 = k_1;
                    if (_e291 < valid) {
                    } else {
                        break;
                    }
                    {
                        uint _e296 = k_1;
                        float _e299 = weights.inner[metal::min(unsigned((query * 16u) + _e296), 63u)];
                        float _e300 = next;
                        float weight = metal::exp(_e299 - _e300);
                        uint _e306 = k_1;
                        weights.inner[metal::min(unsigned((query * 16u) + _e306), 63u)] = weight;
                        float _e309 = total;
                        total = _e309 + weight;
                    }
                }
                corrections.inner[metal::min(unsigned(query), 3u)] = correction;
                float _e318 = next;
                maxima.inner[metal::min(unsigned(query), 3u)] = _e318;
                float _e323 = denominators.inner[metal::min(unsigned(query), 3u)];
                float _e325 = total;
                denominators.inner[metal::min(unsigned(query), 3u)] = (_e323 * correction) + _e325;
            }
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            float _e327 = acc0_;
            float _e330 = corrections.inner[metal::min(unsigned(query), 3u)];
            acc0_ = _e327 * _e330;
            float _e332 = acc1_;
            float _e335 = corrections.inner[metal::min(unsigned(query), 3u)];
            acc1_ = _e332 * _e335;
            k_2 = 0u;
            uint2 loop_bound_9 = uint2(4294967295u);
            bool loop_init_8 = true;
            while(true) {
                if (metal::all(loop_bound_9 == uint2(0u))) { break; }
                loop_bound_9 -= uint2(loop_bound_9.y == 0u, 1u);
                if (!loop_init_8) {
                    uint _e371 = k_2;
                    k_2 = _e371 + 1u;
                }
                loop_init_8 = false;
                uint _e339 = k_2;
                if (_e339 < valid) {
                } else {
                    break;
                }
                {
                    uint _e344 = k_2;
                    float weight_1 = weights.inner[metal::min(unsigned((query * 16u) + _e344), 63u)];
                    float _e348 = acc0_;
                    uint _e350 = k_2;
                    float _e355 = values.inner[metal::min(unsigned((_e350 * 32u) + channel), 511u)];
                    acc0_ = _e348 + (weight_1 * _e355);
                    float _e358 = acc1_;
                    uint _e360 = k_2;
                    float _e367 = values.inner[metal::min(unsigned(((_e360 * 32u) + channel) + 16u), 511u)];
                    acc1_ = _e358 + (weight_1 * _e367);
                }
            }
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
    }
    if (row < _e25) {
        output_base = ((batch * _e25) + row) * _e31;
        if (packed) {
            uint _e384 = p(2u, params);
            local_5 = (_e384 & 4u) != 0u;
        } else {
            local_5 = false;
        }
        bool _e390 = local_5;
        if (_e390) {
            uint _e392 = p(15u, params);
            uint _e396 = p(15u, params);
            uint _e399 = p(15u, params);
            uint _e403 = p(15u, params);
            output_base = ((((naga_div(batch, _e392) * _e25) * _e396) + (row * _e399)) + naga_mod(batch, _e403)) * _e31;
        }
        if (channel < _e31) {
            uint _e408 = output_base;
            float _e410 = acc0_;
            float _e413 = denominators.inner[metal::min(unsigned(query), 3u)];
            store_output(_e408 + channel, _e410 / _e413, raw_output, params, _buffer_sizes);
        }
        if ((channel + 16u) < _e31) {
            uint _e418 = output_base;
            float _e422 = acc1_;
            float _e425 = denominators.inner[metal::min(unsigned(query), 3u)];
            store_output((_e418 + channel) + 16u, _e422 / _e425, raw_output, params, _buffer_sizes);
            return;
        } else {
            return;
        }
    } else {
        return;
    }
}

[[max_total_threads_per_threadgroup(64)]] kernel void main_(
  metal::uint3 group [[threadgroup_position_in_grid]]
, uint lane [[thread_index_in_threadgroup]]
, device type_4 const& raw_a [[buffer(0)]]
, device type_4 const& raw_b [[buffer(1)]]
, device type_4 const& raw_c [[buffer(2)]]
, device type_4 const& raw_d [[buffer(3)]]
, device type_4& raw_output [[buffer(5)]]
, constant Parameters& params [[buffer(6)]]
, threadgroup type_6& queries
, threadgroup type_7& keys
, threadgroup type_7& values
, threadgroup type_8& weights
, threadgroup type_9& maxima
, threadgroup type_9& denominators
, threadgroup type_9& corrections
, constant _mslBufferSizes& _buffer_sizes [[buffer(30)]]
, uint simd_width [[threads_per_simdgroup]]
, uint simd_lane [[thread_index_in_simdgroup]]
, uint subgroup [[simdgroup_index_in_threadgroup]]
) {
    if (ATTENTION_FORCE_FALLBACK || simd_width != 32u || p(2u, params) != 7u || p(5u, params) > 32u || p(6u, params) > 32u) {
        attention_original(group, lane, raw_a, raw_b, raw_c, raw_d, raw_output, params, queries, keys, values, weights, maxima, denominators, corrections, _buffer_sizes);
        return;
    }
    if (lane == 0u) {
        queries = {};
        keys = {};
        values = {};
        weights = {};
        maxima = {};
        denominators = {};
        corrections = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// Four queries per group; each SIMD32 contains two independent 16-lane queries.
// Every SIMD lane executes every shuffle. Query halves never read each other.
const uint linear_lane = subgroup * 32u + simd_lane;
const uint block = group.x + group.y * p(1u, params);
if (block >= p(0u, params)) { return; }
const uint batches_per_head = p(8u, params);
const uint batch = block / batches_per_head;
const uint first = (block % batches_per_head) * 4u;
const uint query = linear_lane / 16u;
const uint channel = linear_lane % 16u;
const uint half_base = simd_lane & 16u;
const uint row = first + query;
const uint rows = p(3u, params), count = p(4u, params);
const uint d = p(5u, params), dv = p(6u, params);
const uint packed_stride = 3u * p(15u, params) * d;
const uint packed_base = (batch / p(15u, params)) * rows * packed_stride
    + (batch % p(15u, params)) * d;
for (uint i = linear_lane; i < 128u; i += 64u) {
    const uint r = first + i / 32u, c = i % 32u;
    queries.inner[i] = 0.0;
    if (r < rows && c < d) {
        float value = load_a(packed_base + r * packed_stride + c, raw_a, _buffer_sizes);
        value = output_round(value * load_d(0u, raw_d, _buffer_sizes));
        queries.inner[i] = value * as_type<float>(p(7u, params));
    }
}
float maximum = -340282350000000000000000000000000000000.0;
float denominator = 0.0, acc0 = 0.0, acc1 = 0.0;
for (uint base = 0u; base < count; base += 16u) {
    for (uint i = linear_lane; i < 512u; i += 64u) {
        const uint k = base + i / 32u, c = i % 32u;
        keys.inner[i] = 0.0; values.inner[i] = 0.0;
        if (k < count && c < d) {
            keys.inner[i] = load_b(packed_base + k * packed_stride + p(15u, params) * d + c, raw_b, _buffer_sizes);
        }
        if (k < count && c < dv) {
            values.inner[i] = load_c(packed_base + k * packed_stride + 2u * p(15u, params) * d + c, raw_c, _buffer_sizes);
        }
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float score = 0.0;
    for (uint i = 0u; i < 32u; ++i) {
        score = score + (queries.inner[query * 32u + i] * keys.inner[channel * 32u + i]);
    }
    const uint valid = metal::min(16u, count - base);
    float next = maximum;
    for (uint k = 0u; k < valid; ++k) {
        next = metal::max(next, metal::simd_shuffle(score, half_base + k));
    }
    const float correction = metal::exp(maximum - next);
    const float local_weight = metal::exp(score - next);
    float total = 0.0;
    acc0 = acc0 * correction; acc1 = acc1 * correction;
    for (uint k = 0u; k < valid; ++k) {
        const float weight = metal::simd_shuffle(local_weight, half_base + k);
        total = total + weight;
        acc0 = acc0 + (weight * values.inner[k * 32u + channel]);
        acc1 = acc1 + (weight * values.inner[k * 32u + channel + 16u]);
    }
    denominator = (denominator * correction) + total;
    maximum = next;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
if (row < rows) {
    const uint output_base = (((batch / p(15u, params)) * rows * p(15u, params))
        + row * p(15u, params) + batch % p(15u, params)) * dv;
    if (channel < dv) { store_output(output_base + channel, acc0 / denominator, raw_output, params, _buffer_sizes); }
    if (channel + 16u < dv) { store_output(output_base + channel + 16u, acc1 / denominator, raw_output, params, _buffer_sizes); }
}
}
