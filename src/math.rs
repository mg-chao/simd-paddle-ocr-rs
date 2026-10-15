//! Portable scalar exponential matching the native Windows F32 primitive.
//!
//! Adapted from AMD win-libm expf.asm (MIT, copyright 2002–2019 AMD).
//! https://github.com/amd/win-libm/blob/master/expf.asm
//! Native scalar expf internally uses a double table and cubic with FMA.
//! Keeping that primitive's operation order avoids host-libm rounding drift;
//! every argument and graph/public-head operation around it remains F32.

const TWO_TO_J_OVER_64: [u64; 64] = [
    0x3ff0000000000000,
    0x3ff02c9a3e778061,
    0x3ff059b0d3158574,
    0x3ff0874518759bc8,
    0x3ff0b5586cf9890f,
    0x3ff0e3ec32d3d1a2,
    0x3ff11301d0125b51,
    0x3ff1429aaea92de0,
    0x3ff172b83c7d517b,
    0x3ff1a35beb6fcb75,
    0x3ff1d4873168b9aa,
    0x3ff2063b88628cd6,
    0x3ff2387a6e756238,
    0x3ff26b4565e27cdd,
    0x3ff29e9df51fdee1,
    0x3ff2d285a6e4030b,
    0x3ff306fe0a31b715,
    0x3ff33c08b26416ff,
    0x3ff371a7373aa9cb,
    0x3ff3a7db34e59ff7,
    0x3ff3dea64c123422,
    0x3ff4160a21f72e2a,
    0x3ff44e086061892d,
    0x3ff486a2b5c13cd0,
    0x3ff4bfdad5362a27,
    0x3ff4f9b2769d2ca7,
    0x3ff5342b569d4f82,
    0x3ff56f4736b527da,
    0x3ff5ab07dd485429,
    0x3ff5e76f15ad2148,
    0x3ff6247eb03a5585,
    0x3ff6623882552225,
    0x3ff6a09e667f3bcd,
    0x3ff6dfb23c651a2f,
    0x3ff71f75e8ec5f74,
    0x3ff75feb564267c9,
    0x3ff7a11473eb0187,
    0x3ff7e2f336cf4e62,
    0x3ff82589994cce13,
    0x3ff868d99b4492ed,
    0x3ff8ace5422aa0db,
    0x3ff8f1ae99157736,
    0x3ff93737b0cdc5e5,
    0x3ff97d829fde4e50,
    0x3ff9c49182a3f090,
    0x3ffa0c667b5de565,
    0x3ffa5503b23e255d,
    0x3ffa9e6b5579fdbf,
    0x3ffae89f995ad3ad,
    0x3ffb33a2b84f15fb,
    0x3ffb7f76f2fb5e47,
    0x3ffbcc1e904bc1d2,
    0x3ffc199bdd85529c,
    0x3ffc67f12e57d14b,
    0x3ffcb720dcef9069,
    0x3ffd072d4a07897c,
    0x3ffd5818dcfba487,
    0x3ffda9e603db3285,
    0x3ffdfc97337b9b5f,
    0x3ffe502ee78b3ff6,
    0x3ffea4afa2a490da,
    0x3ffefa1bee615a27,
    0x3fff50765b6e4540,
    0x3fffa7c1819e90d8,
];

pub(crate) fn exp_f32(value: f32) -> f32 {
    if value.is_nan() {
        return f32::from_bits(value.to_bits() | 0x0040_0000);
    }
    let x = f64::from(value);
    let scaled = x * f64::from_bits(0x4057_1547_652b_82fe);
    if scaled >= 8192. {
        return f32::INFINITY;
    }
    if scaled < -9600. {
        return 0.;
    }
    let n = scaled.round_ties_even() as i32;
    let j = n & 63;
    let m = (n - j) >> 6;
    let r = (-f64::from(n)).mul_add(f64::from_bits(0x3f86_2e42_fefa_39ef), x);
    let q = r.mul_add(f64::from_bits(0x3fc5_5555_5555_5555), 0.5);
    let p = (r * r).mul_add(q, r);
    let table = f64::from_bits(TWO_TO_J_OVER_64[j as usize]);
    let y = table.mul_add(p, table);
    let scale = f64::from_bits(((m + 1023) as u64) << 52);
    (y * scale) as f32
}

#[cfg(test)]
mod tests {
    use super::exp_f32;

    #[test]
    fn exponential_rounding_preserves_native_erf_cancellation_operands() {
        // First differs from Linux expf; second from widened exp then F32.
        for (argument, expected) in [
            (0xbd49_cb6c, 0x3f73_b187),
            (0xbcd6_f525, 0x3f79_5eb4),
            (0xbdd7_33e9, 0x3f66_76fa),
        ] {
            assert_eq!(exp_f32(f32::from_bits(argument)).to_bits(), expected);
        }
        assert_eq!(exp_f32(0.), 1.);
        assert_eq!(exp_f32(-0.), 1.);
        assert_eq!(exp_f32(f32::NEG_INFINITY), 0.);
        assert_eq!(exp_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_f32(-105.), 0.);
        assert_eq!(exp_f32(90.), f32::INFINITY);
        assert!(exp_f32(f32::NAN).is_nan());
    }
}
