//! GPU regression for preserved half store boundaries.
use crate::runtime::gpu::resident::Batch;
use crate::runtime::gpu::{BufferUsage, GpuContext, GpuContextOptions, KernelId, KernelKey};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy)]
struct Case {
    bits: u32,
    category: &'static str,
}

fn signed(cases: &mut Vec<Case>, magnitude: u32, category: &'static str) {
    for sign in [0, 0x8000_0000] {
        cases.push(Case {
            bits: sign | magnitude,
            category,
        });
    }
}

fn midpoint(cases: &mut Vec<Case>, lower: u16) {
    let a = half::f16::from_bits(lower).to_f32();
    let b = half::f16::from_bits(lower + 1).to_f32();
    let middle = ((a + b) * 0.5).to_bits();
    for bits in [middle - 1, middle, middle + 1] {
        signed(cases, bits, "half midpoint and adjacent F32 values");
    }
}

fn fixture() -> Vec<Case> {
    let mut cases = Vec::new();
    for magnitude in [
        0x0000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x32ff_ffff,
        0x3300_0000,
        0x3300_0001,
        0x3380_0000,
        0x387f_ffff,
        0x3880_0000,
        0x3880_0001,
        0x3f80_0000,
        0x477f_dfff,
        0x477f_e000,
        0x477f_e001,
        0x477f_efff,
        0x477f_f000,
        0x477f_f001,
        0x4780_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x7fc0_0000,
        0x7fff_ffff,
    ] {
        signed(
            &mut cases,
            magnitude,
            "zero/subnormal/normal/overflow/IEEE special",
        );
    }
    // Every finite half exponent, even/odd retained mantissas, and positive
    // and negative tie neighbors. These include underflow to signed zero.
    for exponent in 0..31u16 {
        for fraction in [0, 1, 0x01ff, 0x0200, 0x03fe] {
            midpoint(&mut cases, (exponent << 10) | fraction);
        }
    }
    for lower in [0x03ff, 0x07ff, 0x3bff, 0x3fff, 0x77ff] {
        midpoint(&mut cases, lower);
    }
    // A deliberately partial final workgroup also checks the store guard.
    const COUNT: usize = 1153;
    assert!(cases.len() < COUNT);
    let mut state = 0x9e37_79b9u32;
    while cases.len() < COUNT {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        cases.push(Case {
            bits: state,
            category: "deterministic random F32 bits",
        });
    }
    assert!(cases.iter().any(|case| case.bits == 0x3f80_1000));
    assert!(cases.iter().any(|case| case.bits == 0x3f80_3000));
    cases
}

fn context() -> Arc<GpuContext> {
    static CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| GpuContext::new_with_options(GpuContextOptions::default()).unwrap())
        .clone()
}

fn check() {
    let context = context();
    let cases = fixture();
    let input: Vec<_> = cases.iter().map(|case| case.bits).collect();
    let prefix = 3usize;
    let suffix = 5usize;
    let sentinel = 0x4b7f_ffffu32;
    let initial = vec![sentinel; prefix + cases.len() + suffix];
    let source = context
        .create_buffer_init(
            "rounding edge F32 input",
            bytemuck::cast_slice(&input),
            BufferUsage::STORAGE,
        )
        .unwrap();
    let output = context
        .create_buffer_init(
            "rounding edge F32 output with sentinels",
            bytemuck::cast_slice(&initial),
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )
        .unwrap();
    let words = [cases.len() as u32, prefix as u32];
    let groups = [cases.len().div_ceil(64) as u32, 1, 1];
    let mut batch = Batch::new(context).unwrap();
    let name = "round_f16_edges";
    batch
        .dispatch(
            name,
            KernelKey::new(KernelId::RoundHalfProbe),
            &[(&source, true), (&output, false)],
            &words,
            groups,
        )
        .unwrap();
    let (bytes, statistics) = batch.read(&output, initial.len() * 4, false).unwrap();
    assert_eq!(statistics.dispatches, 1);
    assert!(statistics.kernels.iter().any(|kernel| kernel == name));
    let actual: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(&actual[..prefix], &initial[..prefix]);
    assert_eq!(
        &actual[prefix + cases.len()..],
        &initial[prefix + cases.len()..]
    );
    for (index, case) in cases.iter().enumerate() {
        let expected = half::f16::from_f32(f32::from_bits(case.bits)).to_f32();
        let bits = actual[prefix + index];
        if expected.is_nan() {
            // Both helpers intentionally preserve the original NaN payload;
            // f16 conversion may truncate it and quiet a signaling NaN. IEEE
            // classification and sign are the contract, not payload equality.
            assert!(
                f32::from_bits(bits).is_nan(),
                "{name}[{index}]: NaN became 0x{bits:08x}"
            );
            assert_eq!(
                bits >> 31,
                expected.to_bits() >> 31,
                "{name}[{index}]: NaN sign"
            );
        } else {
            assert_eq!(
                bits,
                expected.to_bits(),
                "{name}[{index}]: {} input=0x{:08x}",
                case.category,
                case.bits
            );
        }
    }
}

#[test]
#[ignore = "requires a physical GPU; 1153 IEEE conversion cases, one native dispatch"]
fn gpu_round_f16_ieee_edges_match_independent_oracle() {
    check();
}

#[test]
#[ignore = "requires a physical GPU; 1153 actual half stores with independent IEEE oracle"]
fn gpu_round_f16_tensor_stores_match_independent_oracle() {
    use crate::runtime::gpu::GpuPrecision;
    let context = context();
    let cases = fixture();
    let input: Vec<_> = cases.iter().map(|case| case.bits).collect();
    let source = context
        .create_buffer_init(
            "rounding input",
            bytemuck::cast_slice(&input),
            BufferUsage::STORAGE,
        )
        .unwrap();
    let prefix = 3;
    let initial = vec![-777.0; prefix + cases.len() + 5];
    let bytes = GpuPrecision::MixedF16.encode(&initial);
    let output = context
        .create_buffer_init(
            "guarded half stores",
            &bytes,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )
        .unwrap();
    let shader = KernelKey {
        id: KernelId::HalfStoreProbe,
        precision: 32,
    };
    let mut words = [0; 128];
    words[0] = cases.len() as u32;
    words[60] = prefix as u32;
    let mut batch = Batch::new(context).unwrap();
    batch
        .dispatch(
            "half_tensor_stores",
            shader,
            &[
                (&source, true),
                (&source, true),
                (&source, true),
                (&source, true),
                (&source, true),
                (&output, false),
            ],
            &words,
            [cases.len().div_ceil(64) as u32, 1, 1],
        )
        .unwrap();
    let (bytes, _) = batch.read(&output, bytes.len(), false).unwrap();
    let mut actual = initial.clone();
    GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
    assert_eq!(&actual[..prefix], &initial[..prefix]);
    assert_eq!(
        &actual[prefix + cases.len()..],
        &initial[prefix + cases.len()..]
    );
    for (i, case) in cases.iter().enumerate() {
        let expected = half::f16::from_f32(f32::from_bits(case.bits)).to_f32();
        let actual = actual[prefix + i];
        if expected.is_nan() {
            assert!(actual.is_nan(), "half store {i}: {}", case.category);
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "half store {i}: {} input={:08x}",
                case.category,
                case.bits
            );
        }
    }
}
