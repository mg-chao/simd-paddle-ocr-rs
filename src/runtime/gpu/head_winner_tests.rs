use super::*;

#[test]
fn parallel_head_winner_admission_is_half_logit_native_and_bounded() {
    let limits = crate::runtime::gpu::GpuLimits::default();
    for dimensions in [[32, 32], [160, 108], [144, 293], [136, 293], [32, 512]] {
        assert!(eligible(true, false, dimensions, &limits));
    }
    for dimensions in [
        [0, 108],
        [31, 108],
        [32, 0],
        [32, 31],
        [32, 513],
        [
            limits.max_compute_workgroups_per_dimension as usize + 1,
            108,
        ],
        [usize::MAX, 108],
    ] {
        assert!(!eligible(true, false, dimensions, &limits));
    }
    assert!(eligible(
        true,
        false,
        [limits.max_compute_workgroups_per_dimension as usize, 108],
        &limits
    ));
    for (half, probability) in [(false, false), (true, true)] {
        assert!(!eligible(half, probability, [32, 108], &limits));
    }
    for index in 0..3 {
        let mut limited = limits.clone();
        match index {
            0 => limited.max_compute_workgroup_size_x = 63,
            1 => limited.max_compute_invocations_per_workgroup = 63,
            _ => limited.max_compute_workgroup_storage_size = 1023,
        }
        assert!(!eligible(true, false, [32, 108], &limited));
    }
}

fn hardware() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn serial(items: &[[u32; 4]]) -> [u32; 4] {
    let mut best = u32::MAX;
    #[allow(clippy::excessive_precision)]
    let mut maximum = -3.402823e38f32;
    let mut invalid = 0;
    for item in items {
        let score = f32::from_bits(item[1]);
        invalid |= item[3];
        if score > maximum || (score == maximum && item[0] < best) {
            maximum = score;
            best = item[0];
        }
    }
    [best, maximum.to_bits(), 0, invalid]
}

#[test]
#[ignore = "requires a physical GPU; actual partial protocol, Metal compile fallback and guards"]
fn gpu_parallel_head_winner_protocol_and_pre_admission() {
    let context = hardware();
    let [rows, tiles] = [35usize, 293usize];
    let mut items: Vec<[u32; 4]> = (0..rows * tiles)
        .map(|i| {
            [
                ((i * 17) % 18710) as u32,
                (((i * 19 % 127) as f32 - 63.) / 16.).to_bits(),
                1f32.to_bits(),
                0,
            ]
        })
        .collect();
    for row in 0..5 {
        for item in &mut items[row * tiles..(row + 1) * tiles] {
            *item = [u32::MAX, (-f32::MAX).to_bits(), 1f32.to_bits(), 0];
        }
    }
    items[1] = [4, (-0f32).to_bits(), 1f32.to_bits(), 0x80];
    items[64] = [4, 0f32.to_bits(), 1f32.to_bits(), 0x40];
    items[tiles + 1] = [8, 4f32.to_bits(), 1f32.to_bits(), 0];
    items[tiles + 64] = [2, 4f32.to_bits(), 1f32.to_bits(), 1];
    items[2 * tiles + 2] = [1, f32::NAN.to_bits(), 1f32.to_bits(), 2];
    items[2 * tiles + 65] = [9, f32::INFINITY.to_bits(), 1f32.to_bits(), 4];
    items[3 * tiles + 5] = [1, f32::NEG_INFINITY.to_bits(), 1f32.to_bits(), 8];
    let expected: Vec<[u32; 4]> = items.chunks_exact(tiles).map(serial).collect();
    let input = context
        .create_buffer_init(
            "head partial protocol",
            bytemuck::cast_slice(&items),
            BufferUsage::STORAGE,
        )
        .unwrap();
    let candidate = kernel(&context, true, false, [rows, tiles])
        .unwrap()
        .expect("native reducer must be admitted");
    let baseline = KernelKey::new(KernelId::HeadReduce);
    let sentinel = [0xdeadbeefu32; 4];
    for parallel in [false, true] {
        let mut initial = vec![sentinel; rows + 3];
        let output = context
            .create_buffer_init(
                "head protocol and suffix guards",
                bytemuck::cast_slice(&initial),
                BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let words = [rows as u32, 1, 18710, tiles as u32, 0];
        if parallel {
            batch
                .dispatch_prepared(
                    NAME,
                    &candidate,
                    &[(&input, true), (&output, false)],
                    &words,
                    [rows as u32 + 3, 1, 1],
                )
                .unwrap();
        } else {
            batch
                .dispatch(
                    "projection_winners",
                    baseline,
                    &[(&input, true), (&output, false)],
                    &words,
                    [(rows as u32).div_ceil(64), 1, 1],
                )
                .unwrap();
        }
        let (actual, stats) = batch.read(&output, initial.len() * 16, false).unwrap();
        initial[..rows].copy_from_slice(&expected);
        assert_eq!(actual, bytemuck::cast_slice::<[u32; 4], u8>(&initial));
        assert_eq!(stats.dispatches, 1);
        assert_eq!(stats.native_resident_dispatches, 1);
        assert_eq!(stats.diagnostic_readback_bytes, 0);
    }
    {
        let kind = ErrorKind::Unsupported;
        assert!(
            kernel_with(&context, true, false, [rows, tiles], |_, _, _| Err(
                RuntimeError::with_kind(kind, "scoped compiler rejection")
            ))
            .unwrap()
            .is_none()
        );
    }
    let invalid = kernel_with(&context, true, false, [rows, tiles], |_, _, _| {
        Err(RuntimeError::with_kind(
            ErrorKind::GpuValidation,
            "invalid shader interface",
        ))
    })
    .err()
    .unwrap();
    assert_eq!(invalid.kind(), ErrorKind::GpuValidation);
    let error = kernel_with(&context, true, false, [rows, tiles], |_, _, _| {
        Err(RuntimeError::with_kind(
            ErrorKind::Allocation,
            "fatal allocation",
        ))
    })
    .err()
    .unwrap();
    assert_eq!(error.kind(), ErrorKind::Allocation);
    assert!(
        kernel_with(&context, true, true, [rows, tiles], |_, _, _| panic!(
            "Probability must not compile the candidate"
        ))
        .unwrap()
        .is_none()
    );
    assert!(
        kernel_with(&context, false, false, [rows, tiles], |_, _, _| panic!(
            "strict head must not compile the candidate"
        ))
        .unwrap()
        .is_none()
    );
}
