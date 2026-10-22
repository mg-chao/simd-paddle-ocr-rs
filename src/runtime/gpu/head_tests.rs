use super::*;
use std::sync::{Arc, OnceLock};

fn spec(mode: Mode, inner: usize, classes: usize, bias: bool) -> HeadSpec {
    HeadSpec {
        projection: ProjectionHead {
            feature: TensorId(0),
            feature_node: 0,
            weights: TensorId(1),
            bias: bias.then_some(TensorId(2)),
            inner,
            classes,
        },
        mode,
        node_count: 1,
        source: TensorId(0),
    }
}

#[test]
fn compact_head_shape_and_dispatch_checks_reject_invalid_domains() {
    let logit = spec(Mode::Logit, 3, 65, false);
    assert_eq!(logit.rows(Shape::new(&[2, 7, 3]).unwrap()).unwrap(), 14);
    assert_eq!(
        logit
            .memory_requirements(Shape::new(&[2, 7, 3]).unwrap(), &wgpu::Limits::default())
            .unwrap(),
        (14 * 16, (14 * 2 + 14) * 16),
    );
    assert!(logit.rows(Shape::new(&[2, 7, 65]).unwrap()).is_err());
    assert!(logit.rows(Shape::new(&[14, 3]).unwrap()).is_err());
    let probability = spec(Mode::Probability, 3, 65, false);
    assert_eq!(
        probability.rows(Shape::new(&[2, 7, 65]).unwrap()).unwrap(),
        14
    );
    let classification = spec(Mode::Classification, 3, 2, true);
    assert_eq!(
        classification.rows(Shape::new(&[2, 3]).unwrap()).unwrap(),
        2
    );
    assert_eq!(
        classification
            .memory_requirements(Shape::new(&[2, 3]).unwrap(), &wgpu::Limits::default())
            .unwrap(),
        (2 * 16, 3 * 16),
    );
    assert!(
        classification
            .rows(Shape::new(&[1, 2, 3]).unwrap())
            .is_err()
    );
    let limits = wgpu::Limits {
        max_compute_workgroups_per_dimension: 3,
        ..Default::default()
    };
    assert_eq!(groups(8, &limits).unwrap(), (3, 3));
    assert!(groups(10, &limits).is_err());
    assert!(groups(0, &limits).is_err());
    assert!(groups(usize::MAX, &limits).is_err());
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 64,
        ..Default::default()
    };
    assert_eq!(record_bytes(4, &limits).unwrap(), 64);
    assert!(record_bytes(5, &limits).is_err());
    assert!(record_bytes(usize::MAX, &limits).is_err());
}

#[test]
fn compact_readback_requires_all_candidate_validity_and_valid_records() {
    let logit = spec(Mode::Logit, 3, 65, false);
    let record = |index: u32, score: f32, valid: u32| {
        [index, score.to_bits(), valid, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut output = vec![CompactRow {
        index: 9,
        score: 12.,
    }];
    decode_records(&logit, 1, &record(64, -2., 1), &mut output).unwrap();
    assert_eq!(
        output,
        [CompactRow {
            index: 64,
            score: -2.
        }]
    );
    for bytes in [
        record(64, 2., 0),
        record(65, 2., 1),
        record(0, f32::NAN, 1),
        vec![0; 15],
    ] {
        assert!(decode_records(&logit, 1, &bytes, &mut output).is_err());
        assert_eq!(
            output,
            [CompactRow {
                index: 64,
                score: -2.
            }]
        );
    }
    let probability = spec(Mode::Probability, 3, 65, false);
    assert!(decode_records(&probability, 1, &record(0, -0.1, 1), &mut output).is_err());
    assert!(decode_records(&probability, 1, &record(0, 1.1, 1), &mut output).is_err());
    decode_records(&probability, 1, &record(0, 0.5, 1), &mut output).unwrap();
    assert_eq!(output[0].score, 0.5);
}

fn hardware() -> Arc<GpuContext> {
    static CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let backend = match std::env::var("SIMD_OCR_GPU_BACKEND")
                .as_deref()
                .unwrap_or("vulkan")
            {
                "auto" => super::super::GpuBackendPreference::Auto,
                "vulkan" => super::super::GpuBackendPreference::Vulkan,
                "dx12" => super::super::GpuBackendPreference::Dx12,
                "metal" => super::super::GpuBackendPreference::Metal,
                other => panic!("invalid SIMD_OCR_GPU_BACKEND: {other}"),
            };
            let context = GpuContext::new_with_options(super::super::GpuContextOptions {
                backend,
                ..Default::default()
            })
            .expect("compact head checks require physical GPU hardware");
            let info = context.adapter_info();
            eprintln!(
                "GPU compact head adapter: {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            );
            assert!(matches!(
                info.device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            ));
            context
        })
        .clone()
}

fn run(
    mode: Mode,
    rows: usize,
    inner: usize,
    classes: usize,
    source: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
) -> RuntimeResult<Vec<CompactRow>> {
    let context = hardware();
    let _gate = context.acquire_gate()?;
    context.checked("GPU compact head fixture", || {
        let source_buffer = context.create_buffer_init(
            "head test source",
            bytemuck::cast_slice(source),
            wgpu::BufferUsages::STORAGE,
        )?;
        let weight_buffer = context.create_buffer_init(
            "head test weights",
            bytemuck::cast_slice(weights),
            wgpu::BufferUsages::STORAGE,
        )?;
        let bias_buffer = context.create_buffer_init(
            "head test bias",
            bytemuck::cast_slice(bias.unwrap_or(&[0.])),
            wgpu::BufferUsages::STORAGE,
        )?;
        let pipelines = HeadPipelines::new(&context.device);
        let spec = spec(mode, inner, classes, bias.is_some());
        let columns = if mode == Mode::Probability {
            classes
        } else {
            inner
        };
        let shape = if mode == Mode::Classification {
            Shape::new(&[rows, columns])?
        } else {
            Shape::new(&[1, rows, columns])?
        };
        let plan = HeadPlan::new(&spec, shape, &context, &pipelines, |id| match id.0 {
            0 => &source_buffer,
            1 => &weight_buffer,
            2 => &bias_buffer,
            _ => unreachable!(),
        })?;
        let readback = context.create_buffer(
            "head test readback",
            plan.output_bytes() as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )?;
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU compact head fixture"),
            });
        plan.encode(&mut encoder, &pipelines);
        encoder.copy_buffer_to_buffer(
            plan.output_buffer(),
            0,
            &readback,
            0,
            plan.output_bytes() as u64,
        );
        context.submit_and_map(encoder, &readback, plan.output_bytes() as u64, |bytes| {
            let mut output = Vec::new();
            plan.decode(bytes, &mut output)?;
            Ok(output)
        })
    })
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_compact_logit_tiles_preserve_bias_order_and_cross_tile_ties() {
    // Bias-first produces [0, .5]; MatMul then Add would produce [1, .5].
    let result = run(
        Mode::Logit,
        1,
        3,
        2,
        &[1., 1., 1.],
        &[1e20, 0., 1., 0., -1e20, 0.],
        Some(&[1., 0.5]),
    )
    .unwrap();
    assert_eq!(
        result,
        [CompactRow {
            index: 1,
            score: 0.5
        }]
    );
    let mut weights = vec![0.; 65];
    weights[0] = 2.;
    weights[64] = 2.;
    let result = run(Mode::Logit, 2, 1, 65, &[1., -1.], &weights, None).unwrap();
    assert_eq!(
        result[0],
        CompactRow {
            index: 0,
            score: 2.
        }
    );
    assert_eq!(
        result[1],
        CompactRow {
            index: 1,
            score: 0.
        }
    );
    // Real dictionary class count exercises an incomplete final tile.
    let mut weights = vec![0.; 18710];
    weights[18709] = 3.;
    let result = run(Mode::Logit, 1, 1, 18710, &[2.], &weights, None).unwrap();
    assert_eq!(
        result,
        [CompactRow {
            index: 18709,
            score: 6.
        }]
    );
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_compact_probability_uses_rounded_values_and_lowest_tie() {
    let mut probabilities = vec![0.; 2 * 65];
    probabilities[0] = 0.5;
    probabilities[64] = 0.5;
    probabilities[65 + 64] = 0.75;
    let result = run(
        Mode::Probability,
        2,
        1,
        65,
        &probabilities,
        &vec![0.; 65],
        None,
    )
    .unwrap();
    assert_eq!(
        result,
        [
            CompactRow {
                index: 0,
                score: 0.5
            },
            CompactRow {
                index: 64,
                score: 0.75
            }
        ]
    );
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_compact_classification_uses_logit_winner_and_probability_score() {
    let result = run(
        Mode::Classification,
        1,
        1,
        2,
        &[0.],
        &[0., 0.],
        Some(&[0., 1e-8]),
    )
    .unwrap();
    assert_eq!(
        result,
        [CompactRow {
            index: 1,
            score: 0.5
        }]
    );
    let result = run(
        Mode::Classification,
        1,
        3,
        2,
        &[1., 1., 1.],
        &[1e20, 0., 1., 0., -1e20, 0.],
        Some(&[1., 0.5]),
    )
    .unwrap();
    assert_eq!(result[0].index, 1);
    let expected = 1. / (1. + crate::math::exp_f32(-0.5));
    assert!((result[0].score - expected).abs() <= 1e-4);
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_compact_heads_reject_nonwinning_nonfinite_candidates() {
    let mut weights = vec![0.; 65];
    weights[0] = 1.;
    weights[64] = -f32::MAX;
    assert!(run(Mode::Logit, 1, 1, 65, &[2.], &weights, None).is_err());
    let mut probabilities = vec![0.; 65];
    probabilities[0] = 0.7;
    probabilities[64] = f32::NAN;
    assert!(
        run(
            Mode::Probability,
            1,
            1,
            65,
            &probabilities,
            &vec![0.; 65],
            None
        )
        .is_err()
    );
    assert!(run(Mode::Classification, 1, 1, 2, &[2.], &[-f32::MAX, 1.], None).is_err());
}
