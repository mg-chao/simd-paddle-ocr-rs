//! Preserve strict geometry and isolate the explicit Half/native-trig policy.
use super::*;

#[test]
fn native_trig_requires_explicit_half_matrix_native_and_bounded_geometry() {
    let dims = [512, 512, 500, 500, 500, 500];
    let eligible = native_trig_eligible;
    assert!(eligible(
        GpuMatrixPrecision::F16,
        GpuPrecision::MixedF16,
        dims,
        1.4
    ));
    for matrix in [GpuMatrixPrecision::F32, GpuMatrixPrecision::F16] {
        for storage in [GpuPrecision::F32, GpuPrecision::MixedF16] {
            assert_eq!(
                eligible(matrix, storage, dims, 1.4),
                matrix == GpuMatrixPrecision::F16 && storage == GpuPrecision::MixedF16
            );
        }
    }
    for axis in 0..6 {
        for value in [0, 4097, usize::MAX] {
            let mut bad = dims;
            bad[axis] = value;
            assert!(!eligible(
                GpuMatrixPrecision::F16,
                GpuPrecision::MixedF16,
                bad,
                1.4
            ));
        }
    }
    for ratio in [0., 0.249, 2.001, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!eligible(
            GpuMatrixPrecision::F16,
            GpuPrecision::MixedF16,
            dims,
            ratio
        ));
    }
    for ratio in [0.25, 2.] {
        assert!(eligible(
            GpuMatrixPrecision::F16,
            GpuPrecision::MixedF16,
            [4096; 6],
            ratio
        ));
    }
}

fn hardware() -> Arc<GpuContext> {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    assert!(matches!(
        context.info.device_type,
        crate::runtime::gpu::GpuDeviceType::DiscreteGpu
            | crate::runtime::gpu::GpuDeviceType::IntegratedGpu
    ));
    context
}

struct Fixture {
    w: usize,
    h: usize,
    ratio: f32,
    score: f32,
    threshold: f32,
    rows: Vec<Vec<(u32, u32, u32)>>,
    accepted: Vec<bool>,
}

fn rect(x: u32, y: u32, w: u32, h: u32) -> Vec<(u32, u32, u32)> {
    (y..=y + h).map(|row| (row, x, x + w)).collect()
}

fn polygon(points: &[[i64; 2]; 4]) -> Vec<(u32, u32, u32)> {
    let (left, right) = (
        points.iter().map(|p| p[0]).min().unwrap(),
        points.iter().map(|p| p[0]).max().unwrap(),
    );
    let (top, bottom) = (
        points.iter().map(|p| p[1]).min().unwrap(),
        points.iter().map(|p| p[1]).max().unwrap(),
    );
    (top..=bottom)
        .filter_map(|y| {
            let inside: Vec<_> = (left..=right)
                .filter(|&x| {
                    (0..4).all(|i| {
                        let (a, b) = (points[i], points[(i + 1) % 4]);
                        (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= 0
                    })
                })
                .collect();
            Some((y as u32, *inside.first()? as u32, *inside.last()? as u32))
        })
        .collect()
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            w: 128,
            h: 128,
            ratio: 0.25,
            score: 0.5,
            threshold: 0.5,
            rows: vec![
                rect(10, 10, 24, 8),
                rect(45, 10, 20, 2),
                rect(10, 40, 20, 3),
                rect(45, 40, 20, 6),
            ],
            accepted: vec![true, false, false, true],
        },
        Fixture {
            w: 192,
            h: 192,
            ratio: 1.4,
            score: 0.75,
            threshold: 0.5,
            rows: vec![
                polygon(&[[40, 25], [60, 45], [54, 51], [34, 31]]),
                polygon(&[[100, 80], [160, 83], [159, 90], [99, 87]]),
                rect(0, 120, 24, 8),
            ],
            accepted: vec![true, true, true],
        },
        Fixture {
            w: 4096,
            h: 4096,
            ratio: 1.4,
            score: 0.75,
            threshold: 0.5,
            rows: vec![
                polygon(&[[3900, 3900], [3920, 3910], [3918, 3914], [3898, 3904]]),
                polygon(&[[4000, 3900], [4020, 3910], [4019, 3912], [3999, 3902]]),
                polygon(&[[3900, 4000], [3920, 4010], [3918, 4013], [3898, 4003]]),
            ],
            accepted: vec![true, false, true],
        },
        Fixture {
            w: 128,
            h: 128,
            ratio: 0.25,
            score: 0.75,
            threshold: 0.5,
            rows: vec![
                rect(32, 32, 8, 8),
                rect(64, 64, 8, 8),
                polygon(&[[80, 20], [88, 28], [84, 32], [76, 24]]),
            ],
            accepted: vec![true, true, true],
        },
        Fixture {
            w: 64,
            h: 64,
            ratio: 2.,
            score: 0.5,
            threshold: f32::from_bits(0x3f000001),
            rows: vec![rect(8, 8, 20, 8), rect(32, 32, 20, 12)],
            accepted: vec![false, false],
        },
        Fixture {
            w: 64,
            h: 64,
            ratio: 2.,
            score: 0.75,
            threshold: 0.5,
            rows: vec![],
            accepted: vec![],
        },
    ]
}

fn geometry_words(f: &Fixture) -> Vec<u32> {
    vec![
        f.w as u32,
        f.h as u32,
        f.threshold.to_bits(),
        f.ratio.to_bits(),
        f.w as u32,
        f.h as u32,
        f.w as u32,
        f.h as u32,
    ]
}

fn run_geometry(context: &Arc<GpuContext>, f: &Fixture, native: bool) -> Vec<u32> {
    const POISON: u32 = 0x7fc12345;
    let mut metadata = vec![0u32; 16 + f.rows.len() * 8];
    metadata[1] = f.rows.len() as u32;
    let mut rows = Vec::new();
    for (i, component) in f.rows.iter().enumerate() {
        let top = component.first().unwrap().0;
        let bottom = component.last().unwrap().0;
        let at = 16 + i * 8;
        metadata[at + 2] = top;
        metadata[at + 4] = bottom;
        metadata[at + 5] = (rows.len() / 2) as u32;
        for y in top..=bottom {
            let row = component
                .iter()
                .find(|p| p.0 == y)
                .map_or((f.w as u32, 0), |p| (p.1, p.2));
            rows.extend([row.0, row.1]);
        }
    }
    metadata[3] = (rows.len() / 2) as u32;
    let scratch_count = rows.len() / 2 * 4 + f.rows.len() * 1024;
    if rows.is_empty() {
        rows.extend([f.w as u32, 0]);
    }
    let prediction = vec![half::f16::from_f32(f.score).to_bits(); f.w * f.h];
    let init = |label, bytes: &[u8], copy_source: bool| {
        context
            .create_buffer_init(
                label,
                bytes,
                BufferUsage::STORAGE
                    | if copy_source {
                        BufferUsage::COPY_SRC
                    } else {
                        BufferUsage::empty()
                    },
            )
            .unwrap()
    };
    let prediction = init(
        "geometry Half map",
        bytemuck::cast_slice(&prediction),
        false,
    );
    let metadata = init("geometry metadata", bytemuck::cast_slice(&metadata), false);
    let rows = init("geometry row extrema", bytemuck::cast_slice(&rows), false);
    let scratch = init(
        "geometry poisoned scratch",
        bytemuck::cast_slice(&vec![POISON; (scratch_count.max(1) + 8) * 2]),
        true,
    );
    let output_words = f.rows.len().max(1) * 16;
    let output = init(
        "geometry poisoned boxes",
        bytemuck::cast_slice(&vec![POISON; output_words + 11]),
        true,
    );
    let (name, kernel) = geometry_kernel(context, GpuPrecision::MixedF16, native).unwrap();
    assert_eq!(name.contains("native_trig"), native);
    let mut batch = Batch::new(context.clone()).unwrap();
    batch
        .dispatch_prepared(
            &name,
            &kernel,
            &[
                (&prediction, true),
                (&metadata, true),
                (&rows, true),
                (&scratch, false),
                (&output, false),
            ],
            &geometry_words(f),
            [f.rows.len().max(1) as u32, 1, 1],
        )
        .unwrap();
    let readback = batch
        .storage("geometry guarded readback", output.bytes() + 64)
        .unwrap();
    batch.copy(&output, 0, &readback, 0, output.bytes() as u64);
    batch.copy(
        &scratch,
        (scratch_count.max(1) * 8) as u64,
        &readback,
        output.bytes() as u64,
        64,
    );
    let (bytes, stats) = batch.read(&readback, readback.bytes(), false).unwrap();
    assert_eq!(stats.dispatches, 1);
    assert!(stats.kernels.contains(&name));
    let words: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    assert!(words[output_words..].iter().all(|&v| v == POISON));
    if f.rows.is_empty() {
        assert!(words.iter().all(|&v| v == POISON));
    }
    for (i, &accepted) in f.accepted.iter().enumerate() {
        assert_eq!(words[i * 16 + 9], u32::from(accepted), "component {i}");
        assert!(words[i * 16 + 10..i * 16 + 16].iter().all(|&v| v == POISON));
        if accepted {
            assert_eq!(f32::from_bits(words[i * 16 + 8]), f.score);
            assert!(
                words[i * 16..i * 16 + 9]
                    .iter()
                    .all(|&v| f32::from_bits(v).is_finite())
            );
        }
    }
    words[..output_words].to_vec()
}

fn polygon_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.;
    }
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
        .sum::<f64>()
        * 0.5
}

fn assert_box_overlap(a: &[[f32; 2]; 4], b: &[[f32; 2]; 4]) {
    let matching = (0..4)
        .flat_map(|offset| {
            [false, true].map(move |reverse| {
                (0..4)
                    .flat_map(|i| {
                        let j = if reverse {
                            (offset + 4 - i) % 4
                        } else {
                            (offset + i) % 4
                        };
                        (0..2).map(move |axis| (a[i][axis] - b[j][axis]).abs())
                    })
                    .fold(0., f32::max)
            })
        })
        .fold(f32::INFINITY, f32::min);
    assert!(matching <= 2., "coordinate displacement {matching}");
    let to_double = |p: &[[f32; 2]; 4]| p.map(|q| q.map(f64::from)).to_vec();
    let (subject, clip) = (to_double(a), to_double(b));
    let cross = |p: [f64; 2], q: [f64; 2], v: [f64; 2]| {
        (q[0] - p[0]) * (v[1] - p[1]) - (q[1] - p[1]) * (v[0] - p[0])
    };
    let direction = polygon_area(&clip).signum();
    let mut intersection = subject.clone();
    for edge in 0..4 {
        if intersection.is_empty() {
            break;
        }
        let input = std::mem::take(&mut intersection);
        let (p, q) = (clip[edge], clip[(edge + 1) % 4]);
        let mut previous = *input.last().unwrap();
        let mut pv = cross(p, q, previous) * direction;
        for next in input {
            let nv = cross(p, q, next) * direction;
            if (pv >= 0.) != (nv >= 0.) {
                let t = pv / (pv - nv);
                intersection.push([
                    previous[0] + t * (next[0] - previous[0]),
                    previous[1] + t * (next[1] - previous[1]),
                ]);
            }
            if nv >= 0. {
                intersection.push(next);
            }
            previous = next;
            pv = nv;
        }
    }
    let area = polygon_area(&intersection).abs();
    let union = polygon_area(&subject).abs() + polygon_area(&clip).abs() - area;
    assert!(
        union > 0. && area / union >= 0.8,
        "polygon IoU {}",
        area / union
    );
}

#[test]
#[ignore = "requires a physical GPU; Half metadata, high-origin tiny rotations, offset ties and count/scratch guards"]
fn gpu_native_trig_geometry_metadata_ties_and_boundary_guards() {
    let context = hardware();
    for fixture in fixtures() {
        let baseline = run_geometry(&context, &fixture, false);
        let candidate = run_geometry(&context, &fixture, true);
        for (i, &accepted) in fixture.accepted.iter().enumerate() {
            assert_eq!(baseline[i * 16 + 9], candidate[i * 16 + 9]);
            if accepted {
                assert_eq!(baseline[i * 16 + 8], candidate[i * 16 + 8]);
                let points = |words: &[u32]| {
                    std::array::from_fn(|corner| {
                        [
                            f32::from_bits(words[i * 16 + corner * 2]),
                            f32::from_bits(words[i * 16 + corner * 2 + 1]),
                        ]
                    })
                };
                assert_box_overlap(&points(&baseline), &points(&candidate));
            }
        }
    }
}

#[test]
#[ignore = "requires a physical GPU; policy, pre-admission compilation fallback and prepared validation errors"]
fn gpu_native_trig_policy_compile_fallback_and_prepared_validation() {
    let context = hardware();
    let mut calls = Vec::new();
    let (name, _) = geometry_kernel_with(
        &context,
        GpuPrecision::MixedF16,
        true,
        |ctx, key, layout| {
            calls.push(key);
            if key.id == KernelId::DbGeometryFast {
                Err(RuntimeError::with_kind(
                    ErrorKind::Unsupported,
                    "unsupported candidate capability",
                ))
            } else {
                Kernel::get(ctx, key, layout)
            }
        },
    )
    .unwrap();
    assert_eq!(name, "db_geometry_MixedF16");
    assert_eq!(calls.len(), 2);
    let mut calls = 0;
    let error = geometry_kernel_with(&context, GpuPrecision::MixedF16, true, |_, _, _| {
        calls += 1;
        Err(RuntimeError::with_kind(
            ErrorKind::Allocation,
            "simulated fatal allocation failure",
        ))
    })
    .err()
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(error.kind(), ErrorKind::Allocation);
    let invalid = geometry_kernel_with(&context, GpuPrecision::MixedF16, true, |_, _, _| {
        Err(RuntimeError::with_kind(
            ErrorKind::GpuValidation,
            "invalid shader interface",
        ))
    })
    .err()
    .unwrap();
    assert_eq!(invalid.kind(), ErrorKind::GpuValidation);
    let (_, kernel) = geometry_kernel(&context, GpuPrecision::MixedF16, true).unwrap();
    let dummy = context
        .create_buffer_init(
            "prepared rejected binding",
            &[0u8; 16],
            BufferUsage::STORAGE,
        )
        .unwrap();
    let mut batch = Batch::new(context.clone()).unwrap();
    let valid = [
        (&dummy, true),
        (&dummy, true),
        (&dummy, true),
        (&dummy, false),
        (&dummy, false),
    ];
    let error = batch
        .dispatch_prepared(
            "invalid group",
            &kernel,
            &valid,
            &[],
            [
                context.limits.max_compute_workgroups_per_dimension + 1,
                1,
                1,
            ],
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    let wrong = [
        (&dummy, false),
        (&dummy, true),
        (&dummy, true),
        (&dummy, false),
        (&dummy, false),
    ];
    let error = batch
        .dispatch_prepared("invalid layout", &kernel, &wrong, &[], [1, 1, 1])
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(batch.statistics.dispatches, 0);
    assert_eq!(batch.statistics.native_resident_dispatches, 0);

    let other_context = hardware();
    let other_buffer = other_context
        .create_buffer_init("foreign prepared binding", &[0u8; 16], BufferUsage::STORAGE)
        .unwrap();
    let other_bindings = [
        (&other_buffer, true),
        (&other_buffer, true),
        (&other_buffer, true),
        (&other_buffer, false),
        (&other_buffer, false),
    ];
    let mut foreign_batch = Batch::new(other_context.clone()).unwrap();
    let error = foreign_batch
        .dispatch_prepared("foreign kernel", &kernel, &other_bindings, &[], [1, 1, 1])
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(foreign_batch.statistics.dispatches, 0);
    assert_eq!(foreign_batch.statistics.native_resident_dispatches, 0);

    for context in [context.clone(), other_context.clone()] {
        for (matrix, precision) in [
            (GpuMatrixPrecision::F32, GpuPrecision::MixedF16),
            (GpuMatrixPrecision::F16, GpuPrecision::F32),
            (GpuMatrixPrecision::F16, GpuPrecision::MixedF16),
        ] {
            let mut map = vec![0.0; 96 * 96];
            for y in 10..25 {
                for x in 12..70 {
                    map[y * 96 + x] = 0.75;
                }
            }
            let tensor = Tensor::allocate(
                context.clone(),
                Shape::new(&[1, 1, 96, 96]).unwrap(),
                precision,
            )
            .unwrap();
            if precision == GpuPrecision::MixedF16 {
                let half: Vec<_> = map
                    .iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
                    .collect();
                context
                    .upload_resident(&tensor.buffer, bytemuck::cast_slice(&half))
                    .unwrap();
            } else {
                context
                    .upload_resident(&tensor.buffer, bytemuck::cast_slice(&map))
                    .unwrap();
            }
            let mut previous: Option<Vec<DetectionBox>> = None;
            for profiling in [false, true] {
                let mut batch = Batch::new(context.clone()).unwrap();
                batch.profiling = profiling;
                let (boxes, _, stats, _) = postprocess_with_math(
                    batch,
                    &tensor,
                    DetectorOptions {
                        max_candidates: 100,
                        box_threshold: Some(0.5),
                        ..Default::default()
                    },
                    DetSourceGeometry::new(96, 96).unwrap(),
                    false,
                    matrix,
                )
                .unwrap();
                assert_eq!(boxes.len(), 1);
                assert_eq!(
                    stats
                        .kernels
                        .iter()
                        .any(|name| name.contains("native_trig")),
                    matrix == GpuMatrixPrecision::F16 && precision == GpuPrecision::MixedF16
                );
                if let Some(previous) = &previous {
                    assert_eq!(boxes[0].score.to_bits(), previous[0].score.to_bits());
                    assert_box_overlap(&boxes[0].points, &previous[0].points);
                }
                previous = Some(boxes);
            }
        }
    }
}
