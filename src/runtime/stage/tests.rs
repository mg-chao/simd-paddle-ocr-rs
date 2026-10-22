use super::*;
use crate::{
    image::{ImageView, PixelFormat},
    model::{LoadOptions, OptimizationPolicy},
    orientation::{Classifier, ClassifierTrace},
    recognizer::{
        Dictionary, RecognitionScoreMode, RecognitionTrace, Recognizer, RecognizerOptions,
    },
    runtime::gpu::{GpuBackendPreference, GpuCompiledModel, GpuContext, GpuContextOptions},
};

fn hardware() -> Arc<GpuContext> {
    static CONTEXT: std::sync::OnceLock<Arc<GpuContext>> = std::sync::OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let backend = match std::env::var("SIMD_OCR_GPU_BACKEND")
                .as_deref()
                .unwrap_or("vulkan")
            {
                "auto" => GpuBackendPreference::Auto,
                "vulkan" => GpuBackendPreference::Vulkan,
                "dx12" => GpuBackendPreference::Dx12,
                "metal" => GpuBackendPreference::Metal,
                other => panic!("invalid SIMD_OCR_GPU_BACKEND: {other}"),
            };
            let context = GpuContext::new_with_options(GpuContextOptions {
                backend,
                ..Default::default()
            })
            .expect("explicit GPU stage tests require a physical GPU");
            assert!(matches!(
                context.adapter_info().device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            ));
            eprintln!("GPU stage adapter: {:?}", context.adapter_info());
            context
        })
        .clone()
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let byte = (value & 127) as u8;
        value >>= 7;
        output.push(byte | if value == 0 { 0 } else { 128 });
        if value == 0 {
            return output;
        }
    }
}
fn number(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}
fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(value.len() as u64),
        value.to_vec(),
    ]
    .concat()
}
fn integer(name: &str, value: u64) -> Vec<u8> {
    bytes(
        5,
        &[bytes(1, name.as_bytes()), number(3, value), number(20, 2)].concat(),
    )
}
fn integers(name: &str, values: &[u64]) -> Vec<u8> {
    bytes(
        5,
        &[
            bytes(1, name.as_bytes()),
            values.iter().flat_map(|&value| number(8, value)).collect(),
            number(20, 7),
        ]
        .concat(),
    )
}
fn node(op: &str, inputs: &[&str], output: &str, attributes: Vec<u8>) -> Vec<u8> {
    bytes(
        1,
        &[
            inputs
                .iter()
                .flat_map(|input| bytes(1, input.as_bytes()))
                .collect(),
            bytes(2, output.as_bytes()),
            bytes(4, op.as_bytes()),
            attributes,
        ]
        .concat(),
    )
}
fn tensor(name: &str, shape: &[u64], values: &[f32]) -> Vec<u8> {
    bytes(
        5,
        &[
            shape
                .iter()
                .flat_map(|&dimension| number(1, dimension))
                .collect(),
            number(2, 1),
            bytes(8, name.as_bytes()),
            bytes(
                9,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ]
        .concat(),
    )
}
fn info(name: &str, shape: &[Option<u64>]) -> Vec<u8> {
    let shape = shape
        .iter()
        .enumerate()
        .flat_map(|(index, dimension)| {
            bytes(
                1,
                &dimension.map_or_else(
                    || bytes(2, format!("d{index}").as_bytes()),
                    |dimension| number(1, dimension),
                ),
            )
        })
        .collect::<Vec<_>>();
    bytes(
        if name == "x" { 11 } else { 12 },
        &[
            bytes(1, name.as_bytes()),
            bytes(2, &bytes(1, &[number(1, 1), bytes(2, &shape)].concat())),
        ]
        .concat(),
    )
}
fn head_model(classification: bool) -> Arc<Model> {
    let classes = if classification { 2 } else { 3 };
    let mut bias = vec![0.0; classes];
    bias[1] = 1e-8;
    if !classification {
        bias[2] = -2.0;
    }
    let mut parts = vec![
        tensor("weights", &[3, classes as u64], &vec![0.0; 3 * classes]),
        tensor("bias", &[classes as u64], &bias),
    ];
    if classification {
        parts.push(node(
            "ReduceMean",
            &["x"],
            "feature",
            [integers("axes", &[2, 3]), integer("keepdims", 0)].concat(),
        ));
    } else {
        parts.push(node(
            "ReduceMean",
            &["x"],
            "mean",
            [integers("axes", &[2]), integer("keepdims", 0)].concat(),
        ));
        parts.push(node(
            "Transpose",
            &["mean"],
            "feature",
            integers("perm", &[0, 2, 1]),
        ));
    }
    parts.push(node(
        "MatMul",
        &["feature", "weights"],
        "projection",
        vec![],
    ));
    parts.push(node("Add", &["projection", "bias"], "logits", vec![]));
    parts.push(node(
        "Softmax",
        &["logits"],
        "y",
        integer("axis", if classification { 1 } else { 2 }),
    ));
    parts.push(info(
        "x",
        if classification {
            &[Some(1), Some(3), Some(80), Some(160)]
        } else {
            &[None, Some(3), Some(48), None]
        },
    ));
    parts.push(info(
        "y",
        if classification {
            &[Some(1), Some(2)]
        } else {
            &[None, None, Some(3)]
        },
    ));
    let source = [
        number(1, 8),
        bytes(8, &number(2, 7)),
        bytes(7, &parts.concat()),
    ]
    .concat();
    Arc::new(
        Model::load_bytes_with_options(
            &source,
            &LoadOptions {
                optimization: OptimizationPolicy::PreserveArithmetic,
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_stage_heads_preserve_rounded_probability_ties_and_public_trace() {
    let context = hardware();
    eprintln!("GPU stage heads: {:?}", context.adapter_info());
    let model = head_model(false);
    let compiled = Arc::new(GpuCompiledModel::compile(model.clone(), context.clone()).unwrap());
    assert_eq!(model.cached_packed_weight_bytes(), 0);
    let dictionary = Arc::new(Dictionary::from_utf8(b"A\n").unwrap());
    let pixels = vec![255u8; 64 * 48 * 3];
    let image = ImageView::new(&pixels, 64, 48, 0, PixelFormat::Bgr24).unwrap();
    for mode in [
        RecognitionScoreMode::Logit,
        RecognitionScoreMode::Probability,
    ] {
        let mut recognizer = Recognizer::new(
            compiled.clone(),
            dictionary.clone(),
            RecognizerOptions {
                score_mode: mode,
                ..Default::default()
            },
            ExecutionLimits::default(),
        )
        .unwrap();
        let expected = recognizer
            .recognize(image, true)
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(
            expected.text,
            if mode == RecognitionScoreMode::Logit {
                "A"
            } else {
                ""
            }
        );
        let mut saw_dense = false;
        let mut saw_public = false;
        let traced = recognizer
            .recognize_with_trace(image, true, |event| {
                match event {
                    RecognitionTrace::Probabilities { values, .. } => {
                        assert_eq!(values[0], values[1]);
                        saw_dense = true;
                    }
                    RecognitionTrace::Public(view) => {
                        assert_eq!(view.text, expected.text);
                        saw_public = true;
                    }
                    _ => {}
                }
                Ok(())
            })
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(traced, expected);
        assert!(saw_dense && saw_public);
        for result in recognizer
            .recognize_batch(&[image, image], true)
            .unwrap()
            .iter()
        {
            assert_eq!(result.try_to_owned().unwrap(), expected);
        }
        assert_eq!(
            recognizer
                .recognize(image, true)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert!(recognizer.execution_statistics().plan_cache_hits > 0);
        assert!(
            recognizer.retained_activation_bytes()
                + recognizer.diagnostic_retained_activation_bytes()
                <= recognizer.execution_limits().max_activation_bytes
        );
    }

    let compiled = Arc::new(GpuCompiledModel::compile(head_model(true), context).unwrap());
    let mut classifier = Classifier::new(compiled, ExecutionLimits::default()).unwrap();
    let expected = classifier.classify(image).unwrap();
    assert_eq!(expected.label, 1);
    assert_eq!(expected.score, 0.5);
    let mut dense_tie = false;
    let traced = classifier
        .classify_with_trace(image, |event| {
            if let ClassifierTrace::Probabilities(values) = event {
                assert_eq!(values, &[0.5, 0.5]);
                dense_tie = true;
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(traced, expected);
    assert!(dense_tie);
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_stage_diagnostic_limits_include_production_and_remain_reusable() {
    let context = hardware();
    let compiled = Arc::new(GpuCompiledModel::compile(head_model(false), context.clone()).unwrap());
    let dictionary = Arc::new(Dictionary::from_utf8(b"A\n").unwrap());
    let options = RecognizerOptions {
        min_width: 64,
        ..Default::default()
    };
    let baseline = Recognizer::new(
        compiled.clone(),
        dictionary.clone(),
        options,
        ExecutionLimits::default(),
    )
    .unwrap();
    let mut limited = Recognizer::new(
        compiled,
        dictionary,
        options,
        ExecutionLimits {
            max_activation_bytes: baseline.retained_activation_bytes(),
        },
    )
    .unwrap();
    let pixels = vec![255u8; 64 * 48 * 3];
    let image = ImageView::new(&pixels, 64, 48, 0, PixelFormat::Bgr24).unwrap();
    assert!(
        limited
            .recognize_with_trace(image, true, |_| Ok(()))
            .is_err()
    );
    assert_eq!(limited.recognize(image, true).unwrap().text, "A");
    assert!(
        limited.retained_activation_bytes() + limited.diagnostic_retained_activation_bytes()
            <= limited.execution_limits().max_activation_bytes
    );

    let compiled = Arc::new(GpuCompiledModel::compile(head_model(true), context).unwrap());
    let baseline = Classifier::new(compiled.clone(), ExecutionLimits::default()).unwrap();
    let mut limited = Classifier::new(
        compiled,
        ExecutionLimits {
            max_activation_bytes: baseline.retained_activation_bytes(),
        },
    )
    .unwrap();
    assert!(limited.classify_with_trace(image, |_| Ok(())).is_err());
    assert_eq!(limited.classify(image).unwrap().label, 1);
}
