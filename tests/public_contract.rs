use simd_paddle_ocr_rs::{
    cpu::CpuOptions,
    detector::DetectionBox,
    error::ErrorKind,
    image::ImageError,
    lines::{LineMetadata, OrientedLines, OwnedOrientedLines},
    onnx::ImportError,
    runtime::RuntimeError,
    tensor::{Shape, Tensor, TensorError},
};
use std::error::Error;

fn line() -> LineMetadata {
    LineMetadata {
        detection: DetectionBox {
            points: [[0.; 2]; 4],
            score: 1.,
        },
        offset: 0,
        bytes: 6,
        width: 2,
        height: 1,
        unrotated_width: 2,
        unrotated_height: 1,
        vertical_rotation: false,
        classification: None,
        applied_rotation_degrees: 0,
    }
}

#[test]
fn checked_line_collections_reject_bad_ranges_and_preserve_every_valid_line() {
    for (offset, bytes, width) in [(usize::MAX, 6, 2), (1, 6, 2), (0, 3, 2), (0, 6, 0)] {
        let mut metadata = line();
        metadata.offset = offset;
        metadata.bytes = bytes;
        metadata.width = width;
        assert!(OrientedLines::new(2, 1, 1., 1., &[metadata.clone()], &[0; 6]).is_err());
        assert!(OwnedOrientedLines::new(2, 1, 1., 1., vec![metadata], vec![0; 6]).is_err());
    }
    let metadata = [line(), line()];
    let view = OrientedLines::new(2, 1, 1., 1., &metadata, &[0; 6]).unwrap();
    assert_eq!(view.iter().len(), 2);
    assert_eq!(view.iter().count(), view.len());
    assert!(view.get(2).is_none());
    assert_eq!(view.try_to_owned().unwrap().as_view().iter().count(), 2);
}

#[test]
fn cpu_budgets_are_valid_for_zero_and_extreme_line_counts() {
    for capacity in [1, 2, 8, 16, 1024] {
        for workers in [0, 1, 16, usize::MAX] {
            let budget = CpuOptions {
                line_workers: workers,
                ..Default::default()
            }
            .resolve(capacity, true)
            .unwrap();
            for lines in [0, 1, 4, 16, usize::MAX] {
                assert!((1..=capacity.min(16)).contains(&budget.recognizer_threads(lines)));
            }
            assert_eq!(budget.capacity(), capacity);
        }
    }
    for capacity in [0, 1025, usize::MAX] {
        assert!(CpuOptions::default().resolve(capacity, true).is_err());
    }
}

#[test]
fn error_kinds_and_concrete_causes_survive_stage_boundaries_and_cloning() {
    let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
    let import = ImportError::from(io).context("model.onnx");
    assert_eq!(import.kind(), ErrorKind::Io);
    assert_eq!(
        import
            .clone()
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::PermissionDenied
    );
    let tensor = TensorError::new("bad shape");
    let runtime = RuntimeError::from(tensor).context("node 7");
    let image = ImageError::from(runtime);
    assert_eq!(image.kind(), ErrorKind::InvalidInput);
    assert!(image.source().unwrap().is::<RuntimeError>());
    assert!(
        image
            .source()
            .unwrap()
            .source()
            .unwrap()
            .is::<TensorError>()
    );
    assert!(image.to_string().contains("node 7"));
}

#[test]
fn defaulted_tensor_does_not_promise_numeric_zero() {
    #[derive(Clone, Debug, PartialEq)]
    struct Nonzero(u8);
    impl Default for Nonzero {
        fn default() -> Self {
            Self(7)
        }
    }
    let tensor = Tensor::<Nonzero>::defaulted(Shape::new(&[2]).unwrap()).unwrap();
    assert_eq!(tensor.data(), &[Nonzero(7), Nonzero(7)]);
}
