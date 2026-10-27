//! Asset-backed regression and probability-map replay; no performance measurements.
use super::*;

#[test]
#[ignore = "physical Metal and local OCR assets; set SIMD_OCR_TEST_MODEL and SIMD_OCR_TEST_IMAGE"]
fn metal_detector_geometry_precision_replay_with_assets() {
    let model = Arc::new(
        crate::model::Model::load_path(std::env::var("SIMD_OCR_TEST_MODEL").unwrap()).unwrap(),
    );
    let pixels = image::open(std::env::var("SIMD_OCR_TEST_IMAGE").unwrap())
        .unwrap()
        .to_rgb8();
    let view = ImageView::new(
        pixels.as_raw(),
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        crate::image::PixelFormat::Rgb24,
    )
    .unwrap();
    let geometry = DetSourceGeometry::new(view.width(), view.height()).unwrap();
    let options = DetectorOptions::default();
    let size = geometry
        .resize(options.limit_side_length, options.resize_policy)
        .unwrap();
    let mut full_geometry = Vec::new();
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        let context = GpuContext::new().unwrap();
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                model.clone(),
                context.clone(),
                crate::runtime::CompileOptions {
                    gpu: GpuCompileOptions {
                        precision,
                        matrix_precision: GpuMatrixPrecision::F16,
                    },
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        compiled.prepare_images().unwrap();
        let options = crate::detector::DetectorPlan::new(
            compiled.clone(),
            options,
            ExecutionLimits::default(),
        )
        .unwrap()
        .options();
        let mut session = GpuSession::new_resident(
            compiled,
            &[1, 3, size.height, size.width],
            ExecutionLimits::default(),
            None,
        )
        .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let source = Image::upload(&mut batch, view).unwrap();
        preprocess(
            &mut batch,
            &source,
            &session.resident_input(),
            [geometry.padded_width, geometry.padded_height],
            size.width,
            true,
        )
        .unwrap();
        let prediction = batch.graph(&mut session, "detector").unwrap();
        let (bytes, _) = batch
            .read(&prediction.buffer, prediction.bytes(), false)
            .unwrap();
        let mut values = vec![0.; size.width * size.height];
        prediction
            .descriptor
            .precision
            .decode(&bytes, &mut values)
            .unwrap();
        let mut oracle = crate::db::Workspace::default();
        crate::db::run(
            &values,
            size.width,
            size.height,
            &options,
            geometry,
            &mut oracle,
        )
        .unwrap();
        eprintln!("{precision:?} CPU geometry {:?}", oracle.boxes);
        if precision == GpuPrecision::F32 {
            full_geometry = oracle.boxes.clone();
        }
        let mut resident =
            Detector::new(session.compiled.clone(), ExecutionLimits::default()).unwrap();
        let stable = resident.run(view, options, false).unwrap();
        assert_eq!(stable.boxes.len(), full_geometry.len());
        for (a, b) in stable.boxes.iter().zip(&full_geometry) {
            for (a, b) in a.points.iter().flatten().zip(b.points.iter().flatten()) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "resident detector changed FP32 crop geometry"
                );
            }
        }

        for matrix in [GpuMatrixPrecision::F32, GpuMatrixPrecision::F16] {
            let (actual, _, _, _) = postprocess_with_math(
                Batch::new(context.clone()).unwrap(),
                &prediction,
                options,
                geometry,
                false,
                matrix,
            )
            .unwrap();
            eprintln!("{precision:?} {matrix:?} GPU geometry {actual:?}");
            assert_eq!(actual.len(), oracle.boxes.len());
            if matrix == GpuMatrixPrecision::F32 {
                for (a, b) in actual.iter().zip(&oracle.boxes) {
                    for (a, b) in a.points.iter().flatten().zip(b.points.iter().flatten()) {
                        assert!((a - b).abs() < 1e-3);
                    }
                }
            }
        }
    }
}
