use serde_json::Value;
use simd_paddle_ocr_rs::ocr::OwnedOcr;
use std::error::Error;

/// Validate the public OCR contract before warming a performance configuration.
pub fn compare(actual: &OwnedOcr, expected: &Value, capture: bool) -> Result<(), Box<dyn Error>> {
    let actual = serde_json::to_value(actual)?;
    for field in [
        "text",
        "detected_count",
        "detector_resized_width",
        "detector_resized_height",
    ] {
        if actual[field] != expected[field] {
            return Err(format!("CPU reference differs: {field}").into());
        }
    }
    let a = actual["lines"].as_array().ok_or("missing lines")?;
    let b = expected["lines"]
        .as_array()
        .ok_or("missing reference lines")?;
    if a.len() != b.len() {
        return Err("CPU reference differs: line count".into());
    }
    let close = |a: &Value, b: &Value, coordinate: bool| -> Result<(), Box<dyn Error>> {
        let (a, b) = (
            a.as_f64().ok_or("nonfinite/missing value")?,
            b.as_f64().ok_or("nonfinite/missing reference value")?,
        );
        let tolerance = if coordinate {
            0.5
        } else {
            1e-3 + 1e-3 * b.abs()
        };
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance {
            return Err(format!("CPU reference differs: {a} vs {b}").into());
        }
        Ok(())
    };
    for (a, b) in a.iter().zip(b) {
        for field in [
            "text",
            "classification_label",
            "applied_rotation_degrees",
            "recognition_content_width",
            "recognition_tensor_width",
            "recognition_time_steps",
            "emitted_count",
        ] {
            if a[field] != b[field] {
                return Err(format!("CPU reference differs: line {field}").into());
            }
        }
        for field in ["recognition_score", "classification_score"] {
            close(&a[field], &b[field], false)?;
        }
        close(&a["detection"]["score"], &b["detection"]["score"], false)?;
        for point in 0..4 {
            for axis in 0..2 {
                close(
                    &a["detection"]["points"][point][axis],
                    &b["detection"]["points"][point][axis],
                    true,
                )?;
            }
        }
        if capture {
            let aa = a["ctc_spans"].as_array().ok_or("missing CTC spans")?;
            let bb = b["ctc_spans"]
                .as_array()
                .ok_or("missing reference CTC spans")?;
            if aa.len() != bb.len() {
                return Err("CPU reference differs: CTC span count".into());
            }
            for (a, b) in aa.iter().zip(bb) {
                for field in ["text", "start_column", "end_column"] {
                    if a[field] != b[field] {
                        return Err(format!("CPU reference differs: CTC {field}").into());
                    }
                }
                close(&a["score"], &b["score"], false)?;
            }
        }
    }
    Ok(())
}
