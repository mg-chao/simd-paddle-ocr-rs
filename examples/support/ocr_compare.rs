use serde_json::Value;
use simd_paddle_ocr_rs::ocr::OwnedOcr;
use std::error::Error;

/// Validate the public OCR contract before warming a performance configuration.
pub fn compare(actual: &OwnedOcr, expected: &Value, capture: bool) -> Result<(), Box<dyn Error>> {
    compare_with_score_tolerance(actual, expected, capture, 1e-3, 1e-3)
}

/// Validate explicitly selected FP16 matrix arithmetic against the C# mixed gate.
/// Text, metadata, CTC columns, and coordinate tolerances match the FP32 contract.
#[allow(dead_code)] // Retain the strict-geometry mixed gate for comparisons.
pub fn compare_mixed(
    actual: &OwnedOcr,
    expected: &Value,
    capture: bool,
) -> Result<(), Box<dyn Error>> {
    compare_with_score_tolerance(actual, expected, capture, 0.05, 0.01)
}

const FAST_MIN_POLYGON_IOU: f64 = 0.8;
const FAST_MAX_COORDINATE_ERROR: f64 = 2.0;
const FAST_EQUIVALENT_COORDINATE_ERROR: f64 = 1e-3;

/// Explicit fast-mode contract: text and reading order remain exact. Geometry
/// may move by at most two pixels per coordinate of each matched corner and
/// must retain at least 80% true polygon overlap. The overlap bound protects
/// thin text boxes for which a small movement can change most of the region.
/// Derived recognition dimensions and CTC columns are checked internally rather
/// than replaced with reference values. Strict validators above are unchanged.
/// Confidence scores use the mixed numeric gate only when the input crop
/// geometry and recognition dimensions match; shifted crops report raw scores.
pub fn compare_fast(
    actual: &OwnedOcr,
    expected: &Value,
    capture: bool,
) -> Result<(), Box<dyn Error>> {
    let actual = serde_json::to_value(actual)?;
    compare_fast_values(&actual, expected, capture)
}

/// The same fast validator for live output and frozen JSON replay.
pub fn compare_fast_values(
    actual: &Value,
    expected: &Value,
    capture: bool,
) -> Result<(), Box<dyn Error>> {
    validate_fast_document(actual, capture)?;
    validate_fast_document(expected, capture)?;
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
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        for field in [
            "text",
            "classification_label",
            "applied_rotation_degrees",
            "emitted_count",
        ] {
            if a[field] != b[field] {
                return Err(format!("CPU reference differs: line {index} {field}").into());
            }
        }
        let (iou, coordinate_error, distance) = geometry_difference(a, b)?;
        if iou < FAST_MIN_POLYGON_IOU || coordinate_error > FAST_MAX_COORDINATE_ERROR {
            return Err(format!("CPU reference differs: line {index} polygon IoU {iou}, maximum coordinate error {coordinate_error}, maximum corner distance {distance}").into());
        }
        if confidence_scores_comparable(a, b, coordinate_error) {
            fast_score(&a["recognition_score"], &b["recognition_score"])?;
            fast_score(&a["classification_score"], &b["classification_score"])?;
            fast_score(&a["detection"]["score"], &b["detection"]["score"])?;
        }
    }
    Ok(())
}

/// Report raw geometry, shape and CTC changes separately from qualification.
/// Invalid or absent polygons remain reportable when comparison itself failed.
pub fn fast_differences(
    actual: &OwnedOcr,
    expected: &Value,
    capture: bool,
) -> Result<Value, Box<dyn Error>> {
    let actual = serde_json::to_value(actual)?;
    fast_differences_values(&actual, expected, capture)
}

/// Report frozen JSON without altering or reconstructing its public fields.
pub fn fast_differences_values(
    actual: &Value,
    expected: &Value,
    capture: bool,
) -> Result<Value, Box<dyn Error>> {
    let a = actual["lines"].as_array();
    let b = expected["lines"].as_array();
    let mut lines = Vec::new();
    for index in 0..a.map_or(0, Vec::len).max(b.map_or(0, Vec::len)) {
        let a = a.and_then(|lines| lines.get(index)).unwrap_or(&Value::Null);
        let b = b.and_then(|lines| lines.get(index)).unwrap_or(&Value::Null);
        let mut line = serde_json::json!({"line":index});
        for (field, actual, expected) in [
            (
                "recognition_score",
                &a["recognition_score"],
                &b["recognition_score"],
            ),
            (
                "classification_score",
                &a["classification_score"],
                &b["classification_score"],
            ),
            (
                "detection_score",
                &a["detection"]["score"],
                &b["detection"]["score"],
            ),
        ] {
            let delta = actual.as_f64().zip(expected.as_f64()).map(|(a, b)| a - b);
            line[field] = serde_json::json!({"actual":actual, "expected":expected, "delta":delta});
        }
        match geometry_difference(a, b) {
            Ok((iou, coordinate_error, distance)) => {
                line["polygon_iou"] = serde_json::json!(iou);
                line["maximum_coordinate_error"] = serde_json::json!(coordinate_error);
                line["maximum_corner_distance"] = serde_json::json!(distance);
                let comparable = confidence_scores_comparable(a, b, coordinate_error);
                line["numeric_score_comparison"] = serde_json::json!({
                    "applied":comparable,
                    "reason":if comparable { "equivalent input geometry" } else { "changed input geometry" },
                });
            }
            Err(error) => {
                line["geometry_error"] = serde_json::json!(error.to_string());
                line["numeric_score_comparison"] = serde_json::json!({
                    "applied":false, "reason":"invalid input geometry",
                });
            }
        }
        let mut changes = serde_json::Map::new();
        for field in [
            "recognition_content_width",
            "recognition_tensor_width",
            "recognition_time_steps",
            "emitted_count",
        ] {
            if a[field] != b[field] {
                changes.insert(field.into(), difference(&a[field], &b[field]));
            }
        }
        line["shape_changes"] = Value::Object(changes);
        if capture || a["ctc_spans"].is_array() || b["ctc_spans"].is_array() {
            let aa = a["ctc_spans"].as_array();
            let bb = b["ctc_spans"].as_array();
            let mut columns = Vec::new();
            for token in 0..aa.map_or(0, Vec::len).max(bb.map_or(0, Vec::len)) {
                let x = aa
                    .and_then(|spans| spans.get(token))
                    .unwrap_or(&Value::Null);
                let y = bb
                    .and_then(|spans| spans.get(token))
                    .unwrap_or(&Value::Null);
                if x["text"] != y["text"]
                    || x["start_column"] != y["start_column"]
                    || x["end_column"] != y["end_column"]
                {
                    columns.push(serde_json::json!({"token":token, "actual_text":x["text"], "expected_text":y["text"],
                        "start_column":difference(&x["start_column"], &y["start_column"]),
                        "end_column":difference(&x["end_column"], &y["end_column"])}));
                }
            }
            line["ctc"] = serde_json::json!({
                "span_count":difference(&serde_json::json!(aa.map(Vec::len)), &serde_json::json!(bb.map(Vec::len))),
                "column_changes":columns,
            });
        }
        lines.push(line);
    }
    Ok(
        serde_json::json!({"minimum_polygon_iou":FAST_MIN_POLYGON_IOU,
        "maximum_allowed_coordinate_error":FAST_MAX_COORDINATE_ERROR,
        "line_count":difference(&serde_json::json!(a.map(Vec::len)), &serde_json::json!(b.map(Vec::len))),
        "lines":lines}),
    )
}

fn difference(actual: &Value, expected: &Value) -> Value {
    let delta = actual
        .as_i64()
        .zip(expected.as_i64())
        .and_then(|(a, b)| a.checked_sub(b));
    serde_json::json!({"actual":actual,"expected":expected,"delta":delta})
}

fn confidence_scores_comparable(actual: &Value, expected: &Value, coordinate_error: f64) -> bool {
    coordinate_error <= FAST_EQUIVALENT_COORDINATE_ERROR
        && [
            "recognition_content_width",
            "recognition_tensor_width",
            "recognition_time_steps",
        ]
        .into_iter()
        .all(|field| actual[field].as_u64().is_some() && actual[field] == expected[field])
}

fn fast_score(actual: &Value, expected: &Value) -> Result<(), Box<dyn Error>> {
    let a = actual.as_f64().ok_or("nonfinite/missing score")?;
    let b = expected
        .as_f64()
        .ok_or("nonfinite/missing reference score")?;
    if !a.is_finite() || !b.is_finite() || (a - b).abs() > 0.05 + 0.01 * b.abs() {
        return Err(format!("CPU reference differs: score {a} vs {b}").into());
    }
    Ok(())
}

fn count(value: &Value, field: &str) -> Result<u64, Box<dyn Error>> {
    value[field]
        .as_u64()
        .ok_or_else(|| format!("missing/invalid {field}").into())
}

fn validate_fast_document(document: &Value, capture: bool) -> Result<(), Box<dyn Error>> {
    let lines = document["lines"].as_array().ok_or("missing lines")?;
    if count(document, "detected_count")? != lines.len() as u64 {
        return Err("detected count does not match lines".into());
    }
    if count(document, "detector_resized_width")? == 0
        || count(document, "detector_resized_height")? == 0
    {
        return Err("invalid detector dimensions".into());
    }
    let mut text = Vec::with_capacity(lines.len());
    for line in lines {
        let line_text = line["text"].as_str().ok_or("missing line text")?;
        text.push(line_text);
        let content = count(line, "recognition_content_width")?;
        let tensor = count(line, "recognition_tensor_width")?;
        let steps = count(line, "recognition_time_steps")?;
        let emitted = count(line, "emitted_count")?;
        if content == 0
            || tensor < content
            || steps == 0
            || emitted > steps
            || (emitted == 0 && !line_text.is_empty())
        {
            return Err("invalid recognition dimensions or emitted count".into());
        }
        if !matches!(count(line, "applied_rotation_degrees")?, 0 | 180) {
            return Err("invalid applied rotation".into());
        }
        for value in [
            &line["recognition_score"],
            &line["classification_score"],
            &line["detection"]["score"],
        ] {
            if value.as_f64().is_none_or(|score| !score.is_finite()) {
                return Err("nonfinite/missing score".into());
            }
        }
        for value in [&line["classification_score"], &line["detection"]["score"]] {
            if value
                .as_f64()
                .is_none_or(|score| !(0.0..=1.0).contains(&score))
            {
                return Err("classification/detection score is outside [0,1]".into());
            }
        }
        if let Some(spans) = line["ctc_spans"].as_array() {
            if spans.len() as u64 != emitted {
                return Err("CTC span count differs from emitted count".into());
            }
            let mut previous = 0;
            let mut joined = String::new();
            for span in spans {
                let start = count(span, "start_column")?;
                let end = count(span, "end_column")?;
                if start < previous || start >= end || end > steps {
                    return Err("invalid CTC column range/order".into());
                }
                if span["score"]
                    .as_f64()
                    .is_none_or(|score| !score.is_finite())
                {
                    return Err("nonfinite/missing CTC score".into());
                }
                joined.push_str(span["text"].as_str().ok_or("missing CTC text")?);
                previous = end;
            }
            if joined != line_text {
                return Err("CTC token concatenation differs from line text".into());
            }
        } else if capture || !line["ctc_spans"].is_null() {
            return Err("missing/invalid CTC spans".into());
        }
    }
    if document["text"].as_str() != Some(text.join("\n").as_str()) {
        return Err("joined text differs from lines".into());
    }
    Ok(())
}

type Point = [f64; 2];

struct Quad([Point; 4]);
impl Quad {
    fn parse(value: &Value) -> Result<Self, Box<dyn Error>> {
        let vertices = value
            .as_array()
            .filter(|v| v.len() == 4)
            .ok_or("detection requires four corners")?;
        let mut points = [[0.; 2]; 4];
        for (point, vertex) in points.iter_mut().zip(vertices) {
            let coordinates = vertex
                .as_array()
                .filter(|v| v.len() == 2)
                .ok_or("invalid detection corner")?;
            for axis in 0..2 {
                point[axis] = coordinates[axis]
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or("nonfinite detection corner")?;
            }
        }
        let area = signed_area(&points);
        if !area.is_finite() || area.abs() <= 1e-9 {
            return Err("degenerate detection polygon".into());
        }
        let orientation = area.signum();
        if (0..4).any(|i| {
            let turn = cross(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) * orientation;
            !turn.is_finite() || turn <= 1e-9
        }) {
            return Err("detection polygon is not ordered and convex".into());
        }
        Ok(Self(points))
    }
    fn overlap(&self, other: &Self) -> f64 {
        // Sutherland-Hodgman clips the true convex polygons, including rotated
        // quads. No axis-aligned bounding rectangle substitutes for the area.
        let mut polygon = self.0.to_vec();
        let orientation = signed_area(&other.0).signum();
        for edge in 0..4 {
            let a = other.0[edge];
            let b = other.0[(edge + 1) % 4];
            let input = std::mem::take(&mut polygon);
            if input.is_empty() {
                break;
            }
            let mut previous = *input.last().unwrap();
            let mut old_distance = cross(a, b, previous) * orientation;
            for current in input {
                let distance = cross(a, b, current) * orientation;
                if (distance >= 0.) != (old_distance >= 0.) {
                    let t = old_distance / (old_distance - distance);
                    polygon.push([
                        previous[0] + t * (current[0] - previous[0]),
                        previous[1] + t * (current[1] - previous[1]),
                    ]);
                }
                if distance >= 0. {
                    polygon.push(current);
                }
                previous = current;
                old_distance = distance;
            }
        }
        let a = signed_area(&self.0).abs();
        let b = signed_area(&other.0).abs();
        let intersection = signed_area(&polygon).abs().min(a.min(b));
        (intersection / (a + b - intersection)).clamp(0., 1.)
    }
    fn corner_errors(&self, other: &Self) -> (f64, f64) {
        (0..4)
            .flat_map(|shift| {
                [false, true].map(move |reverse| {
                    (0..4)
                        .map(|i| {
                            let j = if reverse {
                                (shift + 4 - i) % 4
                            } else {
                                (shift + i) % 4
                            };
                            let dx = (self.0[i][0] - other.0[j][0]).abs();
                            let dy = (self.0[i][1] - other.0[j][1]).abs();
                            (dx.max(dy), dx.hypot(dy))
                        })
                        .fold((0.0_f64, 0.0_f64), |(coordinate, distance), (c, d)| {
                            (coordinate.max(c), distance.max(d))
                        })
                })
            })
            // Use one correspondence for both diagnostics. Minimize the
            // coordinate gate first, breaking ties by Euclidean distance.
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))
            .unwrap()
    }
}

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn signed_area(points: &[Point]) -> f64 {
    if points.len() < 3 {
        return 0.;
    }
    let origin = points[0];
    (1..points.len() - 1)
        .map(|i| cross(origin, points[i], points[i + 1]))
        .sum::<f64>()
        * 0.5
}

fn geometry_difference(
    actual: &Value,
    expected: &Value,
) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let a = Quad::parse(&actual["detection"]["points"])?;
    let b = Quad::parse(&expected["detection"]["points"])?;
    let (coordinate_error, distance) = a.corner_errors(&b);
    Ok((a.overlap(&b), coordinate_error, distance))
}

fn compare_with_score_tolerance(
    actual: &OwnedOcr,
    expected: &Value,
    capture: bool,
    score_absolute: f64,
    score_relative: f64,
) -> Result<(), Box<dyn Error>> {
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
            score_absolute + score_relative * b.abs()
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

#[cfg(test)]
mod tests {
    use super::*;
    use simd_paddle_ocr_rs::{detector::DetectionBox, ocr::OwnedOcrLine, recognizer::OwnedCtcSpan};

    fn rectangle(width: f64, height: f64, angle: f64) -> [[f32; 2]; 4] {
        let (sine, cosine) = angle.sin_cos();
        [
            [-width / 2., -height / 2.],
            [width / 2., -height / 2.],
            [width / 2., height / 2.],
            [-width / 2., height / 2.],
        ]
        .map(|[x, y]| {
            [
                (30. + x * cosine - y * sine) as f32,
                (40. + x * sine + y * cosine) as f32,
            ]
        })
    }

    fn fixture() -> OwnedOcr {
        OwnedOcr {
            text: "甲乙".into(),
            detected_count: 1,
            detector_resized_width: 256,
            detector_resized_height: 256,
            width_ratio: 1.,
            height_ratio: 1.,
            lines: vec![OwnedOcrLine {
                detection: DetectionBox {
                    points: rectangle(100., 40., 0.),
                    score: 0.9,
                },
                text: "甲乙".into(),
                recognition_score: 20.,
                classification_score: 0.95,
                classification_label: 0,
                applied_rotation_degrees: 0,
                emitted_count: 2,
                recognition_content_width: 128,
                recognition_tensor_width: 128,
                recognition_time_steps: 8,
                ctc_spans: Some(vec![
                    OwnedCtcSpan {
                        text: "甲".into(),
                        score: 20.,
                        start_column: 0,
                        end_column: 2,
                    },
                    OwnedCtcSpan {
                        text: "乙".into(),
                        score: 20.,
                        start_column: 4,
                        end_column: 6,
                    },
                ]),
            }],
        }
    }

    fn shifted(actual: &mut OwnedOcr, x: f32, y: f32) {
        for point in &mut actual.lines[0].detection.points {
            point[0] += x;
            point[1] += y;
        }
    }

    #[test]
    fn fast_accepts_small_overlap_and_reports_shape_and_column_changes() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected.clone();
        shifted(&mut actual, 1.5, 1.5);
        actual.lines[0].recognition_content_width = 130;
        actual.lines[0].recognition_tensor_width = 160;
        actual.lines[0].recognition_time_steps = 10;
        actual.lines[0].recognition_score += 0.1;
        let spans = actual.lines[0].ctc_spans.as_mut().unwrap();
        spans[0].start_column = 1;
        spans[0].end_column = 3;
        spans[1].start_column = 5;
        spans[1].end_column = 9;
        assert!(compare_fast(&actual, &reference, true).is_ok());
        assert!(compare(&actual, &reference, true).is_err());
        assert!(compare_mixed(&actual, &reference, true).is_err());
        let report = fast_differences(&actual, &reference, true).unwrap();
        assert!(report["lines"][0]["polygon_iou"].as_f64().unwrap() > 0.9);
        assert!(
            (report["lines"][0]["maximum_corner_distance"]
                .as_f64()
                .unwrap()
                - 1.5f64.hypot(1.5))
            .abs()
                < 1e-9
        );
        assert_eq!(report["lines"][0]["maximum_coordinate_error"], 1.5);
        assert_eq!(report["maximum_allowed_coordinate_error"], 2.0);
        assert_eq!(
            report["lines"][0]["shape_changes"]["recognition_time_steps"]["delta"],
            2
        );
        assert_eq!(
            report["lines"][0]["ctc"]["column_changes"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            report["lines"][0]["ctc"]["column_changes"][1]["end_column"]["delta"],
            3
        );
        assert_eq!(
            actual.lines[0].recognition_time_steps, 10,
            "reporting must not normalize outputs"
        );
    }

    #[test]
    fn fast_enforces_both_coordinate_bound_and_true_rotated_polygon_overlap() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected.clone();
        shifted(&mut actual, 2.01, 0.);
        let report = fast_differences(&actual, &reference, true).unwrap();
        assert!(report["lines"][0]["polygon_iou"].as_f64().unwrap() > 0.9);
        assert!(
            report["lines"][0]["maximum_coordinate_error"]
                .as_f64()
                .unwrap()
                > 2.0
        );
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual = expected.clone();
        shifted(&mut actual, 200., 0.);
        assert_eq!(
            fast_differences(&actual, &reference, true).unwrap()["lines"][0]["polygon_iou"],
            0.
        );
        assert!(compare_fast(&actual, &reference, true).is_err());

        let mut expected = fixture();
        expected.lines[0].detection.points = rectangle(10., 1., 5f64.to_radians());
        let mut actual = expected.clone();
        actual.lines[0].detection.points = rectangle(10., 1., -5f64.to_radians());
        let reference = serde_json::to_value(&expected).unwrap();
        let (iou, coordinate_error, distance) = geometry_difference(
            &serde_json::to_value(&actual.lines[0]).unwrap(),
            &reference["lines"][0],
        )
        .unwrap();
        assert!(
            coordinate_error < 2. && distance < 2. && iou < 0.8,
            "thin rotated quads need true area overlap: {iou}/{distance}"
        );
        let bounds = |q: [[f32; 2]; 4]| {
            [
                q.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min),
                q.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max),
                q.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min),
                q.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max),
            ]
        };
        for (a, b) in bounds(actual.lines[0].detection.points)
            .into_iter()
            .zip(bounds(expected.lines[0].detection.points))
        {
            assert!(
                (a - b).abs() < 1e-5,
                "axis-aligned bounds are identical despite poor polygon overlap"
            );
        }
        assert!(compare_fast(&actual, &reference, true).is_err());
    }

    #[test]
    fn fast_handles_small_rotations_and_equivalent_corner_order() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected.clone();
        actual.lines[0].detection.points = rectangle(100., 40., 0.02);
        assert!(compare_fast(&actual, &reference, true).is_ok());
        let points = expected.lines[0].detection.points;
        for order in [[1, 2, 3, 0], [0, 3, 2, 1], [2, 1, 0, 3]] {
            actual.lines[0].detection.points = order.map(|i| points[i]);
            assert!(compare_fast(&actual, &reference, true).is_ok());
            let report = fast_differences(&actual, &reference, true).unwrap();
            assert_eq!(report["lines"][0]["polygon_iou"], 1.);
            assert_eq!(report["lines"][0]["maximum_corner_distance"], 0.);
            assert_eq!(report["lines"][0]["maximum_coordinate_error"], 0.);
        }
    }

    #[test]
    fn fast_rejects_invalid_geometry_and_still_reports_failures() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut actual = expected.clone();
            actual.lines[0].detection.points[0][0] = invalid;
            assert!(compare_fast(&actual, &reference, true).is_err());
            assert!(
                fast_differences(&actual, &reference, true).unwrap()["lines"][0]["geometry_error"]
                    .is_string()
            );
        }
        for polygon in [
            [[0., 0.]; 4],
            [[0., 0.], [4., 4.], [4., 0.], [0., 4.]],
            [[0., 0.], [4., 0.], [1., 1.], [0., 4.]],
        ] {
            let mut actual = expected.clone();
            actual.lines[0].detection.points = polygon;
            assert!(compare_fast(&actual, &reference, true).is_err());
        }
    }

    #[test]
    fn fast_keeps_text_count_order_classification_and_scores_checked() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected.clone();
        actual.text = "different".into();
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual = expected.clone();
        actual.lines[0].text = "different".into();
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual = expected.clone();
        actual.detected_count = 2;
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual = expected.clone();
        actual.lines[0].classification_label = 1;
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual = expected.clone();
        actual.lines[0].applied_rotation_degrees = 180;
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual = expected.clone();
        actual.lines[0].recognition_score = 21.;
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual = expected.clone();
        actual.lines[0].detection.score = f32::NAN;
        assert!(compare_fast(&actual, &reference, true).is_err());
        let mut expected = expected;
        let mut second = expected.lines[0].clone();
        for point in &mut second.detection.points {
            point[1] += 100.;
        }
        expected.lines.push(second);
        expected.detected_count = 2;
        expected.text = "甲乙\n甲乙".into();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected;
        actual.lines.swap(0, 1);
        assert!(
            compare_fast(&actual, &reference, true).is_err(),
            "same-text lines still retain spatial reading order"
        );
    }

    #[test]
    fn fast_compares_confidence_scores_only_for_equivalent_input_geometry() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected.clone();
        actual.lines[0].recognition_score += 1.;
        assert!(compare_fast(&actual, &reference, true).is_err());
        let report = fast_differences(&actual, &reference, true).unwrap();
        assert_eq!(
            report["lines"][0]["numeric_score_comparison"]["applied"],
            true
        );
        assert_eq!(report["lines"][0]["recognition_score"]["delta"], 1.);

        for classification in [true, false] {
            let mut changed = expected.clone();
            if classification {
                changed.lines[0].classification_score = 0.5;
            } else {
                changed.lines[0].detection.score = 0.5;
            }
            assert!(compare_fast(&changed, &reference, true).is_err());
        }

        shifted(&mut actual, 1., 0.);
        actual.lines[0].classification_score = 0.5;
        actual.lines[0].detection.score = 0.5;
        actual.lines[0].recognition_content_width = 130;
        actual.lines[0].recognition_tensor_width = 160;
        actual.lines[0].recognition_time_steps = 10;
        assert!(compare_fast(&actual, &reference, true).is_ok());
        assert!(compare_mixed(&actual, &reference, true).is_err());
        let report = fast_differences(&actual, &reference, true).unwrap();
        assert_eq!(report["lines"][0]["recognition_score"]["actual"], 21.);
        assert_eq!(report["lines"][0]["recognition_score"]["expected"], 20.);
        assert_eq!(report["lines"][0]["recognition_score"]["delta"], 1.);
        assert_eq!(report["lines"][0]["classification_score"]["actual"], 0.5);
        assert_eq!(report["lines"][0]["detection_score"]["actual"], 0.5);
        for field in ["classification_score", "detection_score"] {
            assert!(report["lines"][0][field]["delta"].as_f64().unwrap() < -0.3);
        }
        assert_eq!(
            report["lines"][0]["numeric_score_comparison"]["applied"],
            false
        );
        assert_eq!(
            report["lines"][0]["numeric_score_comparison"]["reason"],
            "changed input geometry"
        );

        let mut same_box = expected.clone();
        same_box.lines[0].recognition_score += 1.;
        same_box.lines[0].recognition_content_width -= 1;
        assert!(compare_fast(&same_box, &reference, true).is_ok());
        // Changed inputs still require finite, bounded probabilities.
        for invalid in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
            let mut changed = actual.clone();
            changed.lines[0].classification_score = invalid;
            assert!(compare_fast(&changed, &reference, true).is_err());
            changed.lines[0].classification_score = 0.5;
            changed.lines[0].detection.score = invalid;
            assert!(compare_fast(&changed, &reference, true).is_err());
        }
        actual.lines[0].classification_score = expected.lines[0].classification_score;
        actual.lines[0].detection.score = expected.lines[0].detection.score;
        actual.lines[0].recognition_score = f32::NAN;
        assert!(compare_fast(&actual, &reference, true).is_err());
        actual.lines[0].recognition_score = 21.;
        shifted(&mut actual, 2., 0.);
        assert!(compare_fast(&actual, &reference, true).is_err());
    }

    #[test]
    fn fast_frozen_json_replay_uses_the_live_policy_and_diagnostics() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected;
        shifted(&mut actual, 1., 1.);
        actual.lines[0].recognition_score += 1.;
        actual.lines[0].classification_score = 0.5;
        actual.lines[0].detection.score = 0.5;
        let frozen = serde_json::to_value(&actual).unwrap();
        assert!(compare_fast(&actual, &reference, true).is_ok());
        assert!(compare_fast_values(&frozen, &reference, true).is_ok());
        assert_eq!(
            fast_differences(&actual, &reference, true).unwrap(),
            fast_differences_values(&frozen, &reference, true).unwrap()
        );
        let mut invalid = frozen;
        invalid["lines"][0]["detection"]["score"] = serde_json::json!(1.1);
        assert!(compare_fast_values(&invalid, &reference, true).is_err());
    }

    #[test]
    fn fast_rejects_invalid_internal_shapes_spans_and_token_concatenation() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        for case in 0..10 {
            let mut actual = expected.clone();
            let line = &mut actual.lines[0];
            match case {
                0 => line.recognition_content_width = line.recognition_tensor_width + 1,
                1 => line.recognition_time_steps = 0,
                2 => line.emitted_count = 9,
                3 => line.ctc_spans = None,
                4 => line.ctc_spans.as_mut().unwrap()[0].end_column = 0,
                5 => line.ctc_spans.as_mut().unwrap()[1].end_column = 9,
                6 => line.ctc_spans.as_mut().unwrap()[1].start_column = 1,
                7 => line.ctc_spans.as_mut().unwrap()[0].text = "other".into(),
                8 => line.ctc_spans.as_mut().unwrap()[0].score = f32::NAN,
                9 => line.ctc_spans.as_mut().unwrap().pop().map(|_| ()).unwrap(),
                _ => unreachable!(),
            }
            assert!(
                compare_fast(&actual, &reference, true).is_err(),
                "invalid case{case}"
            );
        }
        let mut actual = expected.clone();
        actual.lines[0].ctc_spans = None;
        assert!(compare_fast(&actual, &reference, false).is_ok());
        let mut actual = expected;
        actual.text.clear();
        actual.lines.clear();
        actual.detected_count = 0;
        assert!(compare_fast(&actual, &serde_json::to_value(&actual).unwrap(), true).is_ok());
    }
}
