//! Matched, real-model configuration qualification; never changes the serial goldens.
#![allow(dead_code)]
use crate::baseline::{self, CaseResult, Frame, Result};
use serde::{Deserialize, Serialize};

pub const REPRESENTATIVES: &[&str] = &["imgs_1", "imgs_00018069", "imgs_00006737"];
pub const ALIGNMENT_FIXTURES: &[&str] = &[
    "imgs_1",
    "imgs_00018069",
    "imgs_00006737",
    "imgs_11_rotate90",
    "imgs_11_rotate180",
    "blank",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub id: String,
    /// Zero uses the machine's available CPU count in both workers.
    pub capacity: usize,
    pub line_workers: usize,
    pub preprocess_workers: usize,
    pub detector_threads: usize,
    pub batch_size: usize,
    pub capture: bool,
    pub format: String,
    pub padding: usize,
    pub classification: bool,
    pub adaptive_width: bool,
    pub target_width: usize,
    pub default_thresholds: bool,
    pub dilation: bool,
    pub max_image_pixels: usize,
    pub fixtures: Vec<String>,
}
pub fn profiles(all: &[String]) -> Vec<Profile> {
    let base = Profile {
        id: "default-parallel".into(),
        capacity: 0,
        line_workers: 0,
        preprocess_workers: 0,
        detector_threads: 0,
        batch_size: 1,
        capture: false,
        format: "Bgr24".into(),
        padding: 0,
        classification: true,
        adaptive_width: true,
        target_width: 320,
        default_thresholds: false,
        dilation: false,
        max_image_pixels: 40_000_000,
        fixtures: REPRESENTATIVES.iter().map(|s| (*s).into()).collect(),
    };
    let mut full = base.clone();
    full.fixtures = all.to_vec();
    let mut out = vec![full];
    for (id, capacity) in [("serial", 1), ("workers-2", 2), ("workers-4", 4)] {
        let mut p = base.clone();
        p.id = id.into();
        p.capacity = capacity;
        p.line_workers = capacity;
        p.preprocess_workers = capacity;
        p.detector_threads = capacity;
        out.push(p);
    }
    for batch in [2, 4] {
        let mut p = base.clone();
        p.id = format!("batch-{batch}");
        p.batch_size = batch;
        p.fixtures = REPRESENTATIVES[1..].iter().map(|s| (*s).into()).collect();
        out.push(p);
    }
    let mut p = base.clone();
    p.id = "alignment".into();
    p.capture = true;
    p.fixtures = ALIGNMENT_FIXTURES.iter().map(|s| (*s).into()).collect();
    out.push(p);
    for (id, format) in [
        ("rgb-padded", "Rgb24"),
        ("bgra-padded", "Bgra32"),
        ("rgba-padded", "Rgba32"),
    ] {
        let mut p = base.clone();
        p.id = id.into();
        p.format = format.into();
        p.padding = 13;
        out.push(p);
    }
    let mut p = base.clone();
    p.id = "cls-off".into();
    p.classification = false;
    p.fixtures = vec![
        "imgs_11_rotate90".into(),
        "imgs_11_rotate180".into(),
        "blank".into(),
    ];
    out.push(p);
    let mut p = base.clone();
    p.id = "fixed-width".into();
    p.adaptive_width = false;
    out.push(p);
    let mut p = base.clone();
    p.id = "default-thresholds".into();
    p.default_thresholds = true;
    out.push(p);
    let mut p = base.clone();
    p.id = "dilation".into();
    p.dilation = true;
    out.push(p);
    let mut p = base.clone();
    p.id = "reuse".into();
    p.fixtures.push("imgs_1".into());
    out.push(p);
    let mut p = base;
    p.id = "recovery".into();
    p.max_image_pixels = 600_000;
    p.fixtures = vec![
        "imgs_00018069".into(),
        "imgs_00018069".into(),
        "imgs_00006737".into(),
        "imgs_1".into(),
        "imgs_00018069".into(),
    ];
    out.push(p);
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub id: String,
    pub fixture: String,
    pub path: String,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub format: String,
    pub expected_error: Option<String>,
}
pub fn pixels(frame: &Frame, profile: &Profile) -> Result<(Vec<u8>, usize)> {
    let bpp = match profile.format.as_str() {
        "Bgr24" | "Rgb24" => 3,
        "Bgra32" | "Rgba32" => 4,
        _ => return Err("unsupported pixel format".into()),
    };
    let width = frame.width as usize;
    let stride = width * bpp + profile.padding;
    // Poison padding and vary alpha to prove neither contributes to OCR.
    let mut out = vec![0xa5; stride * frame.height as usize];
    let swap = matches!(profile.format.as_str(), "Rgb24" | "Rgba32");
    for y in 0..frame.height as usize {
        for x in 0..width {
            let src = (y * width + x) * 3;
            let dst = y * stride + x * bpp;
            let p = &frame.pixels[src..src + 3];
            let rgb = [p[2], p[1], p[0]];
            out[dst..dst + 3].copy_from_slice(if swap { &rgb } else { p });
            if bpp == 4 {
                out[dst + 3] = ((x * 17 + y * 31) & 255) as u8;
            }
        }
    }
    Ok((out, stride))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub text: String,
    pub score: f64,
    pub start_column: usize,
    pub end_column: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterBox {
    pub text: String,
    pub score: f64,
    pub points: [[f64; 2]; 4],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alignment {
    pub spans: Option<Vec<Span>>,
    pub character_boxes: Option<Vec<CharacterBox>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub id: String,
    pub result: Option<CaseResult>,
    pub alignments: Vec<Alignment>,
    pub error: Option<String>,
}
pub fn validate(response: &Response, input: &Input, profile: &Profile) -> Result<()> {
    if response.id != input.id {
        return Err("response id mismatch".into());
    }
    if let Some(kind) = &input.expected_error {
        let message = response
            .error
            .as_deref()
            .ok_or("expected rejection was accepted")?
            .to_lowercase();
        let needle = match kind.as_str() {
            "invalid_stride" => "stride",
            "pixel_limit" => "pixel",
            _ => return Err("unknown expected rejection".into()),
        };
        if response.result.is_some() || !response.alignments.is_empty() || !message.contains(needle)
        {
            return Err(format!("wrong rejection for {kind}: {response:?}").into());
        }
        return Ok(());
    }
    if let Some(error) = &response.error {
        return Err(format!("unexpected rejection: {error}").into());
    }
    let result = response.result.as_ref().ok_or("missing result")?;
    result.validate()?;
    if result.id != input.id || response.alignments.len() != result.lines.len() {
        return Err("result/alignment count mismatch".into());
    }
    for (line, alignment) in result.lines.iter().zip(&response.alignments) {
        if !profile.capture {
            if alignment.spans.is_some() || alignment.character_boxes.is_some() {
                return Err("uncaptured alignment must be null".into());
            }
            continue;
        }
        let spans = alignment.spans.as_ref().ok_or("captured spans missing")?;
        let boxes = alignment
            .character_boxes
            .as_ref()
            .ok_or("captured character boxes missing")?;
        if spans.len() != line.emitted_count
            || boxes.len() != spans.len()
            || spans.iter().map(|s| s.text.as_str()).collect::<String>() != line.text
        {
            return Err("captured token/text/count mismatch".into());
        }
        let mut previous = 0;
        for (span, b) in spans.iter().zip(boxes) {
            if span.start_column < previous
                || span.end_column <= span.start_column
                || span.end_column > line.recognition_time_steps
                || !span.score.is_finite()
                || !b.score.is_finite()
                || !b.points.iter().flatten().all(|f| f.is_finite())
                || b.text != span.text
            {
                return Err("invalid captured token/character box".into());
            }
            previous = span.end_column;
        }
    }
    Ok(())
}
pub fn compare(
    model: &str,
    expected: &Response,
    actual: &Response,
    input: &Input,
    profile: &Profile,
) -> Vec<String> {
    let mut errors = Vec::new();
    for (language, result) in [("C#", expected), ("Rust", actual)] {
        if let Err(error) = validate(result, input, profile) {
            errors.push(format!(
                "{model}/{}/{} {language}: {error}",
                profile.id, input.id
            ));
        }
    }
    let (Some(e), Some(a)) = (&expected.result, &actual.result) else {
        return errors;
    };
    errors.extend(baseline::compare(model, e, a));
    for (i, (e, a)) in expected
        .alignments
        .iter()
        .zip(&actual.alignments)
        .enumerate()
    {
        let path = format!("{model}/{}/{} line[{i}]", profile.id, input.id);
        match (&e.spans, &a.spans) {
            (Some(e), Some(a)) => {
                if e.len() != a.len() {
                    errors.push(format!("{path} span count differs"));
                }
                for (j, (e, a)) in e.iter().zip(a).enumerate() {
                    if (e.text.as_str(), e.start_column, e.end_column)
                        != (a.text.as_str(), a.start_column, a.end_column)
                    {
                        errors.push(format!("{path} span[{j}] token/time columns differ"));
                    }
                    score(
                        &mut errors,
                        &format!("{path} span[{j}].score"),
                        e.score,
                        a.score,
                    );
                }
            }
            (None, None) => (),
            _ => errors.push(format!("{path} span capture differs")),
        }
        match (&e.character_boxes, &a.character_boxes) {
            (Some(e), Some(a)) => {
                if e.len() != a.len() {
                    errors.push(format!("{path} character box count differs"));
                }
                for (j, (e, a)) in e.iter().zip(a).enumerate() {
                    if e.text != a.text {
                        errors.push(format!("{path} character[{j}] token differs"));
                    }
                    score(
                        &mut errors,
                        &format!("{path} character[{j}].score"),
                        e.score,
                        a.score,
                    );
                    for p in 0..4 {
                        for xy in 0..2 {
                            let (ev, av) = (e.points[p][xy], a.points[p][xy]);
                            if !ev.is_finite()
                                || !av.is_finite()
                                || (ev - av).abs() > baseline::COORDINATE_TOLERANCE
                            {
                                errors.push(format!("{path} character[{j}].points[{p}][{xy}]: expected {ev}, actual {av}"));
                            }
                        }
                    }
                }
            }
            (None, None) => (),
            _ => errors.push(format!("{path} character capture differs")),
        }
    }
    errors
}
fn score(errors: &mut Vec<String>, path: &str, expected: f64, actual: f64) {
    if !expected.is_finite()
        || !actual.is_finite()
        || (actual - expected).abs()
            > baseline::SCORE_ABSOLUTE + baseline::SCORE_RELATIVE * expected.abs()
    {
        errors.push(format!("{path}: expected {expected}, actual {actual}"));
    }
}
