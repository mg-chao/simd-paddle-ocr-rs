//! Source-image token boxes estimated from captured CTC alignment.
use crate::{
    crop::CropTransform,
    detector::DetectionBox,
    image::{ImageError, ImageResult},
    ocr::{OcrLineView, OwnedOcrLine},
    recognizer::{CtcSpans, RecognitionView, copy_text},
};
use serde::Serialize;

/// One CTC token, in the point order of the crop the recognizer read.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct CharacterBox<'a> {
    pub text: &'a str,
    pub score: f32,
    pub points: [[f32; 2]; 4],
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OwnedCharacterBox {
    pub text: String,
    pub score: f32,
    pub points: [[f32; 2]; 4],
}
impl CharacterBox<'_> {
    pub fn try_to_owned(self) -> ImageResult<OwnedCharacterBox> {
        Ok(OwnedCharacterBox {
            text: copy_text(self.text)?,
            score: self.score,
            points: self.points,
        })
    }
}
impl OwnedCharacterBox {
    pub fn as_view(&self) -> CharacterBox<'_> {
        CharacterBox {
            text: &self.text,
            score: self.score,
            points: self.points,
        }
    }
}

/// Map fired token runs onto a detection quad, splitting intervening blank gaps
/// at their midpoints. Alignment must have been captured, even for an empty line.
#[allow(clippy::too_many_arguments)]
pub fn estimate<'a>(
    detection: &DetectionBox,
    spans: Option<CtcSpans<'a>>,
    time_steps: usize,
    content_width: usize,
    tensor_width: usize,
    applied_rotation_degrees: u16,
) -> ImageResult<Vec<CharacterBox<'a>>> {
    let spans = spans
        .ok_or_else(|| ImageError::new("CTC alignment was not captured; run with capture: true"))?;
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    if time_steps == 0
        || content_width == 0
        || tensor_width < content_width
        || !matches!(applied_rotation_degrees, 0 | 180)
    {
        return Err(ImageError::new(
            "invalid character-box dimensions or rotation",
        ));
    }
    let transform = CropTransform::new(detection)?;
    let scale = tensor_width as f64 / content_width as f64 / time_steps as f64;
    let mut result = Vec::new();
    result
        .try_reserve_exact(spans.len())
        .map_err(|e| ImageError::new(format!("character-box allocation: {e}")))?;
    for (index, span) in spans.iter().enumerate() {
        if span.start_column >= time_steps
            || span.end_column <= span.start_column
            || span.end_column > time_steps
        {
            return Err(ImageError::new(
                "CTC span is outside the recognition time axis",
            ));
        }
        let mut left = span.start_column as f64;
        let mut right = span.end_column as f64;
        if let Some(previous) = index.checked_sub(1).and_then(|i| spans.get(i)) {
            if previous.end_column > span.start_column {
                return Err(ImageError::new("CTC spans overlap or are out of order"));
            }
            left = (previous.end_column as f64 + left) / 2.;
        }
        if let Some(next) = spans.get(index + 1) {
            if span.end_column > next.start_column {
                return Err(ImageError::new("CTC spans overlap or are out of order"));
            }
            right = (right + next.start_column as f64) / 2.;
        }
        let (mut left, mut right) = (left * scale, right * scale);
        if applied_rotation_degrees == 180 {
            (left, right) = (1. - right, 1. - left);
        }
        left = left.clamp(0., 1.);
        right = right.clamp(0., 1.).max(left);
        result.push(CharacterBox {
            text: span.text,
            score: span.score,
            points: transform.map_span(left, right)?,
        });
    }
    Ok(result)
}

pub fn estimate_recognition<'a>(
    detection: &DetectionBox,
    recognition: RecognitionView<'a>,
    rotation: u16,
) -> ImageResult<Vec<CharacterBox<'a>>> {
    estimate(
        detection,
        recognition.ctc_spans,
        recognition.time_steps,
        recognition.resized_width,
        recognition.tensor_width,
        rotation,
    )
}
impl<'a> OcrLineView<'a> {
    pub fn estimate_character_boxes(self) -> ImageResult<Vec<CharacterBox<'a>>> {
        estimate(
            &self.detection,
            self.ctc_spans,
            self.recognition_time_steps,
            self.recognition_content_width,
            self.recognition_tensor_width,
            self.applied_rotation_degrees,
        )
    }
}
impl OwnedOcrLine {
    pub fn estimate_character_boxes(&self) -> ImageResult<Vec<CharacterBox<'_>>> {
        self.as_view().estimate_character_boxes()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::recognizer::OwnedCtcSpan;
    fn axis(width: f32, height: f32) -> DetectionBox {
        DetectionBox {
            points: [[0., 0.], [width, 0.], [width, height], [0., height]],
            score: 1.,
        }
    }
    fn spans(ranges: &[(usize, usize)]) -> Vec<OwnedCtcSpan> {
        ranges
            .iter()
            .map(|&(start_column, end_column)| OwnedCtcSpan {
                text: "中".into(),
                score: 0.9,
                start_column,
                end_column,
            })
            .collect()
    }
    fn check(actual: &CharacterBox<'_>, expected: [[f32; 2]; 4]) {
        for (actual, expected) in actual
            .points
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
        {
            assert!((actual - expected).abs() < 0.02, "{actual} != {expected}");
        }
    }
    #[test]
    fn reference_midpoint_gap_padding_and_rotation_goldens() {
        let s = spans(&[(0, 2), (4, 6)]);
        let boxes = estimate(&axis(80., 20.), Some(CtcSpans::from_owned(&s)), 8, 8, 8, 0).unwrap();
        check(&boxes[0], [[0., 0.], [30., 0.], [30., 20.], [0., 20.]]);
        check(&boxes[1], [[30., 0.], [60., 0.], [60., 20.], [30., 20.]]);
        let s = spans(&[(0, 1), (4, 5)]);
        let boxes = estimate(&axis(80., 10.), Some(CtcSpans::from_owned(&s)), 8, 8, 8, 0).unwrap();
        assert_eq!(boxes[0].points[1], [25., 0.]);
        let s = spans(&[(0, 1)]);
        check(
            &estimate(
                &axis(100., 10.),
                Some(CtcSpans::from_owned(&s)),
                10,
                5,
                10,
                0,
            )
            .unwrap()[0],
            [[0., 0.], [20., 0.], [20., 10.], [0., 10.]],
        );
        let s = spans(&[(0, 2)]);
        check(
            &estimate(
                &axis(100., 10.),
                Some(CtcSpans::from_owned(&s)),
                10,
                10,
                10,
                180,
            )
            .unwrap()[0],
            [[80., 0.], [100., 0.], [100., 10.], [80., 10.]],
        );
    }
    #[test]
    fn reference_vertical_and_perspective_goldens() {
        let tall = DetectionBox {
            points: [[10., 10.], [30., 10.], [30., 90.], [10., 90.]],
            score: 1.,
        };
        let s = spans(&[(0, 1)]);
        check(
            &estimate(&tall, Some(CtcSpans::from_owned(&s)), 4, 4, 4, 0).unwrap()[0],
            [[10., 90.], [10., 70.], [30., 70.], [30., 90.]],
        );
        check(
            &estimate(&tall, Some(CtcSpans::from_owned(&s)), 4, 4, 4, 180).unwrap()[0],
            [[10., 30.], [10., 10.], [30., 10.], [30., 30.]],
        );
        let quad = DetectionBox {
            points: [[0., 0.], [100., 0.], [80., 40.], [10., 40.]],
            score: 1.,
        };
        let s = spans(&[(0, 4)]);
        let boxes = estimate(&quad, Some(CtcSpans::from_owned(&s)), 4, 4, 4, 0).unwrap();
        check(&boxes[0], quad.points);
        assert_eq!(boxes[0].try_to_owned().unwrap().as_view(), boxes[0]);
        let slant = DetectionBox {
            points: [[0., 0.], [80., 16.], [80., 36.], [0., 20.]],
            score: 1.,
        };
        let s = spans(&[(2, 4)]);
        check(
            &estimate(&slant, Some(CtcSpans::from_owned(&s)), 8, 8, 8, 0).unwrap()[0],
            [[20., 4.], [40., 8.], [40., 28.], [20., 24.]],
        );
    }
    #[test]
    fn rejects_missing_overlapping_and_invalid_alignment() {
        assert!(estimate(&axis(40., 10.), None, 4, 4, 4, 0).is_err());
        assert!(
            estimate(
                &axis(40., 10.),
                Some(CtcSpans::from_owned(&[])),
                0,
                0,
                0,
                90
            )
            .unwrap()
            .is_empty()
        );
        for ranges in [
            &[(0, 9)][..],
            &[(1, 1)][..],
            &[(0, 3), (2, 4)][..],
            &[(3, 4), (0, 1)][..],
        ] {
            let s = spans(ranges);
            assert!(estimate(&axis(40., 10.), Some(CtcSpans::from_owned(&s)), 4, 4, 4, 0).is_err());
        }
        let s = spans(&[(0, 1)]);
        for (steps, content, tensor, rotation) in
            [(0, 4, 4, 0), (4, 0, 4, 0), (4, 5, 4, 0), (4, 4, 4, 90)]
        {
            assert!(
                estimate(
                    &axis(40., 10.),
                    Some(CtcSpans::from_owned(&s)),
                    steps,
                    content,
                    tensor,
                    rotation
                )
                .is_err()
            );
        }
    }
}
