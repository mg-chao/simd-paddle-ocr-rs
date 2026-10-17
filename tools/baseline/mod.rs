//! Development-only baseline contract, shared by the CLI and focused tests.
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const SCHEMA: u32 = 1;
pub const COORDINATE_TOLERANCE: f64 = 0.5;
pub const SCORE_ABSOLUTE: f64 = 1e-4;
pub const SCORE_RELATIVE: f64 = 1e-4;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub cpu_threads: usize,
    pub kernel_policy: String,
    pub layout: String,
    pub fusion: bool,
    pub direction_classification: bool,
    pub classifier_threshold: f32,
    pub recognition_batch_size: usize,
    pub adaptive_width: bool,
    pub target_width: usize,
    pub detector_limit: usize,
    pub bitmap_threshold: f32,
    pub unclip_ratio: f32,
    pub max_candidates: usize,
    pub use_dilation: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    pub det: String,
    pub rec: String,
    pub dictionary: String,
    pub cls: String,
    pub box_threshold: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub file: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256_crlf: Option<String>,
}
impl Asset {
    pub fn check(&self, path: &Path) -> Result<()> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let actual = hash(&bytes);
        require(
            actual == self.sha256 || self.sha256_crlf.as_deref() == Some(actual.as_str()),
            format!(
                "{}: SHA-256 mismatch: expected {}{}; got {actual}",
                path.display(),
                self.sha256,
                self.sha256_crlf
                    .as_ref()
                    .map(|s| format!(" or {s} (CRLF)"))
                    .unwrap_or_default()
            ),
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub id: String,
    pub file: Option<String>,
    pub url: Option<String>,
    pub sha256: Option<String>,
    pub transform: String,
    pub base: Option<String>,
    pub width: u32,
    pub height: u32,
    pub pixel_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub reference_commit: String,
    pub corpus_commit: String,
    pub settings: Settings,
    pub models: Vec<ModelSpec>,
    pub assets: Vec<Asset>,
    pub fixtures: Vec<Fixture>,
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}
fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn local_path(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn check_file(path: &Path, expected: &str) -> Result<()> {
    require(
        hex(expected, 64),
        format!("{}: invalid SHA-256", path.display()),
    )?;
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let actual = hash(&bytes);
    require(
        actual == expected,
        format!(
            "{}: SHA-256 mismatch: expected {expected}, got {actual}",
            path.display()
        ),
    )
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()).into())
}
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    fs::write(path, text)?;
    Ok(())
}
impl Manifest {
    pub fn validate(&self) -> Result<()> {
        require(self.schema_version == SCHEMA, "unsupported manifest schema")?;
        require(
            hex(&self.reference_commit, 40) && hex(&self.corpus_commit, 40),
            "invalid source commit",
        )?;
        let s = &self.settings;
        require(
            s.cpu_threads == 1 && s.kernel_policy == "auto" && s.layout == "auto" && s.fusion,
            "baseline requires one CPU thread, automatic kernels/layout and fusion",
        )?;
        require(
            s.direction_classification && s.recognition_batch_size == 1 && s.adaptive_width,
            "baseline requires classification and adaptive per-line recognition",
        )?;
        require(
            self.models
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>()
                == ["tiny", "small", "medium"],
            "model coverage must be tiny, small, medium in that order",
        )?;
        let asset_names: BTreeSet<_> = self.assets.iter().map(|a| a.file.as_str()).collect();
        require(
            self.assets.len() == 9 && asset_names.len() == 9,
            "expected nine unique model/dictionary assets",
        )?;
        for asset in &self.assets {
            require(
                hex(&asset.sha256, 64)
                    && asset.sha256_crlf.as_ref().is_none_or(|s| {
                        ["ppocr_keys.txt", "rec_keys.txt"].contains(&asset.file.as_str())
                            && hex(s, 64)
                    }),
                format!("{}: invalid asset checksum declaration", asset.file),
            )?;
        }
        let mut used = BTreeSet::new();
        for m in &self.models {
            for name in [&m.det, &m.rec, &m.dictionary, &m.cls] {
                require(
                    local_path(name) && asset_names.contains(name.as_str()),
                    format!("{}: unpinned asset {name}", m.id),
                )?;
                used.insert(name.as_str());
            }
        }
        require(
            used == asset_names,
            "asset manifest does not match model bundles",
        )?;
        require(
            self.fixtures.len() == 22,
            "corpus must contain all 22 fixtures",
        )?;
        let mut seen = BTreeSet::new();
        let mut original_count = 0;
        let mut transforms = BTreeSet::new();
        for f in &self.fixtures {
            require(
                !f.id.is_empty()
                    && f.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    && seen.insert(f.id.as_str()),
                format!("invalid or duplicate fixture id {}", f.id),
            )?;
            if f.transform == "identity" {
                original_count += 1;
                require(
                    f.file
                        .as_ref()
                        .is_some_and(|p| local_path(p) && p.starts_with("corpus/"))
                        && f.base.is_none(),
                    "invalid original fixture",
                )?;
                require(
                    f.sha256.as_deref().is_some_and(|s| hex(s, 64)),
                    "missing source checksum",
                )?;
                require(
                    f.url.as_ref().is_some_and(|u| {
                        u.starts_with(&format!(
                            "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/{}/doc/",
                            self.corpus_commit
                        ))
                    }),
                    "fixture URL is not pinned to the corpus commit",
                )?;
            } else {
                require(
                    f.file.is_none()
                        && f.url.is_none()
                        && f.sha256.is_none()
                        && transforms.insert(f.transform.as_str()),
                    "invalid derived fixture",
                )?;
                match f.transform.as_str() {
                    "rotate90" | "rotate180" => require(
                        f.base.as_deref() == Some("imgs_11"),
                        "rotations must derive from imgs_11",
                    )?,
                    "white" => require(
                        f.base.is_none() && f.width == 512 && f.height == 512,
                        "blank must be white 512x512",
                    )?,
                    _ => return Err(format!("unknown transform {}", f.transform).into()),
                }
            }
        }
        require(
            original_count == 19
                && seen.contains("imgs_11")
                && transforms == BTreeSet::from(["rotate90", "rotate180", "white"]),
            "incomplete original/derived corpus coverage",
        )
    }
    pub fn check_assets(&self, directory: &Path) -> Result<()> {
        self.validate()?;
        for a in &self.assets {
            a.check(&directory.join(&a.file))?;
        }
        Ok(())
    }
}

pub struct Frame {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}
pub fn decode_frames(manifest: &Manifest, root: &Path, check_pixels: bool) -> Result<Vec<Frame>> {
    manifest.validate()?;
    let mut originals = BTreeMap::new();
    for f in manifest
        .fixtures
        .iter()
        .filter(|f| f.transform == "identity")
    {
        let path = root.join(f.file.as_ref().unwrap());
        check_file(&path, f.sha256.as_ref().unwrap())?;
        originals.insert(
            f.id.as_str(),
            image::ImageReader::open(path)?
                .with_guessed_format()?
                .decode()?
                .to_rgb8(),
        );
    }
    let mut frames = Vec::new();
    for f in &manifest.fixtures {
        let rgb = match f.transform.as_str() {
            "identity" => originals[&f.id.as_str()].clone(),
            "rotate90" => image::imageops::rotate90(&originals[&f.base.as_deref().unwrap()]),
            "rotate180" => image::imageops::rotate180(&originals[&f.base.as_deref().unwrap()]),
            "white" => image::RgbImage::from_pixel(512, 512, image::Rgb([255, 255, 255])),
            _ => unreachable!(),
        };
        let (width, height) = rgb.dimensions();
        let mut pixels = rgb.into_raw();
        for p in pixels.as_chunks_mut::<3>().0 {
            p.swap(0, 2);
        }
        if check_pixels {
            require(
                (f.width, f.height) == (width, height),
                format!("{}: decoded dimensions changed", f.id),
            )?;
            require(
                hash(&pixels) == f.pixel_sha256,
                format!("{}: decoded BGR24 SHA-256 mismatch", f.id),
            )?;
        }
        frames.push(Frame {
            id: f.id.clone(),
            width,
            height,
            pixels,
        });
    }
    Ok(frames)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseInput {
    pub id: String,
    pub path: String,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Line {
    pub points: [[f64; 2]; 4],
    pub detection_score: f64,
    pub text: String,
    pub recognition_score: f64,
    pub classification_score: f64,
    pub classification_label: u32,
    pub applied_rotation_degrees: u16,
    pub emitted_count: usize,
    pub recognition_content_width: usize,
    pub recognition_tensor_width: usize,
    pub recognition_time_steps: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub id: String,
    pub text: String,
    pub detected_count: usize,
    pub detector_resized_width: usize,
    pub detector_resized_height: usize,
    pub lines: Vec<Line>,
}
impl CaseResult {
    pub fn validate(&self) -> Result<()> {
        require(
            self.detected_count == self.lines.len(),
            format!("{}: detected/line count mismatch", self.id),
        )?;
        require(
            self.detector_resized_width > 0 && self.detector_resized_height > 0,
            "invalid detector dimensions",
        )?;
        require(
            self.text
                == self
                    .lines
                    .iter()
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            "joined text differs from lines",
        )?;
        for (i, l) in self.lines.iter().enumerate() {
            require(
                l.points.iter().flatten().all(|v| v.is_finite())
                    && [
                        l.detection_score,
                        l.recognition_score,
                        l.classification_score,
                    ]
                    .iter()
                    .all(|v| v.is_finite()),
                format!("{} line {i}: non-finite float", self.id),
            )?;
            require(
                l.recognition_content_width > 0
                    && l.recognition_tensor_width >= l.recognition_content_width
                    && l.recognition_time_steps > 0,
                format!("{} line {i}: invalid recognition dimensions", self.id),
            )?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Golden {
    pub schema_version: u32,
    pub model: String,
    pub reference_commit: String,
    pub manifest_sha256: String,
    pub settings: Settings,
    pub provenance: BTreeMap<String, String>,
    pub cases: Vec<CaseResult>,
}
impl Golden {
    pub fn validate(&self, model: &str, manifest: &Manifest, manifest_hash: &str) -> Result<()> {
        require(
            self.schema_version == SCHEMA
                && self.model == model
                && self.reference_commit == manifest.reference_commit
                && self.manifest_sha256 == manifest_hash
                && self.settings == manifest.settings,
            format!("{model}: baseline provenance/configuration mismatch"),
        )?;
        require(
            self.cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
                == manifest
                    .fixtures
                    .iter()
                    .map(|f| f.id.as_str())
                    .collect::<Vec<_>>(),
            format!("{model}: baseline corpus coverage/order mismatch"),
        )?;
        for case in &self.cases {
            case.validate()?;
        }
        Ok(())
    }
}

pub fn compare(model: &str, expected: &CaseResult, actual: &CaseResult) -> Vec<String> {
    let mut errors = Vec::new();
    let prefix = format!("{model}/{}", expected.id);
    if let Err(e) = expected.validate() {
        errors.push(format!("{prefix}: invalid expected result: {e}"));
    }
    if let Err(e) = actual.validate() {
        errors.push(format!("{prefix}: invalid actual result: {e}"));
    }
    macro_rules! exact {
        ($a:expr, $b:expr, $field:expr) => {
            if $a != $b {
                errors.push(format!(
                    "{} {}: expected {:?}, actual {:?}",
                    prefix, $field, $a, $b
                ));
            }
        };
    }
    exact!(expected.id, actual.id, "id");
    exact!(expected.text, actual.text, "text");
    exact!(
        expected.detected_count,
        actual.detected_count,
        "detected_count"
    );
    exact!(
        expected.detector_resized_width,
        actual.detector_resized_width,
        "detector_resized_width"
    );
    exact!(
        expected.detector_resized_height,
        actual.detector_resized_height,
        "detector_resized_height"
    );
    exact!(expected.lines.len(), actual.lines.len(), "lines.length");
    for (i, (e, a)) in expected.lines.iter().zip(&actual.lines).enumerate() {
        macro_rules! field {
            ($name:ident) => {
                exact!(e.$name, a.$name, format!("line[{i}].{}", stringify!($name)));
            };
        }
        field!(text);
        field!(classification_label);
        field!(applied_rotation_degrees);
        field!(emitted_count);
        field!(recognition_content_width);
        field!(recognition_tensor_width);
        field!(recognition_time_steps);
        for (name, ev, av) in [
            ("detection_score", e.detection_score, a.detection_score),
            (
                "recognition_score",
                e.recognition_score,
                a.recognition_score,
            ),
            (
                "classification_score",
                e.classification_score,
                a.classification_score,
            ),
        ] {
            float_difference(
                &mut errors,
                &format!("{prefix} line[{i}].{name}"),
                ev,
                av,
                SCORE_ABSOLUTE + SCORE_RELATIVE * ev.abs(),
            );
        }
        for p in 0..4 {
            for xy in 0..2 {
                float_difference(
                    &mut errors,
                    &format!("{prefix} line[{i}].points[{p}][{xy}]"),
                    e.points[p][xy],
                    a.points[p][xy],
                    COORDINATE_TOLERANCE,
                );
            }
        }
    }
    errors
}
fn float_difference(
    errors: &mut Vec<String>,
    path: &str,
    expected: f64,
    actual: f64,
    tolerance: f64,
) {
    if !expected.is_finite() || !actual.is_finite() || (actual - expected).abs() > tolerance {
        errors.push(format!(
            "{path}: expected {expected}, actual {actual}, delta {}, tolerance {tolerance}",
            (actual - expected).abs()
        ));
    }
}
