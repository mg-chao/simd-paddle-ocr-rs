//! Validation and stage-by-stage comparison of version 1 OCR reference bundles.

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fmt, fs,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            atol: 1e-5,
            rtol: 1e-4,
        }
    }
}

#[derive(Debug)]
pub struct BundleError(String);

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BundleError {}

type Result<T> = std::result::Result<T, BundleError>;

fn invalid(message: impl Into<String>) -> BundleError {
    BundleError(message.into())
}

#[derive(Debug)]
pub struct ComparisonReport {
    pub checked_cases: usize,
    pub checked_artifacts: usize,
    pub first_difference: Option<String>,
}

impl ComparisonReport {
    pub fn matches(&self) -> bool {
        self.first_difference.is_none()
    }
}

impl fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(difference) = &self.first_difference {
            write!(f, "DIFFERENCE: {difference}")
        } else {
            write!(
                f,
                "MATCH: {} cases, {} artifacts; all model/artifact hashes and lengths verified",
                self.checked_cases, self.checked_artifacts
            )
        }
    }
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    contract: Value,
    models: Vec<Model>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Model {
    id: String,
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    kind: String,
    artifacts: Vec<Artifact>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Dtype {
    F32,
    U8,
    Json,
}

#[derive(Deserialize)]
struct Artifact {
    id: String,
    path: String,
    dtype: Dtype,
    shape: Vec<usize>,
    layout: String,
    sha256: String,
    bytes: u64,
}

struct Bundle {
    root: PathBuf,
    manifest: Manifest,
}

enum Payload {
    F32(Vec<f32>),
    U8(Vec<u8>),
    Json(Value),
}

impl Bundle {
    fn load(directory: &Path) -> Result<Self> {
        let root = directory
            .canonicalize()
            .map_err(|e| invalid(format!("{}: {e}", directory.display())))?;
        if !root.is_dir() {
            return Err(invalid(format!(
                "{} is not a directory",
                directory.display()
            )));
        }
        let manifest_path = safe_file(&root, "manifest.json")?;
        let bytes = fs::read(&manifest_path)
            .map_err(|e| invalid(format!("{}: {e}", manifest_path.display())))?;
        let manifest: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| invalid(format!("{}: {e}", manifest_path.display())))?;
        if manifest.schema_version != 1 {
            return Err(invalid(format!(
                "{}: unsupported schema_version {}",
                root.display(),
                manifest.schema_version
            )));
        }
        if !manifest.contract.is_object() {
            return Err(invalid(format!(
                "{}: contract must be an object",
                root.display()
            )));
        }
        unique_ids(
            manifest.models.iter().map(|m| m.id.as_str()),
            "model",
            &root,
        )?;
        unique_ids(manifest.cases.iter().map(|c| c.id.as_str()), "case", &root)?;
        let bundle = Self { root, manifest };
        for model in &bundle.manifest.models {
            read_verified(&bundle.root, &model.path, &model.sha256, model.bytes)?;
        }
        for case in &bundle.manifest.cases {
            if case.kind.is_empty() {
                return Err(invalid(format!("case {}: kind is empty", case.id)));
            }
            unique_ids(
                case.artifacts.iter().map(|a| a.id.as_str()),
                &format!("artifact in case {}", case.id),
                &bundle.root,
            )?;
            for artifact in &case.artifacts {
                bundle.read_artifact(artifact)?;
            }
        }
        Ok(bundle)
    }

    fn read_artifact(&self, artifact: &Artifact) -> Result<Payload> {
        let context = format!("{} ({})", artifact.id, artifact.path);
        if artifact.layout.is_empty() {
            return Err(invalid(format!("{context}: layout is empty")));
        }
        if artifact.dtype == Dtype::Json {
            if !artifact.shape.is_empty() || artifact.layout != "json" {
                return Err(invalid(format!(
                    "{context}: JSON artifacts require shape=[] and layout=json"
                )));
            }
        } else {
            let elements = artifact
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| invalid(format!("{context}: shape element count overflow")))?;
            let element_bytes = if artifact.dtype == Dtype::F32 { 4 } else { 1 };
            let expected = elements
                .checked_mul(element_bytes)
                .ok_or_else(|| invalid(format!("{context}: shape byte count overflow")))?;
            if u64::try_from(expected).ok() != Some(artifact.bytes) {
                return Err(invalid(format!(
                    "{context}: shape implies {expected} bytes, manifest records {}",
                    artifact.bytes
                )));
            }
        }
        let bytes = read_verified(&self.root, &artifact.path, &artifact.sha256, artifact.bytes)?;
        match artifact.dtype {
            Dtype::F32 => {
                let values: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte chunk")))
                    .collect();
                if let Some(index) = values.iter().position(|v| !v.is_finite()) {
                    return Err(invalid(format!(
                        "{context}: nonfinite f32 at index {index}"
                    )));
                }
                Ok(Payload::F32(values))
            }
            Dtype::U8 => Ok(Payload::U8(bytes)),
            Dtype::Json => serde_json::from_slice(&bytes)
                .map(Payload::Json)
                .map_err(|e| invalid(format!("{context}: invalid JSON: {e}"))),
        }
    }
}

fn unique_ids<'a>(ids: impl Iterator<Item = &'a str>, kind: &str, root: &Path) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        if id.is_empty() || !seen.insert(id) {
            return Err(invalid(format!(
                "{}: empty or duplicate {kind} ID {id:?}",
                root.display()
            )));
        }
    }
    Ok(())
}

fn safe_file(root: &Path, relative: &str) -> Result<PathBuf> {
    // Check both separators even on Unix, so a bundle is portable and cannot hide
    // Windows drive prefixes or traversal in a manifest created on another OS.
    if relative.is_empty()
        || relative.starts_with(['/', '\\'])
        || relative
            .split(['/', '\\'])
            .any(|part| !portable_segment(part))
    {
        return Err(invalid(format!("unsafe bundle path {relative:?}")));
    }
    let portable = relative.replace('\\', "/");
    let path = Path::new(&portable);
    if path
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid(format!("unsafe bundle path {relative:?}")));
    }
    let resolved = root
        .join(path)
        .canonicalize()
        .map_err(|e| invalid(format!("{}: {e}", root.join(path).display())))?;
    if !resolved.starts_with(root) || !resolved.is_file() {
        return Err(invalid(format!(
            "bundle path {relative:?} resolves outside its bundle or is not a file"
        )));
    }
    Ok(resolved)
}

fn portable_segment(segment: &str) -> bool {
    if segment.is_empty()
        || segment == "."
        || segment == ".."
        || segment.ends_with(['.', ' '])
        || segment
            .chars()
            .any(|c| c < '\u{20}' || "<>:\"|?*".contains(c))
    {
        return false;
    }
    let stem = segment.split('.').next().expect("nonempty path segment");
    let upper = stem.to_ascii_uppercase();
    !matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && matches!(upper.as_bytes()[3], b'1'..=b'9'))
}

fn read_verified(
    root: &Path,
    path: &str,
    expected_sha: &str,
    expected_bytes: u64,
) -> Result<Vec<u8>> {
    if expected_sha.len() != 64 || !expected_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid(format!(
            "{path}: sha256 must contain 64 hexadecimal digits"
        )));
    }
    let resolved = safe_file(root, path)?;
    let bytes = fs::read(&resolved).map_err(|e| invalid(format!("{}: {e}", resolved.display())))?;
    if bytes.len() as u64 != expected_bytes {
        return Err(invalid(format!(
            "{}: byte length {} differs from manifest {expected_bytes}",
            resolved.display(),
            bytes.len()
        )));
    }
    let actual_sha = format!("{:x}", Sha256::digest(&bytes));
    if !actual_sha.eq_ignore_ascii_case(expected_sha) {
        return Err(invalid(format!(
            "{}: sha256 mismatch (expected {expected_sha}, actual {actual_sha})",
            resolved.display()
        )));
    }
    Ok(bytes)
}

/// Validate both complete bundles, then report the first differing pipeline stage.
/// Provenance, performance, and other unknown manifest fields are deliberately ignored.
pub fn compare_bundles(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    if !tolerance.atol.is_finite()
        || tolerance.atol < 0.0
        || !tolerance.rtol.is_finite()
        || tolerance.rtol < 0.0
    {
        return Err(invalid("tolerances must be finite and nonnegative"));
    }
    let reference = Bundle::load(reference)?;
    let candidate = Bundle::load(candidate)?;
    let mut report = ComparisonReport {
        checked_cases: 0,
        checked_artifacts: 0,
        first_difference: None,
    };
    report.first_difference = compare_validated(&reference, &candidate, tolerance, &mut report)?;
    Ok(report)
}

fn compare_validated(
    reference: &Bundle,
    candidate: &Bundle,
    tolerance: Tolerance,
    report: &mut ComparisonReport,
) -> Result<Option<String>> {
    let r = &reference.manifest;
    let c = &candidate.manifest;
    if r.contract != c.contract {
        return Ok(Some("manifest contract differs".into()));
    }
    if r.models.len() != c.models.len() {
        return Ok(Some(format!(
            "model count differs: reference {}, candidate {}",
            r.models.len(),
            c.models.len()
        )));
    }
    for model in &r.models {
        let Some(other) = c.models.iter().find(|m| m.id == model.id) else {
            return Ok(Some(format!("missing model {}", model.id)));
        };
        if !model.sha256.eq_ignore_ascii_case(&other.sha256) {
            return Ok(Some(format!("model {} sha256 differs", model.id)));
        }
    }
    for (index, case) in r.cases.iter().enumerate() {
        let Some(other) = c.cases.get(index) else {
            return Ok(Some(format!("case {}: missing from candidate", case.id)));
        };
        if case.id != other.id {
            return Ok(Some(format!(
                "case order/ID differs at index {index}: reference {}, candidate {}",
                case.id, other.id
            )));
        }
        if case.kind != other.kind {
            return Ok(Some(format!(
                "case {}: kind differs ({} vs {})",
                case.id, case.kind, other.kind
            )));
        }
        for (stage, artifact) in case.artifacts.iter().enumerate() {
            let Some(other_artifact) = other.artifacts.get(stage) else {
                return Ok(Some(format!(
                    "case {}, artifact {}: missing from candidate",
                    case.id, artifact.id
                )));
            };
            let context = format!("case {}, artifact {}", case.id, artifact.id);
            if artifact.id != other_artifact.id {
                return Ok(Some(format!(
                    "{context}: stage order/ID differs at index {stage} (candidate {})",
                    other_artifact.id
                )));
            }
            if artifact.dtype != other_artifact.dtype
                || artifact.shape != other_artifact.shape
                || artifact.layout != other_artifact.layout
            {
                return Ok(Some(format!(
                    "{context}: metadata differs; reference {:?} {:?} {}, candidate {:?} {:?} {}",
                    artifact.dtype,
                    artifact.shape,
                    artifact.layout,
                    other_artifact.dtype,
                    other_artifact.shape,
                    other_artifact.layout
                )));
            }
            let payload = reference.read_artifact(artifact)?;
            let other_payload = candidate.read_artifact(other_artifact)?;
            let difference = match (&payload, &other_payload) {
                (Payload::F32(a), Payload::F32(b)) => compare_f32(a, b, tolerance),
                (Payload::U8(a), Payload::U8(b)) => compare_bytes(a, b),
                (Payload::Json(a), Payload::Json(b)) => compare_json(a, b, tolerance, "$"),
                _ => unreachable!("metadata comparison ensures matching payload types"),
            };
            report.checked_artifacts += 1;
            if let Some(difference) = difference {
                return Ok(Some(format!("{context}: {difference}")));
            }
        }
        if other.artifacts.len() > case.artifacts.len() {
            return Ok(Some(format!(
                "case {}, artifact {}: extra in candidate",
                case.id,
                other.artifacts[case.artifacts.len()].id
            )));
        }
        report.checked_cases += 1;
    }
    if c.cases.len() > r.cases.len() {
        return Ok(Some(format!(
            "case {}: extra in candidate",
            c.cases[r.cases.len()].id
        )));
    }
    Ok(None)
}

fn within_tolerance(reference: f64, candidate: f64, tolerance: Tolerance) -> bool {
    (candidate - reference).abs() <= tolerance.atol + tolerance.rtol * reference.abs()
}

fn compare_f32(reference: &[f32], candidate: &[f32], tolerance: Tolerance) -> Option<String> {
    let mut mismatches = 0usize;
    let mut first = None;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (index, (&a, &b)) in reference.iter().zip(candidate).enumerate() {
        let a = f64::from(a);
        let b = f64::from(b);
        let absolute = (a - b).abs();
        let relative = if absolute == 0.0 {
            0.0
        } else {
            absolute / a.abs()
        };
        max_abs = max_abs.max(absolute);
        max_rel = max_rel.max(relative);
        if !within_tolerance(a, b, tolerance) {
            mismatches += 1;
            first.get_or_insert((index, a, b));
        }
    }
    first.map(|(index, a, b)| format!("f32 mismatch count={mismatches}/{}, first index={index} reference={a:.9e} candidate={b:.9e}, max abs={max_abs:.9e}, max rel={max_rel:.9e}", reference.len()))
}

fn compare_bytes(reference: &[u8], candidate: &[u8]) -> Option<String> {
    let mismatches = reference
        .iter()
        .zip(candidate)
        .filter(|(a, b)| a != b)
        .count();
    reference
        .iter()
        .zip(candidate)
        .position(|(a, b)| a != b)
        .map(|index| {
            format!(
                "u8 mismatch count={mismatches}/{}, first index={index} reference={} candidate={}",
                reference.len(),
                reference[index],
                candidate[index]
            )
        })
}

fn compare_json(
    reference: &Value,
    candidate: &Value,
    tolerance: Tolerance,
    path: &str,
) -> Option<String> {
    match (reference, candidate) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, value) in a {
                let nested = format!("{path}[{key:?}]");
                let Some(other) = b.get(key) else {
                    return Some(format!("JSON {nested}: missing key"));
                };
                if let Some(difference) = compare_json(value, other, tolerance, &nested) {
                    return Some(difference);
                }
            }
            b.keys()
                .find(|key| !a.contains_key(*key))
                .map(|key| format!("JSON {path}[{key:?}]: extra key"))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!(
                    "JSON {path}: array length differs ({} vs {})",
                    a.len(),
                    b.len()
                ));
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(index, (value, other))| {
                    compare_json(value, other, tolerance, &format!("{path}[{index}]"))
                })
        }
        (Value::Number(a), Value::Number(b)) => {
            if !a.is_f64() && !b.is_f64() {
                if a == b {
                    None
                } else {
                    Some(format!("JSON {path}: integer differs ({a} vs {b})"))
                }
            } else {
                let reference = a.as_f64().expect("finite JSON number");
                let candidate = b.as_f64().expect("finite JSON number");
                if within_tolerance(reference, candidate, tolerance) {
                    None
                } else {
                    Some(format!(
                        "JSON {path}: number differs (reference={a}, candidate={b}, abs={:.9e}, rel={:.9e})",
                        (reference - candidate).abs(),
                        (reference - candidate).abs() / reference.abs()
                    ))
                }
            }
        }
        _ if reference == candidate => None,
        _ => Some(format!(
            "JSON {path}: value/type differs (reference={reference}, candidate={candidate})"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_is_asymmetric_and_reports_metrics() {
        let t = Tolerance {
            atol: 0.1,
            rtol: 0.1,
        };
        assert!(within_tolerance(10.0, 11.05, t));
        assert!(!within_tolerance(10.0, 11.2, t));
        assert!(
            compare_f32(&[0.0, 1.0, 2.0], &[0.2, 1.0, 3.0], t)
                .unwrap()
                .contains("count=2/3, first index=0")
        );
    }

    #[test]
    fn json_integer_pairs_are_exact_and_float_pairs_tolerant() {
        let t = Tolerance {
            atol: 0.1,
            rtol: 0.0,
        };
        assert!(
            compare_json(
                &serde_json::json!({"n": 1}),
                &serde_json::json!({"n": 2}),
                t,
                "$"
            )
            .is_some()
        );
        assert!(
            compare_json(
                &serde_json::json!({"score": 0}),
                &serde_json::json!({"score": 0.01}),
                t,
                "$"
            )
            .is_none()
        );
        let a = serde_json::json!({"lines": [{"text":"a", "score":0.9}]});
        let b = serde_json::json!({"lines": [{"text":"b", "score":0.9}]});
        assert!(
            compare_json(&a, &b, t, "$")
                .unwrap()
                .contains("$[\"lines\"][0][\"text\"]")
        );
    }
}
