//! Integrity validation and bounded-memory comparison of frozen OCR bundles.

#![forbid(unsafe_code)]

pub mod bundle;
pub mod cls;
pub mod crop;
mod db;
pub mod det;
pub mod detector;
pub mod full_ocr;
pub mod image;
pub mod inventory;
pub mod lines;
mod math;
pub mod model;
pub mod ocr;
pub mod onnx;
pub mod orientation;
pub mod preprocess;
mod profile;
#[cfg(feature = "profiling")]
pub use profile::StageProfile;
pub mod rec;
pub mod recognizer;
pub mod runtime;
pub mod step12;
pub mod tensor;

use bundle::{ArtifactDescriptor, AuxiliaryFileDescriptor, CaseDescriptor, Dtype, ModelDescriptor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::{self, Read},
    path::{Component, Path, PathBuf},
    time::Instant,
};

pub const STREAM_BLOCK_BYTES: usize = 64 * 1024;
pub const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize)]
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

#[derive(Clone, Debug, Serialize, Default)]
pub struct ValidationReport {
    pub checked_models: usize,
    pub checked_cases: usize,
    pub checked_artifacts: usize,
    pub checked_auxiliary_files: usize,
    /// Bytes hashed from declared payloads; excludes the manifest itself.
    pub checked_bytes: u64,
    /// Paired comparisons share their clock and report elapsed time on ComparisonReport.
    pub elapsed_ms: Option<f64>,
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VALID: {} models, {} cases, {} artifacts, {} auxiliary files; {} declared bytes verified",
            self.checked_models,
            self.checked_cases,
            self.checked_artifacts,
            self.checked_auxiliary_files,
            self.checked_bytes
        )
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComparisonScope {
    Full,
    Stage {
        case_id: String,
        artifact_id: String,
    },
    Cls {
        required_lines: usize,
        required_artifacts: usize,
    },
    Rec {
        required_lines: usize,
        required_artifacts: usize,
    },
    Det {
        required_cases: usize,
        required_artifacts: usize,
    },
    Step12 {
        required_cases: usize,
        required_lines: usize,
        required_artifacts: usize,
    },
}

#[derive(Debug, Serialize)]
pub struct ComparisonReport {
    pub checked_cases: usize,
    pub checked_artifacts: usize,
    pub first_difference: Option<String>,
    pub scope: ComparisonScope,
    pub reference_validation: ValidationReport,
    pub candidate_validation: ValidationReport,
    pub elapsed_ms: f64,
}

impl ComparisonReport {
    pub fn matches(&self) -> bool {
        self.first_difference.is_none()
    }
}

impl fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let partial = if !matches!(self.scope, ComparisonScope::Full) {
            "PARTIAL "
        } else {
            ""
        };
        if let Some(difference) = &self.first_difference {
            write!(f, "{partial}DIFFERENCE: {difference}")
        } else {
            write!(
                f,
                "{partial}MATCH: {} cases, {} artifacts; all declared model/artifact/auxiliary hashes and lengths verified",
                self.checked_cases, self.checked_artifacts
            )?;
            if let ComparisonScope::Stage {
                case_id,
                artifact_id,
            } = &self.scope
            {
                write!(f, "; scope case={case_id:?}, artifact={artifact_id:?}")?;
            }
            if let ComparisonScope::Cls {
                required_lines,
                required_artifacts,
            } = &self.scope
            {
                write!(
                    f,
                    "; scope CLS: {required_lines} required lines, {required_artifacts} required artifacts; no full OCR parity"
                )?;
            }
            match &self.scope {
                ComparisonScope::Rec {
                    required_lines,
                    required_artifacts,
                } => write!(
                    f,
                    "; scope frozen REC tensors: {required_lines} required lines, {required_artifacts} required artifacts; no full OCR parity"
                )?,
                ComparisonScope::Det {
                    required_cases,
                    required_artifacts,
                } => write!(
                    f,
                    "; scope frozen DET tensors: {required_cases} cases, {required_artifacts} required artifacts; no pixel preprocessing or full OCR parity"
                )?,
                ComparisonScope::Step12 {
                    required_cases,
                    required_lines,
                    required_artifacts,
                } => write!(
                    f,
                    "; scope frozen image pixels through oriented crops: {required_cases} cases, {required_lines} lines, {required_artifacts} required artifacts; no recognition or full OCR parity"
                )?,
                _ => {}
            }
            Ok(())
        }
    }
}

#[derive(Deserialize, Serialize)]
struct Manifest {
    schema_version: u32,
    contract: Value,
    models: Vec<ModelDescriptor>,
    cases: Vec<CaseDescriptor>,
    #[serde(default)]
    auxiliary_files: Vec<AuxiliaryFileDescriptor>,
}

struct Bundle {
    root: PathBuf,
    manifest: Manifest,
}

impl Bundle {
    /// Load metadata only. Payloads are streamed once by validation/comparison.
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
        let bytes = read_bounded(
            File::open(&manifest_path)
                .map_err(|e| invalid(format!("{}: {e}", manifest_path.display())))?,
            "manifest.json",
        )?;
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
        for case in &manifest.cases {
            if case.kind.is_empty() {
                return Err(invalid(format!("case {}: kind is empty", case.id)));
            }
            unique_ids(
                case.artifacts.iter().map(|a| a.id.as_str()),
                &format!("artifact in case {}", case.id),
                &root,
            )?;
            for artifact in &case.artifacts {
                validate_artifact_metadata(artifact)?;
            }
        }
        let mut auxiliary_paths = HashSet::new();
        for auxiliary in &manifest.auxiliary_files {
            if auxiliary.role.is_empty() {
                return Err(invalid(format!(
                    "{}: auxiliary role is empty",
                    auxiliary.path
                )));
            }
            let path = portable_path(&auxiliary.path)?;
            if path == Path::new("manifest.json") || !auxiliary_paths.insert(path) {
                return Err(invalid(format!(
                    "{}: duplicate or self-referential auxiliary path",
                    auxiliary.path
                )));
            }
        }
        Ok(Self { root, manifest })
    }
}

fn validate_artifact_metadata(artifact: &ArtifactDescriptor) -> Result<()> {
    let context = format!("{} ({})", artifact.id, artifact.path);
    if artifact.id.is_empty() || artifact.layout.is_empty() {
        return Err(invalid(format!("{context}: ID or layout is empty")));
    }
    if artifact.dtype == Dtype::Json {
        if !artifact.shape.is_empty() || artifact.layout != "json" {
            return Err(invalid(format!(
                "{context}: JSON artifacts require shape=[] and layout=json"
            )));
        }
        if artifact.bytes > MAX_JSON_BYTES {
            return Err(invalid(format!(
                "{context}: JSON exceeds {MAX_JSON_BYTES} byte limit"
            )));
        }
    } else {
        let elements = artifact
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| invalid(format!("{context}: shape element count overflow")))?;
        let expected = elements
            .checked_mul(if artifact.dtype == Dtype::F32 { 4 } else { 1 })
            .ok_or_else(|| invalid(format!("{context}: shape byte count overflow")))?;
        if u64::try_from(expected).ok() != Some(artifact.bytes) {
            return Err(invalid(format!(
                "{context}: shape implies {expected} bytes, manifest records {}",
                artifact.bytes
            )));
        }
    }
    Ok(())
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

fn portable_path(relative: &str) -> Result<PathBuf> {
    if relative.is_empty()
        || relative.starts_with(['/', '\\'])
        || relative
            .split(['/', '\\'])
            .any(|part| !portable_segment(part))
    {
        return Err(invalid(format!("unsafe bundle path {relative:?}")));
    }
    let path = PathBuf::from(relative.replace('\\', "/"));
    if path
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid(format!("unsafe bundle path {relative:?}")));
    }
    Ok(path)
}

fn safe_file(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = portable_path(relative)?;
    let resolved = root
        .join(&path)
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
    let upper = segment
        .split('.')
        .next()
        .expect("nonempty path segment")
        .to_ascii_uppercase();
    !matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && matches!(upper.as_bytes()[3], b'1'..=b'9'))
}

fn read_bounded(reader: impl Read, context: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| invalid(format!("{context}: {e}")))?;
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(invalid(format!(
            "{context}: JSON exceeds {MAX_JSON_BYTES} byte limit"
        )));
    }
    Ok(bytes)
}

#[derive(Clone, Copy)]
struct FileSpec<'a> {
    path: &'a str,
    sha256: &'a str,
    bytes: u64,
}
impl<'a> From<&'a ArtifactDescriptor> for FileSpec<'a> {
    fn from(a: &'a ArtifactDescriptor) -> Self {
        Self {
            path: &a.path,
            sha256: &a.sha256,
            bytes: a.bytes,
        }
    }
}
impl<'a> From<&'a ModelDescriptor> for FileSpec<'a> {
    fn from(a: &'a ModelDescriptor) -> Self {
        Self {
            path: &a.path,
            sha256: &a.sha256,
            bytes: a.bytes,
        }
    }
}
impl<'a> From<&'a AuxiliaryFileDescriptor> for FileSpec<'a> {
    fn from(a: &'a AuxiliaryFileDescriptor) -> Self {
        Self {
            path: &a.path,
            sha256: &a.sha256,
            bytes: a.bytes,
        }
    }
}

struct VerifiedFile<'a> {
    file: File,
    spec: FileSpec<'a>,
    resolved: PathBuf,
    hash: Sha256,
    read_bytes: u64,
}

impl<'a> VerifiedFile<'a> {
    fn open(root: &Path, spec: FileSpec<'a>) -> Result<Self> {
        if spec.sha256.len() != 64 || !spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid(format!(
                "{}: sha256 must contain 64 hexadecimal digits",
                spec.path
            )));
        }
        let resolved = safe_file(root, spec.path)?;
        let file =
            File::open(&resolved).map_err(|e| invalid(format!("{}: {e}", resolved.display())))?;
        let actual = file
            .metadata()
            .map_err(|e| invalid(format!("{}: {e}", resolved.display())))?
            .len();
        if actual != spec.bytes {
            return Err(invalid(format!(
                "{}: byte length {actual} differs from manifest {}",
                resolved.display(),
                spec.bytes
            )));
        }
        Ok(Self {
            file,
            spec,
            resolved,
            hash: Sha256::new(),
            read_bytes: 0,
        })
    }

    fn block(&mut self, buffer: &mut [u8]) -> Result<usize> {
        let length = read_block(&mut self.file, buffer)
            .map_err(|e| invalid(format!("{}: {e}", self.resolved.display())))?;
        self.hash.update(&buffer[..length]);
        self.read_bytes = self
            .read_bytes
            .checked_add(length as u64)
            .ok_or_else(|| invalid("stream byte count overflow"))?;
        Ok(length)
    }

    fn finish(self) -> Result<()> {
        if self.read_bytes != self.spec.bytes {
            return Err(invalid(format!(
                "{}: byte length {} differs from manifest {}",
                self.resolved.display(),
                self.read_bytes,
                self.spec.bytes
            )));
        }
        let actual = format!("{:x}", self.hash.finalize());
        if !actual.eq_ignore_ascii_case(self.spec.sha256) {
            return Err(invalid(format!(
                "{}: sha256 mismatch (expected {}, actual {actual})",
                self.resolved.display(),
                self.spec.sha256
            )));
        }
        Ok(())
    }
}

/// Fill a block despite short reads so float elements cannot straddle blocks.
fn read_block(reader: &mut impl Read, buffer: &mut [u8]) -> io::Result<usize> {
    let mut offset = 0;
    while offset < buffer.len() {
        match reader.read(&mut buffer[offset..]) {
            Ok(0) => break,
            Ok(length) => offset += length,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(offset)
}

fn account_bytes(report: &mut ValidationReport, bytes: u64) -> Result<()> {
    report.checked_bytes = report
        .checked_bytes
        .checked_add(bytes)
        .ok_or_else(|| invalid("validated byte count overflow"))?;
    Ok(())
}

fn first_nonfinite(bytes: &[u8], offset: usize) -> Option<usize> {
    bytes
        .chunks_exact(4)
        .position(|b| !f32::from_le_bytes(b.try_into().expect("four-byte chunk")).is_finite())
        .map(|index| offset + index)
}

fn finite_error(artifact: &ArtifactDescriptor, index: Option<usize>) -> Result<()> {
    match index {
        Some(index) => Err(invalid(format!(
            "{} ({}): nonfinite f32 at index {index}",
            artifact.id, artifact.path
        ))),
        None => Ok(()),
    }
}

fn stream_validate(root: &Path, spec: FileSpec<'_>, floats: bool) -> Result<Option<usize>> {
    let mut file = VerifiedFile::open(root, spec)?;
    let mut buffer = vec![0; STREAM_BLOCK_BYTES];
    let mut nonfinite = None;
    let mut offset = 0;
    loop {
        let length = file.block(&mut buffer)?;
        if length == 0 {
            break;
        }
        if floats {
            nonfinite = nonfinite.or_else(|| first_nonfinite(&buffer[..length], offset));
            offset += length / 4;
        }
    }
    file.finish()?;
    Ok(nonfinite)
}

fn read_json_artifact(root: &Path, artifact: &ArtifactDescriptor) -> Result<Value> {
    let mut file = VerifiedFile::open(root, artifact.into())?;
    let mut buffer = vec![0; STREAM_BLOCK_BYTES];
    let mut bytes = Vec::new();
    loop {
        let length = file.block(&mut buffer)?;
        if length == 0 {
            break;
        }
        if (bytes.len() as u64 + length as u64) > MAX_JSON_BYTES {
            return Err(invalid(format!(
                "{}: JSON exceeds {MAX_JSON_BYTES} byte limit",
                artifact.path
            )));
        }
        bytes.extend_from_slice(&buffer[..length]);
    }
    file.finish()?;
    serde_json::from_slice(&bytes).map_err(|e| {
        invalid(format!(
            "{} ({}): invalid JSON: {e}",
            artifact.id, artifact.path
        ))
    })
}

fn validate_artifact(
    bundle: &Bundle,
    artifact: &ArtifactDescriptor,
    report: &mut ValidationReport,
) -> Result<()> {
    if artifact.dtype == Dtype::Json {
        read_json_artifact(&bundle.root, artifact)?;
    } else {
        finite_error(
            artifact,
            stream_validate(&bundle.root, artifact.into(), artifact.dtype == Dtype::F32)?,
        )?;
    }
    report.checked_artifacts += 1;
    account_bytes(report, artifact.bytes)
}

fn validate_assets(bundle: &Bundle, report: &mut ValidationReport) -> Result<()> {
    for model in &bundle.manifest.models {
        stream_validate(&bundle.root, model.into(), false)?;
        report.checked_models += 1;
        account_bytes(report, model.bytes)?;
    }
    for auxiliary in &bundle.manifest.auxiliary_files {
        stream_validate(&bundle.root, auxiliary.into(), false)?;
        report.checked_auxiliary_files += 1;
        account_bytes(report, auxiliary.bytes)?;
    }
    report.checked_cases = bundle.manifest.cases.len();
    Ok(())
}

/// Validate every declared file, without comparing outputs or parsing ONNX models.
pub fn validate_bundle(directory: &Path) -> Result<ValidationReport> {
    let start = Instant::now();
    let bundle = Bundle::load(directory)?;
    let mut report = ValidationReport::default();
    validate_assets(&bundle, &mut report)?;
    for case in &bundle.manifest.cases {
        for artifact in &case.artifacts {
            validate_artifact(&bundle, artifact, &mut report)?;
        }
    }
    report.elapsed_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
    Ok(report)
}

/// Compare complete ordered bundles, preserving the v1 tolerance/identity contract.
/// A parity difference is returned only after every declared file is verified.
pub fn compare_bundles(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    compare(reference, candidate, tolerance, ComparisonScope::Full)
}

/// Compare exactly one artifact by ID. This is explicitly partial parity: other
/// candidate artifacts/cases may be absent or reordered, but every present file
/// is still validated, and contracts/model identities must still match.
pub fn compare_stage(
    reference: &Path,
    candidate: &Path,
    case_id: &str,
    artifact_id: &str,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    if case_id.is_empty() || artifact_id.is_empty() {
        return Err(invalid(
            "stage selection requires nonempty case and artifact IDs",
        ));
    }
    compare(
        reference,
        candidate,
        tolerance,
        ComparisonScope::Stage {
            case_id: case_id.into(),
            artifact_id: artifact_id.into(),
        },
    )
}

/// Compare every captured classifier input, node, and probability tensor.
/// The reference determines the complete selection. Other stages remain outside
/// parity scope, but every declared payload on both sides is validated once.
pub fn compare_cls(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    compare(
        reference,
        candidate,
        tolerance,
        ComparisonScope::Cls {
            required_lines: 0,
            required_artifacts: 0,
        },
    )
}

pub fn compare_det(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    compare(
        reference,
        candidate,
        tolerance,
        ComparisonScope::Det {
            required_cases: 0,
            required_artifacts: 0,
        },
    )
}
pub fn compare_step12(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    compare(
        reference,
        candidate,
        tolerance,
        ComparisonScope::Step12 {
            required_cases: 0,
            required_lines: 0,
            required_artifacts: 0,
        },
    )
}

fn identity_difference(reference: &Manifest, candidate: &Manifest) -> Option<String> {
    if reference.contract != candidate.contract {
        return Some("manifest contract differs".into());
    }
    if reference.models.len() != candidate.models.len() {
        return Some(format!(
            "model count differs: reference {}, candidate {}",
            reference.models.len(),
            candidate.models.len()
        ));
    }
    for model in &reference.models {
        let Some(other) = candidate.models.iter().find(|other| other.id == model.id) else {
            return Some(format!("missing model {}", model.id));
        };
        if !model.sha256.eq_ignore_ascii_case(&other.sha256) {
            return Some(format!("model {} sha256 differs", model.id));
        }
    }
    None
}

fn metadata_difference(
    context: &str,
    reference: &ArtifactDescriptor,
    candidate: &ArtifactDescriptor,
) -> Option<String> {
    if reference.dtype != candidate.dtype
        || reference.shape != candidate.shape
        || reference.layout != candidate.layout
    {
        Some(format!(
            "{context}: metadata differs; reference {:?} {:?} {}, candidate {:?} {:?} {}",
            reference.dtype,
            reference.shape,
            reference.layout,
            candidate.dtype,
            candidate.shape,
            candidate.layout
        ))
    } else {
        None
    }
}

fn compare(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
    scope: ComparisonScope,
) -> Result<ComparisonReport> {
    if !tolerance.atol.is_finite()
        || tolerance.atol < 0.0
        || !tolerance.rtol.is_finite()
        || tolerance.rtol < 0.0
    {
        return Err(invalid("tolerances must be finite and nonnegative"));
    }
    let start = Instant::now();
    let reference = Bundle::load(reference)?;
    let candidate = Bundle::load(candidate)?;
    let mut report = ComparisonReport {
        checked_cases: 0,
        checked_artifacts: 0,
        first_difference: identity_difference(&reference.manifest, &candidate.manifest),
        scope,
        reference_validation: ValidationReport::default(),
        candidate_validation: ValidationReport::default(),
        elapsed_ms: 0.0,
    };
    validate_assets(&reference, &mut report.reference_validation)?;
    validate_assets(&candidate, &mut report.candidate_validation)?;
    match report.scope.clone() {
        ComparisonScope::Full => compare_full(&reference, &candidate, tolerance, &mut report)?,
        ComparisonScope::Stage {
            case_id,
            artifact_id,
        } => compare_selected(
            &reference,
            &candidate,
            &case_id,
            &artifact_id,
            tolerance,
            &mut report,
        )?,
        ComparisonScope::Cls { .. } => {
            compare_classifier(&reference, &candidate, tolerance, &mut report)?
        }
        ComparisonScope::Det { .. }
        | ComparisonScope::Step12 { .. }
        | ComparisonScope::Rec { .. } => {
            compare_pipeline_selection(&reference, &candidate, tolerance, &mut report)?
        }
    }
    report.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(report)
}

fn retain_difference(report: &mut ComparisonReport, difference: Option<String>) {
    if report.first_difference.is_none() {
        report.first_difference = difference;
    }
}

fn compare_or_validate(
    reference: &Bundle,
    candidate: &Bundle,
    artifact: &ArtifactDescriptor,
    other: Option<&ArtifactDescriptor>,
    context: &str,
    tolerance: Tolerance,
    report: &mut ComparisonReport,
) -> Result<()> {
    if let Some(other) = other {
        retain_difference(report, metadata_difference(context, artifact, other));
        if report.first_difference.is_none() {
            let difference = compare_artifacts(reference, candidate, artifact, other, tolerance)?;
            report.reference_validation.checked_artifacts += 1;
            account_bytes(&mut report.reference_validation, artifact.bytes)?;
            report.candidate_validation.checked_artifacts += 1;
            account_bytes(&mut report.candidate_validation, other.bytes)?;
            report.checked_artifacts += 1;
            retain_difference(
                report,
                difference.map(|difference| format!("{context}: {difference}")),
            );
            return Ok(());
        }
        validate_artifact(candidate, other, &mut report.candidate_validation)?;
    }
    validate_artifact(reference, artifact, &mut report.reference_validation)
}

fn compare_pipeline_selection(
    reference: &Bundle,
    candidate: &Bundle,
    tolerance: Tolerance,
    report: &mut ComparisonReport,
) -> Result<()> {
    type ArtifactPredicate = fn(&str) -> Result<bool>;
    type ScopedSelection = (HashSet<(usize, usize)>, ArtifactPredicate, ComparisonScope);
    let (selected, predicate, scope): ScopedSelection = match report.scope {
        ComparisonScope::Rec { .. } => {
            let selection = rec::selection(reference)?;
            rec::validate_declared_structures(candidate)?;
            let scope = ComparisonScope::Rec {
                required_lines: selection.lines.len(),
                required_artifacts: selection.artifacts.len(),
            };
            (selection.artifacts, rec::is_rec_artifact, scope)
        }
        ComparisonScope::Det { .. } => {
            let selection = det::detector_selection(&reference.manifest)?;
            det::validate_reference_coverage(reference, &selection)?;
            let scope = ComparisonScope::Det {
                required_cases: selection.cases.len(),
                required_artifacts: selection.artifacts.len(),
            };
            (selection.artifacts, det::is_detector_artifact, scope)
        }
        ComparisonScope::Step12 { .. } => {
            let selection = step12::selection(reference)?;
            step12::validate_declared_structures(candidate)?;
            let scope = ComparisonScope::Step12 {
                required_cases: reference.manifest.cases.len(),
                required_lines: selection.lines,
                required_artifacts: selection.artifacts.len(),
            };
            (selection.artifacts, step12::is_step12_artifact, scope)
        }
        _ => unreachable!(),
    };
    report.scope = scope;
    if matches!(report.scope, ComparisonScope::Step12 { .. }) {
        let reference_cases: HashSet<_> = reference
            .manifest
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect();
        for case in &candidate.manifest.cases {
            if !reference_cases.contains(case.id.as_str()) {
                retain_difference(
                    report,
                    Some(format!("case {}: extra Step12 case in candidate", case.id)),
                );
            }
        }
    }
    let candidate_cases: HashMap<_, _> = candidate
        .manifest
        .cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect();
    let candidate_artifacts: HashMap<_, _> = candidate
        .manifest
        .cases
        .iter()
        .flat_map(|case| {
            case.artifacts
                .iter()
                .map(|a| ((case.id.as_str(), a.id.as_str()), a))
        })
        .collect();
    let mut consumed = HashSet::new();
    for (ci, case) in reference.manifest.cases.iter().enumerate() {
        let selected_case = selected.iter().any(|&(case_index, _)| case_index == ci);
        if selected_case {
            match candidate_cases.get(case.id.as_str()) {
                Some(other) if other.kind == case.kind => {}
                Some(_) => {
                    retain_difference(report, Some(format!("case {} kind differs", case.id)))
                }
                None => retain_difference(
                    report,
                    Some(format!("case {} missing from candidate", case.id)),
                ),
            }
        }
        for (ai, artifact) in case.artifacts.iter().enumerate() {
            if selected.contains(&(ci, ai)) {
                let other = candidate_artifacts
                    .get(&(case.id.as_str(), artifact.id.as_str()))
                    .copied();
                let context = format!("case {}, artifact {}", case.id, artifact.id);
                if other.is_none() {
                    retain_difference(report, Some(format!("{context}: missing from candidate")))
                } else {
                    consumed.insert((case.id.as_str(), artifact.id.as_str()));
                }
                compare_or_validate(
                    reference, candidate, artifact, other, &context, tolerance, report,
                )?;
            } else {
                validate_artifact(reference, artifact, &mut report.reference_validation)?;
            }
        }
        if selected_case && report.first_difference.is_none() {
            report.checked_cases += 1;
        }
    }
    for case in &candidate.manifest.cases {
        for artifact in &case.artifacts {
            if !consumed.contains(&(case.id.as_str(), artifact.id.as_str())) {
                if predicate(&artifact.id)? {
                    retain_difference(
                        report,
                        Some(format!(
                            "case {}, artifact {}: extra in-scope artifact in candidate",
                            case.id, artifact.id
                        )),
                    )
                }
                validate_artifact(candidate, artifact, &mut report.candidate_validation)?;
            }
        }
    }
    Ok(())
}

fn compare_classifier(
    reference: &Bundle,
    candidate: &Bundle,
    tolerance: Tolerance,
    report: &mut ComparisonReport,
) -> Result<()> {
    let selection = cls::classifier_selection(&reference.manifest)?;
    report.scope = ComparisonScope::Cls {
        required_lines: selection.lines.len(),
        required_artifacts: selection.artifacts.len(),
    };
    let candidate_cases: HashMap<_, _> = candidate
        .manifest
        .cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect();
    let candidate_artifacts: HashMap<_, _> = candidate
        .manifest
        .cases
        .iter()
        .flat_map(|case| {
            case.artifacts
                .iter()
                .map(|artifact| ((case.id.as_str(), artifact.id.as_str()), artifact))
        })
        .collect();
    let mut consumed = HashSet::new();
    for (case_index, case) in reference.manifest.cases.iter().enumerate() {
        let candidate_case = candidate_cases.get(case.id.as_str()).copied();
        let selected_case = selection
            .lines
            .iter()
            .any(|line| line.case_index == case_index);
        if selected_case {
            if let Some(other) = candidate_case {
                if other.kind != case.kind {
                    retain_difference(
                        report,
                        Some(format!(
                            "case {}: kind differs ({} vs {})",
                            case.id, case.kind, other.kind,
                        )),
                    );
                }
            } else {
                retain_difference(
                    report,
                    Some(format!("case {}: missing from candidate", case.id)),
                );
            }
        }
        for (artifact_index, artifact) in case.artifacts.iter().enumerate() {
            if selection.artifacts.contains(&(case_index, artifact_index)) {
                let other = candidate_artifacts
                    .get(&(case.id.as_str(), artifact.id.as_str()))
                    .copied();
                let context = format!("case {}, artifact {}", case.id, artifact.id);
                if other.is_none() {
                    retain_difference(report, Some(format!("{context}: missing from candidate")));
                } else {
                    consumed.insert((case.id.as_str(), artifact.id.as_str()));
                }
                compare_or_validate(
                    reference, candidate, artifact, other, &context, tolerance, report,
                )?;
            } else {
                validate_artifact(reference, artifact, &mut report.reference_validation)?;
            }
        }
        if selected_case && report.first_difference.is_none() {
            report.checked_cases += 1;
        }
    }
    for case in &candidate.manifest.cases {
        for artifact in &case.artifacts {
            if !consumed.contains(&(case.id.as_str(), artifact.id.as_str())) {
                if cls::is_classifier_artifact(&artifact.id)? {
                    retain_difference(
                        report,
                        Some(format!(
                            "case {}, artifact {}: extra classifier artifact in candidate",
                            case.id, artifact.id
                        )),
                    );
                }
                validate_artifact(candidate, artifact, &mut report.candidate_validation)?;
            }
        }
    }
    Ok(())
}

fn compare_full(
    reference: &Bundle,
    candidate: &Bundle,
    tolerance: Tolerance,
    report: &mut ComparisonReport,
) -> Result<()> {
    for (index, case) in reference.manifest.cases.iter().enumerate() {
        let other = candidate.manifest.cases.get(index);
        match other {
            None => retain_difference(
                report,
                Some(format!("case {}: missing from candidate", case.id)),
            ),
            Some(other) if case.id != other.id => retain_difference(
                report,
                Some(format!(
                    "case order/ID differs at index {index}: reference {}, candidate {}",
                    case.id, other.id
                )),
            ),
            Some(other) if case.kind != other.kind => retain_difference(
                report,
                Some(format!(
                    "case {}: kind differs ({} vs {})",
                    case.id, case.kind, other.kind
                )),
            ),
            _ => {}
        }
        for (stage, artifact) in case.artifacts.iter().enumerate() {
            let other_artifact = other.and_then(|other| other.artifacts.get(stage));
            let context = format!("case {}, artifact {}", case.id, artifact.id);
            match other_artifact {
                None => {
                    retain_difference(report, Some(format!("{context}: missing from candidate")))
                }
                Some(other) if artifact.id != other.id => retain_difference(
                    report,
                    Some(format!(
                        "{context}: stage order/ID differs at index {stage} (candidate {})",
                        other.id
                    )),
                ),
                _ => {}
            }
            compare_or_validate(
                reference,
                candidate,
                artifact,
                other_artifact,
                &context,
                tolerance,
                report,
            )?;
        }
        if let Some(other) = other {
            for artifact in other.artifacts.iter().skip(case.artifacts.len()) {
                retain_difference(
                    report,
                    Some(format!(
                        "case {}, artifact {}: extra in candidate",
                        case.id, artifact.id
                    )),
                );
                validate_artifact(candidate, artifact, &mut report.candidate_validation)?;
            }
        }
        if report.first_difference.is_none() {
            report.checked_cases += 1;
        }
    }
    for case in candidate
        .manifest
        .cases
        .iter()
        .skip(reference.manifest.cases.len())
    {
        retain_difference(
            report,
            Some(format!("case {}: extra in candidate", case.id)),
        );
        for artifact in &case.artifacts {
            validate_artifact(candidate, artifact, &mut report.candidate_validation)?;
        }
    }
    Ok(())
}

fn compare_selected(
    reference: &Bundle,
    candidate: &Bundle,
    case_id: &str,
    artifact_id: &str,
    tolerance: Tolerance,
    report: &mut ComparisonReport,
) -> Result<()> {
    let selected_case = reference
        .manifest
        .cases
        .iter()
        .find(|case| case.id == case_id)
        .ok_or_else(|| {
            invalid(format!(
                "selected reference case {case_id:?} does not exist"
            ))
        })?;
    if !selected_case
        .artifacts
        .iter()
        .any(|artifact| artifact.id == artifact_id)
    {
        return Err(invalid(format!(
            "selected reference artifact {case_id:?}/{artifact_id:?} does not exist"
        )));
    }
    let candidate_case = candidate
        .manifest
        .cases
        .iter()
        .find(|case| case.id == case_id);
    let selected_candidate = candidate_case.and_then(|case| {
        case.artifacts
            .iter()
            .find(|artifact| artifact.id == artifact_id)
    });
    if let Some(case) = candidate_case
        && case.kind != selected_case.kind
    {
        retain_difference(
            report,
            Some(format!(
                "case {case_id}: kind differs ({} vs {})",
                selected_case.kind, case.kind
            )),
        );
    }
    if selected_candidate.is_none() {
        retain_difference(
            report,
            Some(format!(
                "case {case_id}, artifact {artifact_id}: missing from candidate"
            )),
        );
    }
    for case in &reference.manifest.cases {
        for artifact in &case.artifacts {
            if case.id == case_id && artifact.id == artifact_id {
                compare_or_validate(
                    reference,
                    candidate,
                    artifact,
                    selected_candidate,
                    &format!("case {case_id}, artifact {artifact_id}"),
                    tolerance,
                    report,
                )?;
            } else {
                validate_artifact(reference, artifact, &mut report.reference_validation)?;
            }
        }
    }
    for case in &candidate.manifest.cases {
        for artifact in &case.artifacts {
            if case.id != case_id || artifact.id != artifact_id {
                validate_artifact(candidate, artifact, &mut report.candidate_validation)?;
            }
        }
    }
    if report.first_difference.is_none() {
        report.checked_cases = 1;
    }
    Ok(())
}

#[derive(Default)]
struct FloatMetrics {
    mismatches: usize,
    first: Option<(usize, f64, f64)>,
    max_abs: f64,
    max_rel: f64,
}

impl FloatMetrics {
    fn observe(&mut self, index: usize, reference: f32, candidate: f32, tolerance: Tolerance) {
        let a = f64::from(reference);
        let b = f64::from(candidate);
        let absolute = (a - b).abs();
        let relative = if absolute == 0.0 {
            0.0
        } else {
            absolute / a.abs()
        };
        self.max_abs = self.max_abs.max(absolute);
        self.max_rel = self.max_rel.max(relative);
        if !within_tolerance(a, b, tolerance) {
            self.mismatches += 1;
            self.first.get_or_insert((index, a, b));
        }
    }

    fn difference(self, elements: usize) -> Option<String> {
        self.first.map(|(index, a, b)| format!("f32 mismatch count={}/{elements}, first index={index} reference={a:.9e} candidate={b:.9e}, max abs={:.9e}, max rel={:.9e}", self.mismatches, self.max_abs, self.max_rel))
    }
}

fn compare_artifacts(
    reference: &Bundle,
    candidate: &Bundle,
    artifact: &ArtifactDescriptor,
    other: &ArtifactDescriptor,
    tolerance: Tolerance,
) -> Result<Option<String>> {
    if artifact.dtype == Dtype::Json {
        return Ok(compare_json(
            &read_json_artifact(&reference.root, artifact)?,
            &read_json_artifact(&candidate.root, other)?,
            tolerance,
            "$",
        ));
    }
    let mut reference_file = VerifiedFile::open(&reference.root, artifact.into())?;
    let mut candidate_file = VerifiedFile::open(&candidate.root, other.into())?;
    let mut a = vec![0; STREAM_BLOCK_BYTES];
    let mut b = vec![0; STREAM_BLOCK_BYTES];
    let mut floats = FloatMetrics::default();
    let mut byte_mismatches = 0usize;
    let mut first_byte = None;
    let mut reference_nonfinite = None;
    let mut candidate_nonfinite = None;
    let mut offset = 0usize;
    loop {
        let a_length = reference_file.block(&mut a)?;
        let b_length = candidate_file.block(&mut b)?;
        if a_length == 0 && b_length == 0 {
            break;
        }
        if artifact.dtype == Dtype::F32 {
            reference_nonfinite =
                reference_nonfinite.or_else(|| first_nonfinite(&a[..a_length], offset / 4));
            candidate_nonfinite =
                candidate_nonfinite.or_else(|| first_nonfinite(&b[..b_length], offset / 4));
            for (index, (left, right)) in a[..a_length]
                .chunks_exact(4)
                .zip(b[..b_length].chunks_exact(4))
                .enumerate()
            {
                floats.observe(
                    offset / 4 + index,
                    f32::from_le_bytes(left.try_into().expect("four-byte chunk")),
                    f32::from_le_bytes(right.try_into().expect("four-byte chunk")),
                    tolerance,
                );
            }
        } else {
            for (index, (&left, &right)) in a[..a_length].iter().zip(&b[..b_length]).enumerate() {
                if left != right {
                    byte_mismatches += 1;
                    first_byte.get_or_insert((offset + index, left, right));
                }
            }
        }
        offset += a_length;
    }
    reference_file.finish()?;
    candidate_file.finish()?;
    finite_error(artifact, reference_nonfinite)?;
    finite_error(other, candidate_nonfinite)?;
    Ok(if artifact.dtype == Dtype::F32 {
        floats.difference(offset / 4)
    } else {
        first_byte.map(|(index, a, b)| format!("u8 mismatch count={byte_mismatches}/{offset}, first index={index} reference={a} candidate={b}"))
    })
}

fn within_tolerance(reference: f64, candidate: f64, tolerance: Tolerance) -> bool {
    (candidate - reference).abs() <= tolerance.atol + tolerance.rtol * reference.abs()
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
        let mut metrics = FloatMetrics::default();
        for (index, (a, b)) in [(0.0, 0.2), (1.0, 1.0), (2.0, 3.0)].into_iter().enumerate() {
            metrics.observe(index, a, b, t);
        }
        assert!(
            metrics
                .difference(3)
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
        assert!(
            compare_json(
                &serde_json::json!({"lines": [{"text":"a", "score":0.9}]}),
                &serde_json::json!({"lines": [{"text":"b", "score":0.9}]}),
                t,
                "$"
            )
            .unwrap()
            .contains("$[\"lines\"][0][\"text\"]")
        );
    }

    #[test]
    fn short_reads_fill_float_aligned_blocks_without_losing_bytes() {
        struct ShortReader {
            source: io::Cursor<Vec<u8>>,
            interrupted: bool,
        }
        impl Read for ShortReader {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                if !self.interrupted {
                    self.interrupted = true;
                    return Err(io::ErrorKind::Interrupted.into());
                }
                let length = buffer.len().min(3);
                self.source.read(&mut buffer[..length])
            }
        }
        let source: Vec<u8> = [1.0f32, 2.0, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut reader = ShortReader {
            source: io::Cursor::new(source.clone()),
            interrupted: false,
        };
        let mut buffer = [0; 8];
        assert_eq!(read_block(&mut reader, &mut buffer).unwrap(), 8);
        assert_eq!(&buffer, &source[..8]);
        assert_eq!(read_block(&mut reader, &mut buffer).unwrap(), 4);
        assert_eq!(&buffer[..4], &source[8..]);
        assert_eq!(read_block(&mut reader, &mut buffer).unwrap(), 0);
    }
}
