//! Immutable v1 bundle writer and shared descriptors. It does not run inference.

use crate::{
    BundleError, MAX_JSON_BYTES, STREAM_BLOCK_BYTES, invalid, portable_path,
    validate_artifact_metadata,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

type Result<T> = std::result::Result<T, BundleError>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    U8,
    Json,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct ArtifactDescriptor {
    pub id: String,
    pub path: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub layout: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct ModelDescriptor {
    pub id: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct CaseDescriptor {
    pub id: String,
    pub kind: String,
    pub artifacts: Vec<ArtifactDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct AuxiliaryFileDescriptor {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub role: String,
}

pub struct BundleWriter {
    root: PathBuf,
    contract: Value,
    models: Vec<ModelDescriptor>,
    cases: Vec<CaseDescriptor>,
    auxiliary_files: Vec<AuxiliaryFileDescriptor>,
    written_artifacts: HashMap<String, ArtifactDescriptor>,
}

impl BundleWriter {
    /// Create a fresh output directory. A failed writer leaves no manifest.
    pub fn new(directory: &Path, contract: Value) -> Result<Self> {
        if !contract.is_object() {
            return Err(invalid("contract must be an object"));
        }
        let absolute = std::path::absolute(directory)
            .map_err(|e| invalid(format!("{}: {e}", directory.display())))?;
        if let Some(parent) = absolute.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| invalid(format!("{}: {e}", parent.display())))?;
        }
        fs::create_dir(&absolute).map_err(|e| {
            invalid(format!(
                "{}: cannot create new bundle directory: {e}",
                absolute.display()
            ))
        })?;
        let root = absolute
            .canonicalize()
            .map_err(|e| invalid(format!("{}: {e}", absolute.display())))?;
        Ok(Self {
            root,
            contract,
            models: Vec::new(),
            cases: Vec::new(),
            auxiliary_files: Vec::new(),
            written_artifacts: HashMap::new(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn copy_model(
        &mut self,
        id: &str,
        source: &Path,
        relative: &str,
    ) -> Result<ModelDescriptor> {
        if id.is_empty() || self.models.iter().any(|model| model.id == id) {
            return Err(invalid(format!("empty or duplicate model ID {id:?}")));
        }
        let path = normalized_path(relative)?;
        let (sha256, bytes) = self.copy_file(source, &path)?;
        let descriptor = ModelDescriptor {
            id: id.into(),
            path,
            sha256,
            bytes,
        };
        self.models.push(descriptor.clone());
        Ok(descriptor)
    }

    pub fn write_bytes(
        &mut self,
        id: &str,
        relative: &str,
        values: &[u8],
        shape: &[usize],
        layout: &str,
    ) -> Result<ArtifactDescriptor> {
        let descriptor = Self::artifact(
            id,
            relative,
            Dtype::U8,
            shape,
            layout,
            values.len() as u64,
            format!("{:x}", Sha256::digest(values)),
        )?;
        let mut file = self.create_new(&descriptor.path)?;
        file.write_all(values)
            .map_err(|e| invalid(format!("{}: {e}", descriptor.path)))?;
        self.written_artifacts
            .insert(descriptor.path.clone(), descriptor.clone());
        Ok(descriptor)
    }

    pub fn write_floats(
        &mut self,
        id: &str,
        relative: &str,
        values: &[f32],
        shape: &[usize],
        layout: &str,
    ) -> Result<ArtifactDescriptor> {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "artifact {id}: nonfinite f32 at index {index}"
            )));
        }
        let bytes = (values.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("float byte count overflow"))?;
        let mut descriptor = Self::artifact(
            id,
            relative,
            Dtype::F32,
            shape,
            layout,
            bytes,
            String::new(),
        )?;
        let mut file = self.create_new(&descriptor.path)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; STREAM_BLOCK_BYTES];
        for chunk in values.chunks(STREAM_BLOCK_BYTES / 4) {
            for (target, value) in buffer.chunks_exact_mut(4).zip(chunk) {
                target.copy_from_slice(&value.to_le_bytes());
            }
            let encoded = &buffer[..chunk.len() * 4];
            file.write_all(encoded)
                .map_err(|e| invalid(format!("{}: {e}", descriptor.path)))?;
            hash.update(encoded);
        }
        descriptor.sha256 = format!("{:x}", hash.finalize());
        self.written_artifacts
            .insert(descriptor.path.clone(), descriptor.clone());
        Ok(descriptor)
    }

    pub fn write_json(
        &mut self,
        id: &str,
        relative: &str,
        value: &Value,
    ) -> Result<ArtifactDescriptor> {
        let bytes = json_bytes(value)?;
        let descriptor = Self::artifact(
            id,
            relative,
            Dtype::Json,
            &[],
            "json",
            bytes.len() as u64,
            format!("{:x}", Sha256::digest(&bytes)),
        )?;
        let mut file = self.create_new(&descriptor.path)?;
        file.write_all(&bytes)
            .map_err(|e| invalid(format!("{}: {e}", descriptor.path)))?;
        self.written_artifacts
            .insert(descriptor.path.clone(), descriptor.clone());
        Ok(descriptor)
    }

    pub fn write_auxiliary(
        &mut self,
        relative: &str,
        values: &[u8],
        role: &str,
    ) -> Result<AuxiliaryFileDescriptor> {
        if role.is_empty() {
            return Err(invalid("auxiliary role is empty"));
        }
        let path = normalized_path(relative)?;
        let mut file = self.create_new(&path)?;
        file.write_all(values)
            .map_err(|e| invalid(format!("{path}: {e}")))?;
        let descriptor = AuxiliaryFileDescriptor {
            path,
            sha256: format!("{:x}", Sha256::digest(values)),
            bytes: values.len() as u64,
            role: role.into(),
        };
        self.auxiliary_files.push(descriptor.clone());
        Ok(descriptor)
    }

    pub fn copy_auxiliary(
        &mut self,
        source: &Path,
        relative: &str,
        role: &str,
    ) -> Result<AuxiliaryFileDescriptor> {
        if role.is_empty() {
            return Err(invalid("auxiliary role is empty"));
        }
        let path = normalized_path(relative)?;
        let (sha256, bytes) = self.copy_file(source, &path)?;
        let descriptor = AuxiliaryFileDescriptor {
            path,
            sha256,
            bytes,
            role: role.into(),
        };
        self.auxiliary_files.push(descriptor.clone());
        Ok(descriptor)
    }

    /// Attach artifacts actually produced by this writer, preserving their order.
    pub fn add_case(
        &mut self,
        id: &str,
        kind: &str,
        artifacts: Vec<ArtifactDescriptor>,
    ) -> Result<()> {
        if id.is_empty() || self.cases.iter().any(|case| case.id == id) {
            return Err(invalid(format!("empty or duplicate case ID {id:?}")));
        }
        if kind.is_empty() {
            return Err(invalid(format!("case {id}: kind is empty")));
        }
        let mut ids = HashSet::new();
        for artifact in &artifacts {
            if !ids.insert(&artifact.id) {
                return Err(invalid(format!(
                    "case {id}: duplicate artifact ID {:?}",
                    artifact.id
                )));
            }
            if self.written_artifacts.get(&artifact.path) != Some(artifact) {
                return Err(invalid(format!(
                    "case {id}: artifact {:?} was not produced by this writer",
                    artifact.id
                )));
            }
        }
        self.cases.push(CaseDescriptor {
            id: id.into(),
            kind: kind.into(),
            artifacts,
        });
        Ok(())
    }

    /// Additional fields are informational metadata; they cannot override the
    /// schema, contract or indexed file descriptors. Publish the manifest last.
    pub fn finish(self, metadata: Value) -> Result<PathBuf> {
        let Value::Object(mut object) = metadata else {
            return Err(invalid("manifest metadata must be an object"));
        };
        for reserved in [
            "schema_version",
            "contract",
            "models",
            "cases",
            "auxiliary_files",
        ] {
            if object.contains_key(reserved) {
                return Err(invalid(format!(
                    "manifest metadata cannot override {reserved}"
                )));
            }
        }
        let declared: HashSet<&str> = self
            .cases
            .iter()
            .flat_map(|case| case.artifacts.iter().map(|artifact| artifact.path.as_str()))
            .collect();
        if self
            .written_artifacts
            .keys()
            .any(|path| !declared.contains(path.as_str()))
        {
            return Err(invalid(
                "written artifacts must be attached to a case before finishing",
            ));
        }
        object.insert("schema_version".into(), Value::from(1));
        object.insert("contract".into(), self.contract.clone());
        object.insert(
            "models".into(),
            serde_json::to_value(&self.models).map_err(|e| invalid(e.to_string()))?,
        );
        object.insert(
            "cases".into(),
            serde_json::to_value(&self.cases).map_err(|e| invalid(e.to_string()))?,
        );
        object.insert(
            "auxiliary_files".into(),
            serde_json::to_value(&self.auxiliary_files).map_err(|e| invalid(e.to_string()))?,
        );
        let bytes = json_bytes(&Value::Object(object))?;
        let manifest = self.root.join("manifest.json");
        if manifest.exists() {
            return Err(invalid("manifest.json already exists"));
        }
        let temporary = self.root.join(".manifest.json.tmp");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| invalid(format!("{}: {e}", temporary.display())))?;
        file.write_all(&bytes)
            .map_err(|e| invalid(format!("{}: {e}", temporary.display())))?;
        file.sync_all()
            .map_err(|e| invalid(format!("{}: {e}", temporary.display())))?;
        drop(file);
        fs::rename(&temporary, &manifest)
            .map_err(|e| invalid(format!("{}: {e}", manifest.display())))?;
        Ok(self.root)
    }

    fn artifact(
        id: &str,
        relative: &str,
        dtype: Dtype,
        shape: &[usize],
        layout: &str,
        bytes: u64,
        sha256: String,
    ) -> Result<ArtifactDescriptor> {
        let descriptor = ArtifactDescriptor {
            id: id.into(),
            path: normalized_path(relative)?,
            dtype,
            shape: shape.to_vec(),
            layout: layout.into(),
            sha256,
            bytes,
        };
        validate_artifact_metadata(&descriptor)?;
        Ok(descriptor)
    }

    fn create_new(&self, relative: &str) -> Result<File> {
        let path = portable_path(relative)?;
        let mut parent = self.root.clone();
        if let Some(relative_parent) = path.parent() {
            for component in relative_parent.components() {
                let Component::Normal(segment) = component else {
                    return Err(invalid("unsafe output parent"));
                };
                parent.push(segment);
                match fs::create_dir(&parent) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(invalid(format!("{}: {error}", parent.display()))),
                }
                parent = parent
                    .canonicalize()
                    .map_err(|e| invalid(format!("{}: {e}", parent.display())))?;
                if !parent.starts_with(&self.root) || !parent.is_dir() {
                    return Err(invalid(
                        "output path resolves outside its bundle or is not a directory",
                    ));
                }
            }
        }
        let destination = parent.join(
            path.file_name()
                .ok_or_else(|| invalid("output filename is missing"))?,
        );
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(|e| invalid(format!("{}: {e}", destination.display())))
    }

    fn copy_file(&self, source: &Path, relative: &str) -> Result<(String, u64)> {
        let mut input =
            File::open(source).map_err(|e| invalid(format!("{}: {e}", source.display())))?;
        let mut output = self.create_new(relative)?;
        let mut buffer = vec![0; STREAM_BLOCK_BYTES];
        let mut hash = Sha256::new();
        let mut bytes = 0u64;
        loop {
            let count = input
                .read(&mut buffer)
                .map_err(|e| invalid(format!("{}: {e}", source.display())))?;
            if count == 0 {
                break;
            }
            output
                .write_all(&buffer[..count])
                .map_err(|e| invalid(format!("{relative}: {e}")))?;
            hash.update(&buffer[..count]);
            bytes = bytes
                .checked_add(count as u64)
                .ok_or_else(|| invalid("copied byte count overflow"))?;
        }
        Ok((format!("{:x}", hash.finalize()), bytes))
    }
}

fn normalized_path(relative: &str) -> Result<String> {
    let path = portable_path(relative)?;
    if path.to_str().is_some_and(|path| {
        path.eq_ignore_ascii_case("manifest.json")
            || path.eq_ignore_ascii_case(".manifest.json.tmp")
    }) {
        return Err(invalid("manifest paths are reserved"));
    }
    Ok(relative.replace('\\', "/"))
}

fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| invalid(e.to_string()))?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(invalid(format!("JSON exceeds {MAX_JSON_BYTES} byte limit")));
    }
    Ok(bytes)
}
