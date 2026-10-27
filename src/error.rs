//! Categories shared by the public stage-specific errors.

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidInput,
    InvalidModel,
    Unsupported,
    ResourceLimit,
    Allocation,
    Io,
    Execution,
    DeviceLost,
    GpuValidation,
    GpuMapping,
}

/// OCR stage identity used in errors and execution reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrStage {
    Detector,
    Classifier,
    Recognizer,
    Shared,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ErrorOrigin {
    Operation,
    Callback,
}

// Keep stage-specific types while preserving the concrete underlying error.
macro_rules! stage_error {
    ($name:ident, $default:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            kind: crate::error::ErrorKind,
            message: String,
            origin: ErrorOrigin,
            stage: Option<OcrStage>,
            source: Option<std::sync::Arc<dyn std::error::Error + Send + Sync>>,
        }
        impl $name {
            pub fn new(message: impl Into<String>) -> Self {
                Self::with_kind(crate::error::ErrorKind::$default, message)
            }
            pub fn with_kind(kind: crate::error::ErrorKind, message: impl Into<String>) -> Self {
                Self {
                    kind,
                    origin: ErrorOrigin::Operation,
                    stage: None,
                    message: message.into(),
                    source: None,
                }
            }
            pub fn caused_by(
                kind: crate::error::ErrorKind,
                message: impl Into<String>,
                source: impl std::error::Error + Send + Sync + 'static,
            ) -> Self {
                Self {
                    kind,
                    origin: ErrorOrigin::Operation,
                    stage: None,
                    message: message.into(),
                    source: Some(std::sync::Arc::new(source)),
                }
            }
            pub fn origin(&self) -> ErrorOrigin {
                self.origin
            }
            pub fn stage(&self) -> Option<OcrStage> {
                self.stage
            }
            pub fn at_stage(mut self, stage: OcrStage) -> Self {
                self.stage = Some(stage);
                self
            }
            pub fn callback(mut self) -> Self {
                self.origin = ErrorOrigin::Callback;
                self
            }
            pub fn kind(&self) -> crate::error::ErrorKind {
                self.kind
            }
            pub fn context(mut self, context: impl std::fmt::Display) -> Self {
                self.message = format!("{context}: {}", self.message);
                self
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.message)?;
                if let Some(source) = &self.source {
                    write!(f, ": {source}")?;
                }
                Ok(())
            }
        }
        impl std::error::Error for $name {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.source.as_deref().map(|s| s as &dyn std::error::Error)
            }
        }
        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }
        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }
    };
}
stage_error!(RuntimeError, Execution);
stage_error!(ImageError, InvalidInput);
stage_error!(TensorError, InvalidInput);
stage_error!(ImportError, InvalidModel);

macro_rules! convert_error {
    ($source:ty, $target:ty, $context:literal) => {
        impl From<$source> for $target {
            fn from(source: $source) -> Self {
                let origin = source.origin();
                let stage = source.stage();
                let mut error = Self::caused_by(source.kind(), $context, source);
                error.origin = origin;
                error.stage = stage;
                error
            }
        }
    };
}
convert_error!(TensorError, RuntimeError, "tensor");
convert_error!(TensorError, ImportError, "tensor");
convert_error!(ImageError, RuntimeError, "image stage");
convert_error!(RuntimeError, ImageError, "runtime");
impl From<std::io::Error> for ImportError {
    fn from(error: std::io::Error) -> Self {
        Self::caused_by(ErrorKind::Io, "I/O", error)
    }
}
impl RuntimeError {
    pub(crate) fn allows_fallback(&self) -> bool {
        self.origin == ErrorOrigin::Operation
            && matches!(self.kind, ErrorKind::Unsupported | ErrorKind::DeviceLost)
    }
}

/// Serializable provenance; the concrete source chain remains on the error itself.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FailureProvenance {
    pub kind: ErrorKind,
    pub origin: ErrorOrigin,
    pub stage: Option<OcrStage>,
    pub message: String,
}
impl From<&RuntimeError> for FailureProvenance {
    fn from(error: &RuntimeError) -> Self {
        Self {
            kind: error.kind(),
            origin: error.origin(),
            stage: error.stage(),
            message: error.to_string(),
        }
    }
}
