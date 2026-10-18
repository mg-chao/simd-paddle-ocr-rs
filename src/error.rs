//! Categories shared by the public stage-specific errors.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidInput,
    InvalidModel,
    Unsupported,
    ResourceLimit,
    Allocation,
    Io,
    Execution,
}

// Keep stage-specific types while preserving the concrete underlying error.
macro_rules! stage_error {
    ($name:ident, $default:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            kind: crate::error::ErrorKind,
            message: String,
            source: Option<std::sync::Arc<dyn std::error::Error + Send + Sync>>,
        }
        impl $name {
            pub fn new(message: impl Into<String>) -> Self {
                Self::with_kind(crate::error::ErrorKind::$default, message)
            }
            pub fn with_kind(kind: crate::error::ErrorKind, message: impl Into<String>) -> Self {
                Self {
                    kind,
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
                    message: message.into(),
                    source: Some(std::sync::Arc::new(source)),
                }
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
pub(crate) use stage_error;
