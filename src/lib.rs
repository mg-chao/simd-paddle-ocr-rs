#![cfg_attr(
    all(target_arch = "aarch64", target_os = "macos"),
    feature(stdarch_aarch64_feature_detection)
)]
#![feature(portable_simd)]
// wgpu's nested native resources require deeper Send/Sync trait evaluation.
#![cfg_attr(feature = "gpu", recursion_limit = "256")]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod builder;
pub mod characters;
pub mod cpu;
pub mod crop;
mod db;
pub mod detector;
pub mod engine;
pub mod error;
pub mod image;
pub mod lines;
mod math;
pub mod model;
pub mod ocr;
pub mod onnx;
pub mod orientation;
mod pixel_kernels;
pub mod preprocess;
mod profile;
mod projection;
#[cfg(feature = "profiling")]
pub use profile::{RecognitionOwnerProfile, StageProfile};

#[cfg(all(test, feature = "gpu"))]
mod gpu_integration_tests;
pub mod recognizer;
pub mod runtime;
pub mod tensor;
pub use builder::{ModelPreset, OcrBuilder, OcrModelSet};
pub use characters::{CharacterBox, OwnedCharacterBox};
#[cfg(feature = "gpu")]
pub use runtime::gpu::{GpuPipelineMode, GpuPipelineStatistics, GpuPlacement};
pub use runtime::{CompiledModel, ExecutionBackend, ExecutionConfig, ExecutionTarget, Session};

pub use error::OcrStage;
pub use ocr::{OcrExecution, OcrPlan, OcrRequest, RunOptions, RunStatistics};
pub use runtime::{
    CompileOptions, Compiler, CompilerOptions, CpuCompileOptions, ExecutionCounters, MemorySnapshot,
};
