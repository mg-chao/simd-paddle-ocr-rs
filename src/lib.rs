#![feature(portable_simd)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod builder;
pub mod characters;
pub mod cpu;
pub mod crop;
mod db;
pub mod detector;
pub mod engine;
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
#[cfg(feature = "profiling")]
pub use profile::StageProfile;
pub mod recognizer;
pub mod runtime;
pub mod tensor;
pub use builder::{ModelPreset, OcrBuilder, OcrModelSet};
pub use characters::{CharacterBox, OwnedCharacterBox};
