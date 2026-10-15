//! Safe scalar PaddleOCR inference from borrowed image pixels.

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod crop;
mod db;
pub mod detector;
pub mod image;
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
pub mod recognizer;
pub mod runtime;
pub mod tensor;
