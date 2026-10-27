// `CoreMLError` intentionally carries the model state so a failed load can be
// inspected or retried; boxing that would change the public API, so the
// large-error lint does not apply here.
#![allow(clippy::result_large_err)]

pub mod ffi;
pub mod mlarray;
pub mod mlbatchmodel;
pub mod mlmodel;

// re-exports
pub use ffi::ComputePlatform;
pub use mlmodel::{CoreMLModelOptions, CoreMLModelWithState};
