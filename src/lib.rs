pub mod ffi;
pub mod mlarray;
pub mod mlbatchmodel;
pub mod mlmodel;

// re-exports
pub use ffi::ComputePlatform;
pub use mlmodel::{CoreMLModelOptions, CoreMLModelWithState};
