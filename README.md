## coreml-rs

`coreml-rs` is an experimental Rust library providing bindings for Apple's Core ML framework.
Core ML is Apple's machine learning framework for integrating machine learning models into macOS applications.

The crate talks to the Objective-C framework directly through [`objc2`](https://crates.io/crates/objc2) and
[`objc2-core-ml`](https://crates.io/crates/objc2-core-ml). There is **no Swift runtime dependency and no Swift
toolchain requirement**: building needs only a Rust toolchain and the macOS SDK, and the resulting binary links
only system frameworks.

### Requirements

- macOS 13 or newer (the CoreML features this crate uses, such as `MLModelAsset`, require it).
- The macOS SDK with the CoreML framework.

> Earlier versions linked a Swift static library and required `libswift_Concurrency.dylib` (and, on Intel, a
> correctly deployed Swift runtime). That is no longer the case.

## Status

This project is currently work in progress.
The primary goal is to enable Rust developers to utilize Core ML models within their applications, leveraging Rust's performance and safety features for the rest of the ML infrastructure.

## Roadmap

- Cleanup & fix bugs with the types and allow more input formats.
- Build zerocopy types for more efficiently passing inputs and outputs.
- Provide more configuration options for models.

## Features

- **Model Loading**: Load Core ML models into Rust applications, either from disk or from an in-memory buffer.
- **Inference**: Perform inference using loaded models, with caller-owned output buffers (zero copy).
- **Data Handling**: `f32`, `f16` (as `u16`), and `i32` tensor inputs; `f32` and `f16` outputs.

## Installation

To include `coreml-rs` in your project, add the following to your `Cargo.toml` dependencies:
```toml
[dependencies]
coreml-rs = { git = "https://github.com/aftershootco/coreml-rs" }
```

## Usage

### Simple Inference

Load a Core ML model from a `.mlmodel` file and perform a single inference:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, Array4};

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let model_options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
        ..Default::default()
    };

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict().unwrap();
    for (_name, value) in output.outputs {
        let _output: Array<f32, _> = value.extract_to_tensor();
    }

    // Use output as needed
}
```

`CoreMLModelWithState::new` accepts a `.mlmodel`/`.mlpackage` that is compiled on load, and
`CoreMLModelWithState::new_compiled` accepts an already compiled `.mlmodelc`. Use
`CoreMLModelWithState::compile_to(package, dest)` to compile once into a permanent location and load it
later with `new_compiled`.

### Batch Inference

Batch prediction uses `CoreMLBatchModelWithState`:

```rust
use coreml_rs::{mlbatchmodel::CoreMLBatchModelWithState, ComputePlatform, CoreMLModelOptions};
use ndarray::{Array, Array4};

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let model_options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
        ..Default::default()
    };

    let mut model = CoreMLBatchModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    // Add one input per batch element.
    for index in 0..10 {
        let _ = model.add_input("image", input.clone().into_dyn(), index);
    }

    let output = model.predict().unwrap();
    for element in output.outputs {
        for (_name, value) in element {
            let _output: Array<f32, _> = value.extract_to_tensor();
        }
    }
}
```

### Loading from a Zip Archive

Load a model from a zip archive containing an `.mlpackage`:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, Array4};
use std::path::PathBuf;

fn unzip_to_path_from_hash(buf: &[u8]) -> Option<PathBuf> {
    fn get_cache_filename(model_buffer: &[u8]) -> String {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(model_buffer);
        let hash = hasher.finalize();
        format!("{:x}.mlpackage", hash)
    }
    let name = get_cache_filename(buf);

    let path = PathBuf::from("/tmp/coreml-aftershoot/");
    let path = path.join(name);
    _ = std::fs::remove_dir_all(&path);
    _ = std::fs::remove_file(&path);

    let mut res = zip::ZipArchive::new(std::io::Cursor::new(buf)).ok()?;
    res.extract(&path).ok()?;

    let m = path.join("model.mlpackage");
    if m.exists() {
        Some(m)
    } else {
        None
    }
}

pub fn main() {
    let buf = std::fs::read("./demo/model_2.zip").unwrap();

    let model_path = unzip_to_path_from_hash(&buf).unwrap();

    let model_options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
        ..Default::default()
    };

    let mut model = CoreMLModelWithState::new(model_path, model_options).load().unwrap();

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict().unwrap();
    for (_name, value) in output.outputs {
        let _output: Array<f32, _> = value.extract_to_tensor();
    }

    // Use output as needed
}
```

### Memory Management

Unload and reload models to manage memory usage:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let model_options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
        ..Default::default()
    };

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.clone().into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let _ = model.predict();

    // Unload model to free memory
    let unloaded_model = model.unload().unwrap();

    // Later, reload the model (from the compiled cache, without recompiling)
    let mut model = unloaded_model.load().unwrap();

    // Add input again and predict
    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let _ = model.predict();

    // Use output as needed
}
```

**Note**: These examples assume specific model inputs/outputs. Adjust based on your model's specifications.

## Contributing

Contributions are welcome!
If you have experience with Core ML and Rust, consider helping to advance this project.
