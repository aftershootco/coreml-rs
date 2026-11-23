## coreml-rs

Rust bindings for Apple's Core ML framework, implemented directly through
[`objc2`](https://crates.io/crates/objc2) and
[`objc2-core-ml`](https://crates.io/crates/objc2-core-ml). Building needs a Rust
toolchain and the macOS SDK, not Swift or a Swift runtime dependency.

### Requirements

- macOS 13 or newer and the macOS SDK with the CoreML framework.
- A caller-provided, already compiled **`.mlmodelc` directory**.
- For upstream-encrypted bundles, the deployment's valid CoreML key authorization.

## Compiled-bundle-only loading (breaking change)

`CoreMLModelWithState::new_compiled(path, options)` and
`CoreMLBatchModelWithState::new_compiled(path, options)` are the only model
constructors. The path must be a filesystem path to an existing `.mlmodelc`
directory, not a file URL, a source `.mlmodel`/`.mlpackage`, a specification buffer,
or an archive. CoreML itself validates the compiled contents.

Compile and, when required, encrypt models **upstream**, before handing their
paths to this crate. Loading uses CoreML's asynchronous URL API behind the
synchronous Rust `load` method, allowing CoreML to perform its native encrypted
model authorization flow. A `.mlmodelc` suffix does **not** establish that a
bundle is encrypted. This crate neither implements encryption nor infers it
from the filename.

The caller owns the bundle and its lifetime. This crate never compiles,
extracts, caches, regenerates, rewrites, or deletes model paths. Keep the
compiled directory available and unchanged for every reload. If it is removed,
reload fails rather than falling back to a source model or regenerating a cache.

Removed APIs: `new`, `from_buf`, `from_package`, `from_package_dir`, `compile_to`,
`unload_to_disk`, and `CoreMLModelOptions::cache_dir`. `CoreMLModelLoader` is now a
struct with a public `path: PathBuf` field. The corresponding native source/buffer/package and
compilation helpers have also been removed. Migrate callers to provision their
compiled bundle upstream and call `new_compiled(...).load()` explicitly.

## Installation

```toml
[dependencies]
coreml-rs = { git = "https://github.com/aftershootco/coreml-rs" }
```

## Single prediction

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, Array4};

fn main() {
    let options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
    };
    let mut model = CoreMLModelWithState::new_compiled("./models/image.mlmodelc", options)
        .load()
        .unwrap();

    let input = Array4::<f32>::ones((1, 3, 512, 512)).into_dyn();
    model.add_input("image", input).unwrap();
    for (_name, value) in model.predict().unwrap().outputs {
        // This example assumes the model declares f32 outputs.
        let _output: Array<f32, _> = value.extract_to_tensor().expect("expected an f32 output");
    }
}
```

Adjust feature names, shapes, and element types for your model. Native tensor
inputs and outputs support `f32`, `f16`, and `i32`. Caller-owned output backings
avoid output copies where CoreML supports them; independently returned native
storage is validated and copied when CoreML declines a proposed backing.

### Tensor safety and metadata

`MLType` is sealed to the supported concrete scalar types. Moving an array into
`MLArray` preserves its allocation and logical layout; `extract_to_tensor::<T>()`
now returns a `Result` and rejects a different scalar type rather than
reinterpreting the allocation. This is a breaking return-type change.

Loaded single and batch wrappers expose native typed tensor metadata rather
than relying on formatted feature descriptions. Input validation uses the
declared scalar type and native fixed, enumerated, ranged, or unconstrained
shape constraints. Default dimensions do not restrict an unconstrained feature.
Unsupported feature kinds return errors.

The string-based `description()` APIs have been removed. Use `inputs_info()`
and `outputs_info()` for typed model metadata. `TensorInfo` and `ShapeConstraint`
implement `Display`: use `println!("{info}")` to print a feature's name, dtype,
default shape, accepted shape constraints, and optionality. Ranges print inclusive
upper bounds as `1..=4` and unbounded upper bounds as `1..`. Derived `Debug`
remains available; neither format describes a prediction's actual tensor values.

Nonstandard input layouts are normalized to logical row-major order when
necessary. Each prediction consumes a fresh request: inputs and output backings
are cleared on success and failure, so incomplete or invalid requests cannot
inherit features from a previous call. Output extraction checks actual shape,
dtype and completeness; output conversion failures are errors, not partial
successful predictions.

`CoreMLModel::add_output_f32` and `add_output_f16` now return
`Result<(), CoreMLError>` rather than a boolean. Native binding errors propagate
to the caller, and request buffers transfer ownership as typed vectors rather
than untracked raw pointers.

## Batch prediction

```rust
use coreml_rs::{mlbatchmodel::CoreMLBatchModelWithState, ComputePlatform, CoreMLModelOptions};
use ndarray::Array4;

fn main() {
    let options = CoreMLModelOptions { compute_platform: ComputePlatform::Cpu };
    let mut model = CoreMLBatchModelWithState::new_compiled("./models/image.mlmodelc", options)
        .load()
        .unwrap();
    for index in 0..2 {
        model.add_input("image", Array4::<f32>::ones((1, 3, 512, 512)).into_dyn(), index).unwrap();
    }
    let output = model.predict().unwrap();
    assert_eq!(output.outputs.len(), 2);
}
```

Batch requests use the same native tensor dtype and shape validation.

## Unload, reload, and retry

`unload` releases the live model and inputs/outputs, retaining the same
caller-owned `CoreMLModelLoader::path`. It performs no disk writes. It is safe to unload an
already unloaded model; loading an already loaded model is also idempotent.

```rust
# use coreml_rs::{CoreMLModelOptions, CoreMLModelWithState};
# fn main() -> Result<(), coreml_rs::mlmodel::CoreMLError> {
let model = CoreMLModelWithState::new_compiled("./models/image.mlmodelc", CoreMLModelOptions::default())
    .load()?;
let unloaded = model.unload()?;
let mut reloaded = unloaded.load()?;
// Add inputs again before predicting.
# Ok(())
# }
```

Filesystem/format validation failures return `CoreMLError::FailedToLoad` (or
`FailedToBatchLoad`) with an unloaded model for retry. Native load failures
return `CoreMLError::Native { domain, code, message, model }` (or `NativeBatch`),
preserving the NSError domain, numeric code, localized message, and unloaded
retry state. These fields are returned to the caller, not merely logged. No
native failure causes automatic recompilation or retries; the caller decides
whether to retry the returned `model` after addressing the cause.

## Examples and tests

`examples/simple.rs` and `examples/batch.rs` accept a compiled directory as their
first argument and assume an `image` input of shape `[1, 3, 512, 512]`.

The default integration tests exercise real filesystem validation, real CoreML
NSError propagation, retry state, literal filesystem paths, and caller-owned directory
immutability. They require no model download or compilation.

The opt-in compiled-fixture test exercises real CPU predictions, single and batch
unload/reload cycles, and bundle immutability. The fixture must have fixed-shape
`f32` tensor inputs and outputs. No mock predictions or model compilation occur
in this test. Explicitly selecting it fails if the required fixture is not
provided; it does not silently skip:

```sh
COREML_COMPILED_FIXTURE=/absolute/path/model.mlmodelc \
  cargo test --test load compiled_fixture_lifecycle -- --ignored
```

End-to-end Apple encryption coverage lives in inference-engine's Rust-generated
fixture tests, which compile and encrypt a temporary model using an external
Apple-issued key. See that crate's README for test-host signing and key retrieval
requirements; no encryption key is embedded in this crate.

## Contributing

Contributions are welcome.
