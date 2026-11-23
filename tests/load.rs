#![cfg(target_os = "macos")]

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    os::unix::{ffi::OsStrExt, fs::MetadataExt},
    path::{Path, PathBuf},
};

use coreml_rs::{
    ffi::{self, LoadError},
    mlarray::{MLArray, MLDataType},
    mlbatchmodel::CoreMLBatchModelWithState,
    mlmodel::{CoreMLError, CoreMLModelLoader},
    ComputePlatform, CoreMLModelOptions, CoreMLModelWithState,
};
use ndarray::{ArrayD, IxDyn};
use sha2::{Digest, Sha256};

fn options() -> CoreMLModelOptions {
    CoreMLModelOptions {
        compute_platform: ComputePlatform::Cpu,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Entry {
    inode: u64,
    mode: u32,
    length: u64,
    modified: (i64, i64),
    digest: Option<[u8; 32]>,
}

fn snapshot(root: &Path) -> BTreeMap<PathBuf, Entry> {
    fn visit(root: &Path, path: &Path, entries: &mut BTreeMap<PathBuf, Entry>) {
        let metadata = fs::symlink_metadata(path).unwrap();
        let digest = if metadata.file_type().is_symlink() {
            Some(Sha256::digest(fs::read_link(path).unwrap().as_os_str().as_bytes()).into())
        } else if metadata.is_file() {
            Some(Sha256::digest(fs::read(path).unwrap()).into())
        } else {
            None
        };
        entries.insert(
            path.strip_prefix(root).unwrap().to_path_buf(),
            Entry {
                inode: metadata.ino(),
                mode: metadata.mode(),
                length: metadata.len(),
                modified: (metadata.mtime(), metadata.mtime_nsec()),
                digest,
            },
        );
        if metadata.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                visit(root, &entry.unwrap().path(), entries);
            }
        }
    }
    let mut entries = BTreeMap::new();
    visit(root, root, &mut entries);
    entries
}

fn assert_single_path(model: &CoreMLModelWithState, expected: &Path) {
    let CoreMLModelWithState::Unloaded(_, CoreMLModelLoader { path }) = model else {
        panic!("expected an unloaded model with its original compiled path");
    };
    assert_eq!(path, expected);
}

fn assert_batch_path(model: &CoreMLBatchModelWithState, expected: &Path) {
    let CoreMLBatchModelWithState::Unloaded(_, CoreMLModelLoader { path }) = model else {
        panic!("expected an unloaded batch model with its original compiled path");
    };
    assert_eq!(path, expected);
}

#[test]
fn rejects_noncompiled_paths_and_files_without_mutating_them() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("model.mlmodel");
    let package = temp.path().join("model.mlpackage");
    let zip = temp.path().join("model.zip");
    let compiled_file = temp.path().join("file.mlmodelc");
    fs::write(&source, b"source specification must not be compiled").unwrap();
    fs::create_dir(&package).unwrap();
    fs::write(package.join("Manifest.json"), b"caller-owned package").unwrap();
    fs::write(&zip, b"archive must not be extracted").unwrap();
    fs::write(&compiled_file, b"a compiled model must be a directory").unwrap();
    let missing = temp.path().join("missing.mlmodelc");
    let before = snapshot(temp.path());

    for path in [&source, &package, &zip, &compiled_file, &missing] {
        let error = CoreMLModelWithState::new_compiled(path, options())
            .load()
            .unwrap_err();
        let CoreMLError::FailedToLoad(message, model) = error else {
            panic!("expected a compiled-path validation failure, got {error:?}");
        };
        assert!(!message.is_empty());
        assert_single_path(&model, path);
        assert_single_path(&model.unload().unwrap(), path);

        let error = CoreMLBatchModelWithState::new_compiled(path, options())
            .load()
            .unwrap_err();
        let CoreMLError::FailedToBatchLoad(message, model) = error else {
            panic!("expected a batch compiled-path validation failure, got {error:?}");
        };
        assert!(!message.is_empty());
        assert_batch_path(&model, path);
        assert_batch_path(&model.unload().unwrap(), path);
    }
    assert_eq!(snapshot(temp.path()), before);
}

#[test]
fn malformed_bundle_preserves_real_nserror_and_retry_state() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("invalid.mlmodelc");
    fs::create_dir(&path).unwrap();
    fs::write(
        path.join("caller-owned.txt"),
        b"must remain byte-for-byte unchanged",
    )
    .unwrap();
    let before = snapshot(temp.path());

    let mut direct = ffi::modelWithPath(&path, ComputePlatform::Cpu);
    let LoadError::Native {
        domain,
        code,
        message,
    } = direct.load().unwrap_err()
    else {
        panic!("CoreML must reject the existing malformed bundle with an NSError");
    };
    assert!(!domain.is_empty());
    assert!(!message.is_empty());

    let mut model = CoreMLModelWithState::new_compiled(&path, options());
    for _ in 0..2 {
        let error = model.load().unwrap_err();
        let CoreMLError::Native {
            domain: actual_domain,
            code: actual_code,
            message: actual_message,
            model: retry,
        } = error
        else {
            panic!("expected a public structured NSError");
        };
        assert_eq!(
            (&actual_domain, actual_code, &actual_message),
            (&domain, code, &message)
        );
        assert_single_path(&retry, &path);
        model = retry;
    }

    let mut model = CoreMLBatchModelWithState::new_compiled(&path, options());
    for _ in 0..2 {
        let CoreMLError::NativeBatch {
            domain: actual_domain,
            code: actual_code,
            message: actual_message,
            model: retry,
        } = model.load().unwrap_err()
        else {
            panic!("expected a public structured batch NSError");
        };
        assert_eq!(
            (&actual_domain, actual_code, &actual_message),
            (&domain, code, &message)
        );
        assert_batch_path(&retry, &path);
        model = retry;
    }
    assert_eq!(snapshot(temp.path()), before);
}

#[test]
fn retry_rechecks_caller_path_instead_of_regenerating_a_cache() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provided-later.mlmodelc");
    let CoreMLError::FailedToLoad(_, retry) = CoreMLModelWithState::new_compiled(&path, options())
        .load()
        .unwrap_err()
    else {
        panic!("missing bundle must fail filesystem validation");
    };
    assert_single_path(&retry, &path);
    fs::create_dir(&path).unwrap();
    let before = snapshot(temp.path());
    let CoreMLError::Native { model, .. } = retry.load().unwrap_err() else {
        panic!("retry must now reach CoreML and reject the caller's malformed directory");
    };
    assert_single_path(&model, &path);
    assert_eq!(snapshot(temp.path()), before);
}

#[test]
fn filesystem_paths_do_not_interpret_url_characters() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("invalid-µ #%?.MLMODELC");
    fs::create_dir(&path).unwrap();
    let before = snapshot(temp.path());
    let CoreMLError::Native { model, .. } = CoreMLModelWithState::new_compiled(&path, options())
        .load()
        .unwrap_err()
    else {
        panic!("the exact filesystem directory must reach CoreML rather than an interpreted URL");
    };
    assert_single_path(&model, &path);
    assert_eq!(snapshot(temp.path()), before);
}

type Outputs = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn tensor_outputs(
    outputs: HashMap<String, MLArray>,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Outputs {
    assert_eq!(outputs.len(), expected.len());
    outputs
        .into_iter()
        .map(|(name, value)| {
            let shape = value.shape().to_vec();
            assert_eq!(&shape, expected.get(&name).expect("declared output"));
            let values: Vec<f32> = match value {
                MLArray::Float32Array(array) => array.iter().copied().collect(),
                MLArray::Float16Array(array) => array.iter().map(|value| value.to_f32()).collect(),
                _ => panic!("fixture outputs must be floating point tensors"),
            };
            assert!(
                !values.is_empty(),
                "prediction must produce real tensor data"
            );
            assert!(values.iter().all(|value| value.is_finite()));
            (name, (shape, values))
        })
        .collect()
}

fn assert_predictions_equal(actual: &Outputs, expected: &Outputs) {
    assert_eq!(actual.len(), expected.len());
    for (name, (shape, values)) in expected {
        let (actual_shape, actual_values) = actual.get(name).expect("same output names");
        assert_eq!(actual_shape, shape);
        assert_eq!(actual_values.len(), values.len());
        for (&actual, &expected) in actual_values.iter().zip(values) {
            assert!(
                (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                "{name}: {actual} != {expected}"
            );
        }
    }
}

fn fixture_lifecycle(variable: &str) {
    let path = PathBuf::from(
        std::env::var_os(variable)
            .unwrap_or_else(|| panic!("set {variable} to a real compiled fixture")),
    );
    let before = snapshot(&path);
    let mut metadata_model = ffi::modelWithPath(&path, ComputePlatform::Cpu);
    metadata_model
        .load()
        .expect("real fixture must load through the native async URL API");
    let inputs: Vec<_> = metadata_model
        .inputs_info()
        .unwrap()
        .iter()
        .map(|info| {
            assert_eq!(info.dtype, MLDataType::F32, "fixture needs f32 inputs");
            let shape = &info.shape;
            assert!(
                !shape.is_empty() && shape.iter().all(|&dim| dim > 0),
                "fixture needs fixed tensor input shapes"
            );
            let count: usize = shape.iter().product();
            let values = (0..count).map(|index| (index % 17) as f32 / 17.0).collect();
            (
                info.name.clone(),
                ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
            )
        })
        .collect();
    assert!(!inputs.is_empty());
    let outputs: BTreeMap<_, _> = metadata_model
        .outputs_info()
        .unwrap()
        .iter()
        .map(|info| {
            assert_eq!(info.dtype, MLDataType::F32, "fixture needs f32 outputs");
            let shape = info.shape.clone();
            assert!(!shape.is_empty() && shape.iter().all(|&dim| dim > 0));
            (info.name.clone(), shape)
        })
        .collect();
    assert!(!outputs.is_empty());
    metadata_model.unload();
    drop(metadata_model);

    let mut model = CoreMLModelWithState::new_compiled(&path, options());
    assert!(matches!(model.predict(), Err(CoreMLError::ModelNotLoaded)));
    model = model.unload().unwrap().load().unwrap().load().unwrap();
    for (name, input) in &inputs {
        model.add_input(name, input.clone()).unwrap();
    }
    let expected = tensor_outputs(model.predict().unwrap().outputs, &outputs);
    for _ in 0..2 {
        model = model.unload().unwrap();
        assert_single_path(&model, &path);
        assert!(matches!(model.predict(), Err(CoreMLError::ModelNotLoaded)));
        model = model.load().unwrap();
        for (name, input) in &inputs {
            model.add_input(name, input.clone()).unwrap();
        }
        assert_predictions_equal(
            &tensor_outputs(model.predict().unwrap().outputs, &outputs),
            &expected,
        );
    }
    drop(model.unload().unwrap());

    let mut batch = CoreMLBatchModelWithState::new_compiled(&path, options());
    assert!(matches!(batch.predict(), Err(CoreMLError::ModelNotLoaded)));
    for _ in 0..2 {
        batch = batch.load().unwrap().load().unwrap();
        for index in 0..2 {
            for (name, input) in &inputs {
                batch.add_input(name, input.clone(), index).unwrap();
            }
        }
        let result = batch.predict().unwrap();
        assert_eq!(result.outputs.len(), 2);
        for result in result.outputs {
            assert_predictions_equal(&tensor_outputs(result, &outputs), &expected);
        }
        batch = batch.unload().unwrap();
        assert_batch_path(&batch, &path);
    }
    drop(batch);
    assert_eq!(
        snapshot(&path),
        before,
        "caller-owned bundle must never be rewritten or deleted"
    );
}

#[test]
#[ignore = "requires COREML_COMPILED_FIXTURE pointing to a real compiled model"]
fn compiled_fixture_lifecycle() {
    fixture_lifecycle("COREML_COMPILED_FIXTURE");
}
