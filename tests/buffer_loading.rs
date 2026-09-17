use coreml_rs::{
    mlarray::MLArray,
    mlbatchmodel::{CoreMLBatchModel, CoreMLBatchModelWithState},
    mlmodel::{CoreMLError, CoreMLModel, CoreMLModelInfo, CoreMLModelLoader},
    ComputePlatform, CoreMLModelOptions, CoreMLModelWithState,
};
use std::{
    process::Command,
    time::{Duration, Instant},
};

#[path = "support/allocation.rs"]
mod allocation;
#[path = "support/model_spec.rs"]
mod model_spec;
use model_spec::model_spec;

#[global_allocator]
static ALLOCATOR: allocation::TrackingAllocator = allocation::TrackingAllocator;

// Core ML failures can abort or hang the process. Run each scenario in a child
// so the parent can report the failure and reap it, including on a deadlock.
fn isolated(name: &str, test: impl FnOnce()) {
    if std::env::var("COREML_BUFFER_TEST").as_deref() == Ok(name) {
        test();
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("COREML_BUFFER_TEST", name)
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "{name}: {status}");
            return;
        }
        if start.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{name}: model loading exceeded 30 seconds");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn options() -> CoreMLModelOptions {
    CoreMLModelOptions {
        compute_platform: ComputePlatform::Cpu,
        ..Default::default()
    }
}

#[test]
fn buffer_ownership() {
    isolated("buffer_ownership", || {
        for batch in [false, true] {
            for spare in [0, 16] {
                let bytes = model_spec();
                let mut buffer = Vec::with_capacity(bytes.len() + spare);
                buffer.extend_from_slice(&bytes);
                allocation::watch(&buffer);
                let info = CoreMLModelInfo { opts: options() };
                if batch {
                    let model = CoreMLBatchModel::load_buffer(buffer, info);
                    allocation::assert_live();
                    drop(model);
                } else {
                    let model = CoreMLModel::load_buffer(buffer, info);
                    allocation::assert_live();
                    drop(model);
                }
                allocation::assert_freed();
            }
        }
    });
}

#[test]
fn rejected_buffer_is_freed() {
    isolated("rejected_buffer_is_freed", || {
        for batch in [false, true] {
            let mut buffer = Vec::with_capacity(272);
            buffer.extend_from_slice(&[0xff; 256]);
            allocation::watch(&buffer);
            let info = CoreMLModelInfo { opts: options() };
            if batch {
                drop(CoreMLBatchModel::load_buffer(buffer, info));
            } else {
                drop(CoreMLModel::load_buffer(buffer, info));
            }
            allocation::assert_freed();
        }
    });
}

#[test]
fn load_failure_returns() {
    isolated("load_failure_returns", || {
        let single = CoreMLModelWithState::from_buf(vec![], options()).load();
        assert!(matches!(
            single,
            Err(CoreMLError::FailedToLoadStatic(
                _,
                CoreMLModelWithState::Unloaded(_, _)
            ))
        ));
        let batch = CoreMLBatchModelWithState::from_buf(vec![], options()).load();
        assert!(matches!(
            batch,
            Err(CoreMLError::FailedToLoadBatchStatic(
                _,
                CoreMLBatchModelWithState::Unloaded(_, _)
            ))
        ));
    });
}

#[test]
fn disk_cache_load_failure_returns_unloaded() {
    isolated("disk_cache_load_failure_returns_unloaded", || {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let encoder = flate2::write::ZlibEncoder::new(&mut file, flate2::Compression::fast());
        encoder.finish().unwrap();
        let path = file.path().to_path_buf();
        let single = CoreMLModelWithState::Unloaded(
            CoreMLModelInfo { opts: options() },
            CoreMLModelLoader::BufferToDisk(path.clone()),
        )
        .load();
        assert!(
            matches!(single, Err(CoreMLError::FailedToLoadStatic(_, CoreMLModelWithState::Unloaded(_, CoreMLModelLoader::BufferToDisk(ref p)))) if p == &path)
        );
        let batch = CoreMLBatchModelWithState::Unloaded(
            CoreMLModelInfo { opts: options() },
            CoreMLModelLoader::BufferToDisk(path.clone()),
        )
        .load();
        assert!(
            matches!(batch, Err(CoreMLError::FailedToLoadBatchStatic(_, CoreMLBatchModelWithState::Unloaded(_, CoreMLModelLoader::BufferToDisk(ref p)))) if p == &path)
        );
    });
}

fn assert_output(output: &MLArray, expected: f32) {
    let MLArray::Float32Array(array) = output else {
        panic!("expected FLOAT32 output")
    };
    assert_eq!(array.as_slice().unwrap(), &[expected]);
}

#[test]
fn buffer_predict_and_reload() {
    isolated("buffer_predict_and_reload", || {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = options();
        opts.cache_dir = dir.path().to_path_buf();
        let mut model = CoreMLModelWithState::from_buf(model_spec(), opts)
            .load()
            .unwrap();
        for round in 0..3 {
            let input = ndarray::arr1(&[7.0f32]).into_dyn();
            model.add_input("input", input).unwrap();
            assert_output(&model.predict().unwrap().outputs["output"], 7.0);
            model = if round == 0 {
                model.unload().unwrap()
            } else {
                model.unload_to_disk().unwrap()
            }
            .load()
            .unwrap();
        }
    });
}

#[test]
fn batch_buffer_predict_and_reload() {
    isolated("batch_buffer_predict_and_reload", || {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = options();
        opts.cache_dir = dir.path().to_path_buf();
        let mut model = CoreMLBatchModelWithState::from_buf(model_spec(), opts)
            .load()
            .unwrap();
        for round in 0..3 {
            for (i, value) in [-2.0f32, 7.0].into_iter().enumerate() {
                model
                    .add_input("input", ndarray::arr1(&[value]).into_dyn(), i as isize)
                    .unwrap();
            }
            let output = model.predict().unwrap();
            assert_eq!(output.outputs.len(), 2);
            assert_output(&output.outputs[0]["output"], 0.0);
            assert_output(&output.outputs[1]["output"], 7.0);
            model = if round == 0 {
                model.unload().unwrap()
            } else {
                model.unload_to_disk().unwrap()
            }
            .load()
            .unwrap();
        }
    });
}
