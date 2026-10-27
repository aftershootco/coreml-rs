use std::{path::PathBuf, str::FromStr};

use coreml_rs::{mlmodel::CoreMLError, CoreMLModelOptions, CoreMLModelWithState};
use sha2::{Digest, Sha256};

#[test]
pub fn load_empty() {
    let m = CoreMLModelWithState::from_buf(vec![], CoreMLModelOptions::default());
    let res = m.load();
    assert!(matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
}

/// Reads a fixture from the (gitignored) `demo` directory. Tests that need one
/// report a skip instead of failing with ENOENT when it is absent.
fn read_demo(name: &str) -> Option<Vec<u8>> {
    let path = PathBuf::from("./demo").join(name);
    match std::fs::read(&path) {
        Ok(buf) => Some(buf),
        Err(err) => {
            eprintln!("skipping: {} is unavailable ({err})", path.display());
            None
        }
    }
}

pub fn unzip_to_path_from_hash(buf: &[u8]) -> Option<PathBuf> {
    fn get_cache_filename(model_buffer: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_buffer);
        let hash = hasher.finalize();
        format!("{:x}.cache", hash)
    }
    let name = get_cache_filename(buf);

    // writing to tmp allows faster processing and usage as it's part of swap based in-memory tmpfs
    let path = PathBuf::from_str("/tmp/coreml-aftershoot/").ok()?;
    let path = path.join(name);

    let m = path.join("model.mlpackage");
    if m.exists() {
        return None;
    }

    _ = std::fs::remove_dir_all(&path);
    _ = std::fs::remove_file(&path);
    let mut res = zip::ZipArchive::new(std::io::Cursor::new(buf)).ok()?;
    res.extract(&path).ok()?;

    if m.exists() {
        Some(m)
    } else {
        None
    }
}

#[test]
pub fn reload_from_compiled_path() {
    let Some(buf) = read_demo("model.zip") else {
        return;
    };
    let Some(path) = unzip_to_path_from_hash(&buf) else {
        return;
    };
    let m = CoreMLModelWithState::new(&path, CoreMLModelOptions::default());
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    let m = res.unwrap();
    let res = m.unload();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    _ = std::fs::remove_dir_all(path);
    let m = res.unwrap();
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
}

#[test]
pub fn reload_from_buf() {
    let Some(buf) = read_demo("model_3.mlmodel") else {
        return;
    };
    let m = CoreMLModelWithState::from_buf(buf, CoreMLModelOptions::default());
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    let m = res.unwrap();
    let res = m.unload();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
    let m = res.unwrap();
    let res = m.load();
    assert!(!matches!(res, Err(CoreMLError::FailedToLoadStatic(_, _))));
}
