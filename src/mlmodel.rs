use crate::{
    ffi::{modelWithAssets, modelWithPackage, modelWithPath, ComputePlatform, Model},
    mlarray::MLArray,
    mlbatchmodel::CoreMLBatchModelWithState,
};
use flate2::Compression;
use ndarray::Array;
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

pub use crate::ffi::MLModelOutput;

use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreMLError {
    #[error("CoreML Cache IoError: {0}")]
    IoError(std::io::Error),
    #[error("BadInputShape: {0}")]
    BadInputShape(String),
    // #[error("Lz4 Decompression Error: {0}")]
    // Lz4DecompressError(DecompressError),
    #[error("UnknownError: {0}")]
    UnknownError(String),
    #[error("UnknownError: {0}")]
    UnknownErrorStatic(&'static str),
    #[error("ModelNotLoaded: coreml model not loaded into session")]
    ModelNotLoaded,
    #[error("FailedToLoad: coreml model couldn't be loaded: {0}")]
    FailedToLoadStatic(&'static str, CoreMLModelWithState),
    #[error("FailedToLoad: coreml model couldn't be loaded: {0}")]
    FailedToLoad(String, CoreMLModelWithState),
    #[error("FailedToLoadBatch: coreml model couldn't be loaded: {0}")]
    FailedToLoadBatchStatic(&'static str, CoreMLBatchModelWithState),
    #[error("FailedToLoadBatch: coreml model couldn't be loaded: {0}")]
    FailedToBatchLoad(String, CoreMLBatchModelWithState),
}

#[derive(Default, Clone)]
pub struct CoreMLModelOptions {
    pub compute_platform: ComputePlatform,
    pub cache_dir: PathBuf,
}

impl std::fmt::Debug for CoreMLModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMLModelOptions")
            .field(
                "compute_platform",
                match self.compute_platform {
                    ComputePlatform::Cpu => &"CPU",
                    ComputePlatform::CpuAndANE => &"CpuAndAne",
                    ComputePlatform::CpuAndGpu => &"CpuAndGpu",
                },
            )
            .finish()
    }
}

#[derive(Debug)]
pub enum CoreMLModelLoader {
    /// Model to be loaded from the given path
    ModelPath(PathBuf),
    /// Model cache built and stored at path, to be used for faster reload
    CompiledPath(PathBuf),
    /// Model with buffer to manage the buffer locally
    Buffer(Vec<u8>),
    BufferToDisk(PathBuf),
    /// In-memory `.mlpackage`: the model specification plus its weight blobs,
    /// keyed by the file reference used in the spec (`weights/weight.bin`).
    Package {
        spec: Vec<u8>,
        blobs: Vec<(String, Vec<u8>)>,
    },
}

// The `Loaded` variant carries the whole live model; boxing it would change the
// public enum's shape, and callers match on it directly.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum CoreMLModelWithState {
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    Loaded(CoreMLModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl CoreMLModelWithState {
    pub fn new(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader::ModelPath(path.as_ref().to_path_buf()),
        )
    }
    pub fn new_compiled(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader::CompiledPath(path.as_ref().to_path_buf()),
        )
    }

    pub fn from_buf(buf: Vec<u8>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(CoreMLModelInfo { opts }, CoreMLModelLoader::Buffer(buf))
    }

    /// Builds a model from an in-memory `.mlpackage`: the `model.mlmodel`
    /// specification plus its weight blobs.
    ///
    /// `blobs` maps the weight file reference used by the spec (usually
    /// `weights/weight.bin`, with or without the `@model_path/` prefix) to the
    /// file contents. This avoids materialising the package on disk.
    ///
    /// Requires macOS 15 or newer; on older systems loading fails and the
    /// package must be loaded from its directory with [`Self::new`].
    pub fn from_package(
        spec: Vec<u8>,
        blobs: Vec<(String, Vec<u8>)>,
        opts: CoreMLModelOptions,
    ) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader::Package { spec, blobs },
        )
    }

    /// Reads a `.mlpackage` directory into memory and loads it with
    /// [`Self::from_package`]. Requires macOS 15 or newer.
    pub fn from_package_dir(
        package: impl AsRef<Path>,
        opts: CoreMLModelOptions,
    ) -> Result<Self, CoreMLError> {
        let (spec, blobs) = read_package(package.as_ref())?;
        Ok(Self::from_package(spec, blobs, opts))
    }

    /// Compiles the model package at `package` into a `.mlmodelc` at `dest`, replacing
    /// whatever is there. Unlike [`Self::new`] (which compiles into a system temp dir on
    /// every load), the result persists, so callers can later use [`Self::new_compiled`]
    /// and skip compilation entirely.
    pub fn compile_to(
        package: impl AsRef<Path>,
        dest: impl AsRef<Path>,
    ) -> Result<(), CoreMLError> {
        match crate::ffi::compileModelTo(
            package.as_ref().display().to_string(),
            dest.as_ref().display().to_string(),
        ) {
            None => Ok(()),
            Some(err) => Err(CoreMLError::UnknownError(format!(
                "failed to compile model to {}: {err}",
                dest.as_ref().display()
            ))),
        }
    }

    pub fn load(self) -> Result<Self, CoreMLError> {
        let Self::Unloaded(info, loader) = self else {
            return Ok(self);
        };
        match loader {
            CoreMLModelLoader::ModelPath(path_buf) => {
                let mut coreml_model = CoreMLModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    false,
                );
                if !coreml_model.model.load() {
                    return Err(CoreMLError::FailedToLoadStatic(
                        "Failed to load model; model path not valid",
                        Self::Unloaded(info, CoreMLModelLoader::ModelPath(path_buf)),
                    ));
                }
                Ok(Self::Loaded(
                    coreml_model,
                    info,
                    CoreMLModelLoader::ModelPath(path_buf),
                ))
            }
            CoreMLModelLoader::CompiledPath(path_buf) => {
                let mut coreml_model =
                    CoreMLModel::load_from_path(path_buf.display().to_string(), info.clone(), true);
                if !coreml_model.model.load() {
                    return Err(CoreMLError::FailedToLoadStatic(
                        "Failed to load model; compiled model cache got purged",
                        Self::Unloaded(info, CoreMLModelLoader::CompiledPath(path_buf)),
                    ));
                }
                Ok(Self::Loaded(
                    coreml_model,
                    info,
                    CoreMLModelLoader::CompiledPath(path_buf),
                ))
            }
            CoreMLModelLoader::Buffer(vec) => {
                let mut coreml_model = CoreMLModel::load_buffer(vec.clone(), info.clone());
                coreml_model.model.load();
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadStatic(
                        "Failed to load model; likely not a CoreML mlmodel file",
                        Self::Unloaded(info, CoreMLModelLoader::Buffer(vec)),
                    ));
                }
                let loader = CoreMLModelLoader::Buffer(vec);
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::Package { spec, blobs } => {
                let mut coreml_model =
                    CoreMLModel::load_package(spec.clone(), blobs.clone(), info.clone());
                coreml_model.model.load();
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadStatic(
                        "Failed to load in-memory model package; needs macOS 15+ and every weight blob",
                        Self::Unloaded(info, CoreMLModelLoader::Package { spec, blobs }),
                    ));
                }
                Ok(Self::Loaded(
                    coreml_model,
                    info,
                    CoreMLModelLoader::Package { spec, blobs },
                ))
            }
            CoreMLModelLoader::BufferToDisk(u) => {
                match std::fs::File::open(&u)
                    .map_err(CoreMLError::IoError)
                    .and_then(|file| {
                        let mut vec = vec![];
                        _ = flate2::read::ZlibDecoder::new(file)
                            .read_to_end(&mut vec)
                            .map_err(CoreMLError::IoError)?;
                        Ok(vec)
                    }) {
                    Ok(vec) => {
                        let mut coreml_model = CoreMLModel::load_buffer(vec, info.clone());
                        coreml_model.model.load();
                        let loader = CoreMLModelLoader::BufferToDisk(u);
                        Ok(Self::Loaded(coreml_model, info, loader))
                    }
                    Err(err) => Err(CoreMLError::FailedToLoad(
                        format!("failed to load the model from cached buffer path: {err}"),
                        CoreMLModelWithState::Unloaded(info, CoreMLModelLoader::BufferToDisk(u)),
                    )),
                }
            }
        }
    }

    /// Might fail irrecoverably if the system is too low on disk space(very unlikely)
    pub fn unload(self) -> Result<Self, CoreMLError> {
        if let Self::Loaded(model, info, loader) = self {
            Ok(Self::Unloaded(
                info,
                match loader {
                    CoreMLModelLoader::Buffer(v) => {
                        let mut temp_file = NamedTempFile::new().map_err(CoreMLError::IoError)?;
                        temp_file.write_all(&v).map_err(CoreMLError::IoError)?;
                        let res = std::fs::read(temp_file.path()).map_err(CoreMLError::IoError)?;
                        CoreMLModelLoader::Buffer(res)
                    }
                    CoreMLModelLoader::ModelPath(_) => {
                        // if the model is loaded from modelPath it has to have compiled path
                        let path = model.model.compiled_path().unwrap();
                        CoreMLModelLoader::CompiledPath(path.into())
                    }
                    x => x,
                },
            ))
        } else {
            Ok(self)
        }
    }

    /// Unloads the model buffer to the disk, at cache_dir
    pub fn unload_to_disk(self) -> Result<Self, CoreMLError> {
        match self {
            Self::Loaded(_, mut info, loader) | Self::Unloaded(mut info, loader) => {
                let loader = {
                    match loader {
                        CoreMLModelLoader::Buffer(vec) => {
                            if info.opts.cache_dir.as_os_str().is_empty() {
                                info.opts.cache_dir = PathBuf::from(".");
                            }
                            if !info.opts.cache_dir.exists() {
                                _ = std::fs::remove_dir_all(&info.opts.cache_dir);
                                _ = std::fs::create_dir_all(&info.opts.cache_dir);
                            }
                            // pick the file specified, if it's a folder/dir append model_cache
                            let m = if !info.opts.cache_dir.is_dir() {
                                info.opts.cache_dir.clone()
                            } else {
                                info.opts.cache_dir.join("model_cache")
                            };
                            match std::fs::File::create(&m).map_err(CoreMLError::IoError).map(
                                |file| {
                                    flate2::write::ZlibEncoder::new(file, Compression::best())
                                        .write_all(&vec)
                                        .map_err(CoreMLError::IoError)
                                },
                            ) {
                                Ok(_) => {}
                                Err(err) => {
                                    return Err(CoreMLError::FailedToLoad(
                                        format!("failed to load the model from the buffer: {err}"),
                                        CoreMLModelWithState::Unloaded(
                                            info,
                                            CoreMLModelLoader::Buffer(vec),
                                        ),
                                    ));
                                }
                            };
                            CoreMLModelLoader::BufferToDisk(m)
                        }
                        loader => loader,
                    }
                };
                Ok(Self::Unloaded(info, loader))
            }
        }
    }

    pub fn description(&self) -> Result<HashMap<&str, Vec<String>>, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => Ok(core_mlmodel.description()),
        }
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.add_input(tag, input),
        }
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict(),
        }
    }
}

// Info required to create a coreml model
#[derive(Debug, Clone)]
pub struct CoreMLModelInfo {
    pub opts: CoreMLModelOptions,
}

/// Everything the prediction path needs about one output. Read once from the
/// CoreML description instead of on every `predict`.
#[derive(Clone, Debug)]
pub(crate) struct OutputSpec {
    pub(crate) name: String,
    pub(crate) dtype: &'static str,
    pub(crate) shape: Vec<usize>,
}

#[derive(Debug)]
pub struct CoreMLModel {
    model: Model,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
    input_shapes: HashMap<String, Vec<usize>>,
    output_specs: Vec<OutputSpec>,
    described: bool,
}

unsafe impl Send for CoreMLModel {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

impl CoreMLModel {
    pub fn load_from_path(path: String, info: CoreMLModelInfo, compiled: bool) -> Self {
        Self {
            model: modelWithPath(path, info.opts.compute_platform, compiled),
            outputs: Default::default(),
            input_shapes: Default::default(),
            output_specs: Default::default(),
            described: false,
        }
    }

    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        // `modelWithAssets` reconstructs the `Vec` from the raw pointer, length
        // and capacity, so make sure the length *is* the capacity first.
        buf.shrink_to_fit();
        let coreml_model = Self {
            model: modelWithAssets(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            outputs: Default::default(),
            input_shapes: Default::default(),
            output_specs: Default::default(),
            described: false,
        };
        std::mem::forget(buf);
        coreml_model
    }

    pub fn load_package(
        spec: Vec<u8>,
        blobs: Vec<(String, Vec<u8>)>,
        info: CoreMLModelInfo,
    ) -> Self {
        Self {
            model: modelWithPackage(spec, blobs, info.opts.compute_platform),
            outputs: Default::default(),
            input_shapes: Default::default(),
            output_specs: Default::default(),
            described: false,
        }
    }

    /// Reads the feature metadata once, after the model is loaded, so that
    /// neither `add_input` nor `predict` has to walk CoreML's description
    /// dictionaries on the hot path.
    fn ensure_described(&mut self) {
        if self.described {
            return;
        }
        let description = self.model.description();
        self.input_shapes = description
            .input_names()
            .into_iter()
            .map(|name| {
                let shape = description.input_shape(name.clone());
                (name, shape)
            })
            .collect();
        self.output_specs = description
            .output_names()
            .into_iter()
            .map(|name| {
                let dtype = match description.output_type(name.clone()).as_str() {
                    "f32" => "f32",
                    "f16" => "f16",
                    _ => "",
                };
                let shape = description.output_shape(name.clone());
                OutputSpec { name, dtype, shape }
            })
            .collect();
        self.described = true;
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
    ) -> Result<(), CoreMLError> {
        // route input correctly
        let input: MLArray = input.into();
        let name = tag.as_ref().to_string();
        let shape: Vec<usize> = input.shape().to_vec();
        self.ensure_described();
        let arr = self.input_shapes.get(&name).cloned().unwrap_or_default();
        if arr.len() != shape.len() || !arr.iter().eq(shape.iter()) {
            if arr.is_empty() {
                return Err(CoreMLError::BadInputShape(format!(
                    "Input feature name '{name}' not expected!"
                )));
            }
            return Err(CoreMLError::BadInputShape(format!(
                "expected shape {arr:?} found {shape:?}"
            )));
        }
        match input {
            MLArray::Float32Array(array_base) => {
                let (mut data, offset) = array_base.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                if !self
                    .model
                    .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity())
                {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            MLArray::Float16Array(array_base) => {
                let (mut data, offset) = array_base.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                if !self.model.bindInputU16(
                    shape,
                    name,
                    data.as_mut_ptr() as *mut u16,
                    data.capacity(),
                ) {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            MLArray::Int32Array(array_base) => {
                let (mut data, offset) = array_base.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                if !self
                    .model
                    .bindInputI32(shape, name, data.as_mut_ptr(), data.capacity())
                {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            _ => {
                return Err(CoreMLError::UnknownErrorStatic(
                    "failed to bind input to model",
                ));
            } // MLArray::Int16Array(array_base) => todo!(),
              // MLArray::Int8Array(array_base) => todo!(),
              // MLArray::UInt32Array(array_base) => todo!(),
              // MLArray::UInt16Array(array_base) => todo!(),
              // MLArray::UInt8Array(array_base) => todo!(),
        }
        Ok(())
    }

    pub fn add_output_f32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        let arr: MLArray = out.into();
        let shape = arr.shape();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f32", shape.to_vec()));
        let shape: Vec<i32> = shape.iter().map(|i| *i as i32).collect();
        let (mut data, offset) = arr.extract_to_tensor::<f32>().into_raw_vec_and_offset();
        assert!(
            matches!(offset, Some(0) | None),
            "array base offset is not zero; bad aligned output buffer"
        );
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        if !self.model.bindOutputF32(shape, name, ptr, len) {
            return false;
        }
        std::mem::forget(data);
        true
    }

    /// Binds an f16 output buffer. Unlike `add_output_f32` the array is built
    /// here because `ndarray::zeros` needs `num-traits`, which `half::f16`
    /// only implements behind an off-by-default feature.
    pub fn add_output_f16(&mut self, tag: impl AsRef<str>, shape: Vec<usize>) -> bool {
        let count: usize = shape.iter().product();
        let Ok(array) = Array::from_shape_vec(shape.clone(), vec![half::f16::ZERO; count]) else {
            return false;
        };
        let shape_i32: Vec<i32> = shape.iter().map(|dim| *dim as i32).collect();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f16", shape));
        let (mut data, offset) = array.into_raw_vec_and_offset();
        assert!(
            matches!(offset, Some(0) | None),
            "array base offset is not zero; bad aligned output buffer"
        );
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr() as *mut u16;
        let len = data.capacity();
        if !self.model.bindOutputF16(shape_i32, name, ptr, len) {
            return false;
        }
        std::mem::forget(data);
        true
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        self.ensure_described();
        // Output backings are skipped once CoreML has rejected them for a
        // dynamically-shaped output; predictions are then copied out.
        let with_backings = self.model.backings_supported();
        for index in 0..self.output_specs.len() {
            let (name, dtype, shape) = {
                let spec = &self.output_specs[index];
                (spec.name.clone(), spec.dtype, spec.shape.clone())
            };
            match dtype {
                "f32" => {
                    if with_backings {
                        self.add_output_f32(name, Array::<f32, _>::zeros(shape));
                    }
                }
                "f16" => {
                    if with_backings {
                        self.add_output_f16(name, shape);
                    }
                }
                _ => {
                    self.outputs.clear();
                    return Err(CoreMLError::UnknownErrorStatic(
                        "non-f32/f16 output types are not supported (yet)!",
                    ));
                }
            }
        }
        let mut output = self.model.predict();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }
        let mut outputs = HashMap::with_capacity(self.outputs.len());
        for (key, (ty, shape)) in &self.outputs {
            match *ty {
                "f32" => {
                    let out = output.outputF32(key.clone());
                    if let Ok(array) = Array::from_shape_vec(shape.clone(), out) {
                        outputs.insert(key.clone(), array.into());
                    }
                }
                "f16" => {
                    let out = output.outputU16(key.clone());
                    if let Ok(array) = Array::from_shape_vec(shape.clone(), out) {
                        outputs.insert(key.clone(), reinterpret_u16_to_f16(array).into());
                    }
                }
                _ => {
                    eprintln!(
                        "warning: type not one of f32 or f16, and will be skipped in the output"
                    );
                }
            }
        }
        Ok(MLModelOutput { outputs })
    }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.description();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }
}

/// The specification and weight blobs of an in-memory `.mlpackage`.
pub(crate) type PackageParts = (Vec<u8>, Vec<(String, Vec<u8>)>);

/// Reads `Data/com.apple.CoreML/model.mlmodel` and every `@model_path/...`
/// weight blob it references, resolved against the package's
/// `Data/com.apple.CoreML` directory.
pub(crate) fn read_package(package: &Path) -> Result<PackageParts, CoreMLError> {
    let root = package.join("Data/com.apple.CoreML");
    let spec = std::fs::read(root.join("model.mlmodel")).map_err(CoreMLError::IoError)?;
    let mut blobs = Vec::new();
    for reference in model_path_references(&spec) {
        let data = std::fs::read(root.join(&reference)).map_err(CoreMLError::IoError)?;
        blobs.push((reference, data));
    }
    Ok((spec, blobs))
}

/// Collects the `@model_path/...` file references embedded in a CoreML
/// specification. The protobuf stores them as plain strings.
fn model_path_references(spec: &[u8]) -> Vec<String> {
    const PREFIX: &[u8] = b"@model_path/";
    let mut references = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = spec[search_from..]
        .windows(PREFIX.len())
        .position(|window| window == PREFIX)
    {
        let start = search_from + offset + PREFIX.len();
        let mut end = start;
        while end < spec.len() && is_reference_byte(spec[end]) {
            end += 1;
        }
        if end > start {
            references.push(String::from_utf8_lossy(&spec[start..end]).into_owned());
        }
        search_from = end.max(start + 1);
    }
    references.sort();
    references.dedup();
    references
}

fn is_reference_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-' | b'+' | b'@')
}

fn reinterpret_u16_to_f16(input: ndarray::ArrayD<u16>) -> ndarray::ArrayD<half::f16> {
    let shape = input.shape().to_vec();
    let len = input.len();

    // Consume input and get the raw Vec<u32>
    let (raw_vec, offset) = input.into_raw_vec_and_offset();
    assert!(
        matches!(offset, Some(0) | None),
        "array base offset is not zero; bad aligned data reinterpret"
    );

    // SAFETY:
    // - u32 and f32 have the same size
    // - The underlying data is valid to reinterpret as f32
    // - This creates a new Vec<f32> with the same bytes
    let raw_vec_f16 = {
        let ptr = raw_vec.as_ptr() as *mut half::f16;
        let capacity = raw_vec.capacity();
        std::mem::forget(raw_vec); // prevent drop of original vec
        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    };

    // TODO SA: we know unwrap won't cause ShapeError, but avoid unwrap regardless
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), raw_vec_f16).unwrap()
}

#[cfg(test)]
mod tests {
    use super::model_path_references;

    #[test]
    fn collects_unique_model_path_references() {
        let mut spec = Vec::new();
        // Two BlobFileValue entries for the same file plus unrelated bytes.
        spec.extend_from_slice(b"\x0a\x1e@model_path/weights/weight.bin\x10\x40");
        spec.extend_from_slice(b"\x0a\x1e@model_path/weights/weight.bin\x10\x80\x08");
        spec.extend_from_slice(b"not/a/reference\x00");
        assert_eq!(
            model_path_references(&spec),
            vec!["weights/weight.bin".to_string()]
        );
    }

    #[test]
    fn handles_specs_without_references() {
        assert!(model_path_references(b"\x08\x06").is_empty());
    }
}
