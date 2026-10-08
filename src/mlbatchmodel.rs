use crate::{
    ffi::MLBatchModelOutput,
    ffi::{modelWithAssetsBatch, modelWithPathBatch, BatchModel},
    mlarray::MLArray,
    mlmodel::{CoreMLError, CoreMLModelInfo, CoreMLModelLoader, OutputSpec},
    CoreMLModelOptions,
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

// See `CoreMLModelWithState`; boxing `Loaded` would change the public shape.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum CoreMLBatchModelWithState {
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    Loaded(CoreMLBatchModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl CoreMLBatchModelWithState {
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

    pub fn load(self) -> Result<Self, CoreMLError> {
        let Self::Unloaded(info, loader) = self else {
            return Ok(self);
        };
        match loader {
            CoreMLModelLoader::ModelPath(path_buf) => {
                let mut coreml_model = CoreMLBatchModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    false,
                );
                coreml_model.model.load();
                let loader = CoreMLModelLoader::ModelPath(path_buf);
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML model file",
                        Self::Unloaded(info, loader),
                    ));
                }
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::CompiledPath(path_buf) => {
                let mut coreml_model = CoreMLBatchModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    true,
                );
                coreml_model.model.load();
                let loader = CoreMLModelLoader::CompiledPath(path_buf);
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML model file",
                        Self::Unloaded(info, loader),
                    ));
                }
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::Buffer(vec) => {
                let mut coreml_model = CoreMLBatchModel::load_buffer(vec.clone(), info.clone());
                coreml_model.model.load();
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML mlmodel file",
                        Self::Unloaded(info, CoreMLModelLoader::Buffer(vec)),
                    ));
                }
                let loader = CoreMLModelLoader::Buffer(vec);
                Ok(Self::Loaded(coreml_model, info, loader))
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
                        let mut coreml_model = CoreMLBatchModel::load_buffer(vec, info.clone());
                        coreml_model.model.load();
                        let loader = CoreMLModelLoader::BufferToDisk(u);
                        Ok(Self::Loaded(coreml_model, info, loader))
                    }
                    Err(err) => Err(CoreMLError::FailedToBatchLoad(
                        format!("failed to load the model from cached buffer path: {err}"),
                        CoreMLBatchModelWithState::Unloaded(
                            info,
                            CoreMLModelLoader::BufferToDisk(u),
                        ),
                    )),
                }
            }
        }
    }

    /// Might fail if system disk space too low(very unlikely)
    pub fn unload(self) -> Result<Self, CoreMLError> {
        if let Self::Loaded(model, info, loader) = self {
            Ok(Self::Unloaded(
                info,
                match loader {
                    CoreMLModelLoader::Buffer(v) => {
                        let mut temp_file = NamedTempFile::new().map_err(CoreMLError::IoError)?;
                        temp_file.write_all(&v).map_err(CoreMLError::IoError)?;
                        CoreMLModelLoader::Buffer(
                            std::fs::read(temp_file.path()).map_err(CoreMLError::IoError)?,
                        )
                    }
                    // Mirror the single-model path: after a path load the model
                    // has a compiled copy, so reloading can skip compilation.
                    CoreMLModelLoader::ModelPath(path) => match model.model.compiled_path() {
                        Some(compiled) => CoreMLModelLoader::CompiledPath(compiled.into()),
                        None => CoreMLModelLoader::ModelPath(path),
                    },
                    loader => loader,
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
                                    return Err(CoreMLError::FailedToBatchLoad(
                                        format!("failed to load the model from the buffer: {err}"),
                                        CoreMLBatchModelWithState::Unloaded(
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
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => Ok(core_mlmodel.description()),
        }
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
        idx: isize,
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => {
                core_mlmodel.add_input(tag, input, idx)
            }
        }
    }

    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        match self {
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict(),
        }
    }
}

#[derive(Debug)]
pub struct CoreMLBatchModel {
    model: BatchModel,
    // save_path: Option<PathBuf>,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
    input_shapes: HashMap<String, Vec<usize>>,
    output_specs: Vec<OutputSpec>,
    described: bool,
}

unsafe impl Send for CoreMLBatchModel {}

impl std::fmt::Debug for BatchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchModel").finish()
    }
}

impl CoreMLBatchModel {
    pub fn load_from_path(path: String, info: CoreMLModelInfo, compiled: bool) -> Self {
        Self {
            model: modelWithPathBatch(path, info.opts.compute_platform, compiled),
            // save_path: None,
            outputs: Default::default(),
            input_shapes: Default::default(),
            output_specs: Default::default(),
            described: false,
        }
    }

    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        // `modelWithAssetsBatch` reconstructs the `Vec` from pointer, length and
        // capacity, so ensure the length is the capacity first.
        buf.shrink_to_fit();
        let coreml_model = Self {
            model: modelWithAssetsBatch(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            // save_path: None,
            outputs: Default::default(),
            input_shapes: Default::default(),
            output_specs: Default::default(),
            described: false,
        };
        std::mem::forget(buf);
        coreml_model
    }

    /// Reads the feature metadata once, after the model is loaded, so the
    /// prediction path stays off CoreML's description dictionaries.
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
        idx: isize,
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
                    .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity(), idx)
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
            }
        }
        Ok(())
    }

    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        self.ensure_described();
        for spec in &self.output_specs {
            match spec.dtype {
                "f32" => {
                    self.outputs
                        .insert(spec.name.clone(), ("f32", spec.shape.clone()));
                }
                _ => {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "non-f32 output types are not supported (yet)!",
                    ))
                }
            }
        }

        let output = self.model.predict();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }
        let mut outputs = Vec::with_capacity(output.count().max(0) as usize);
        for index in 0..output.count() {
            let mut prediction = output.for_idx(index);
            let mut values = HashMap::with_capacity(self.outputs.len());
            for (key, (ty, shape)) in &self.outputs {
                if *ty != "f32" {
                    eprintln!(
                        "warning: non-f32 types aren't supported, and will be skipped in the output"
                    );
                    continue;
                }
                let out = prediction.outputF32(key.clone());
                if let Ok(array) = Array::from_shape_vec(shape.clone(), out) {
                    values.insert(key.clone(), array.into());
                }
            }
            outputs.push(values);
        }
        Ok(MLBatchModelOutput { outputs })
    }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.description();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }
}
