use crate::{
    ffi::MLBatchModelOutput,
    ffi::{modelWithPathBatch, BatchModel, LoadError},
    mlarray::MLArray,
    mlmodel::{CoreMLError, CoreMLModelInfo, CoreMLModelLoader, TensorInfo},
    CoreMLModelOptions,
};
use std::path::Path;

pub use crate::ffi::MLModelOutput;

// See `CoreMLModelWithState`; boxing `Loaded` would change the public shape.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum CoreMLBatchModelWithState {
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    Loaded(CoreMLBatchModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl CoreMLBatchModelWithState {
    /// Creates an unloaded batch model from a caller-owned `.mlmodelc` directory.
    pub fn new_compiled(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader {
                path: path.as_ref().to_path_buf(),
            },
        )
    }

    pub fn load(self) -> Result<Self, CoreMLError> {
        let Self::Unloaded(info, loader) = self else {
            return Ok(self);
        };
        let mut model = CoreMLBatchModel::load_compiled(&loader.path, &info);
        match model.model.load() {
            Ok(()) => Ok(Self::Loaded(model, info, loader)),
            Err(LoadError::Native {
                domain,
                code,
                message,
            }) => Err(CoreMLError::NativeBatch {
                domain,
                code,
                message,
                model: Self::Unloaded(info, loader),
            }),
            Err(error) => Err(CoreMLError::FailedToBatchLoad(
                error.to_string(),
                Self::Unloaded(info, loader),
            )),
        }
    }

    /// Releases the loaded model without modifying its caller-owned bundle.
    pub fn unload(self) -> Result<Self, CoreMLError> {
        match self {
            Self::Loaded(_, info, loader) => Ok(Self::Unloaded(info, loader)),
            unloaded => Ok(unloaded),
        }
    }

    pub fn inputs_info(&mut self) -> Result<&[TensorInfo], CoreMLError> {
        match self {
            Self::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            Self::Loaded(model, _, _) => model.inputs_info(),
        }
    }

    pub fn outputs_info(&mut self) -> Result<&[TensorInfo], CoreMLError> {
        match self {
            Self::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            Self::Loaded(model, _, _) => model.outputs_info(),
        }
    }

    pub fn clear_request(&mut self) {
        if let Self::Loaded(model, _, _) = self {
            model.clear_request();
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
}

unsafe impl Send for CoreMLBatchModel {}

impl std::fmt::Debug for BatchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchModel").finish()
    }
}

impl CoreMLBatchModel {
    pub fn load_compiled(path: impl AsRef<Path>, info: &CoreMLModelInfo) -> Self {
        Self {
            model: modelWithPathBatch(path, info.opts.compute_platform),
        }
    }

    pub fn inputs_info(&mut self) -> Result<&[TensorInfo], CoreMLError> {
        self.model.inputs_info().map_err(CoreMLError::InvalidInput)
    }

    pub fn outputs_info(&mut self) -> Result<&[TensorInfo], CoreMLError> {
        self.model
            .outputs_info()
            .map_err(CoreMLError::InvalidOutput)
    }

    pub fn clear_request(&mut self) {
        self.model.clear_request();
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
        idx: isize,
    ) -> Result<(), CoreMLError> {
        self.model
            .bind_input(tag.as_ref(), input.into(), idx)
            .map_err(CoreMLError::InvalidInput)
    }

    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        let output = self.model.predict();
        if let Some(error) = output.getError() {
            return Err(CoreMLError::InvalidOutput(error));
        }
        let specs = self
            .model
            .outputs_info()
            .map_err(CoreMLError::InvalidOutput)?;
        let count = output.count();
        if count < 0 {
            return Err(CoreMLError::InvalidOutput(
                "negative batch output count".into(),
            ));
        }
        let mut outputs = Vec::with_capacity(count as usize);
        for index in 0..count {
            outputs.push(
                output
                    .for_idx(index)
                    .into_outputs(specs)
                    .map_err(CoreMLError::InvalidOutput)?,
            );
        }
        Ok(MLBatchModelOutput { outputs })
    }
}
