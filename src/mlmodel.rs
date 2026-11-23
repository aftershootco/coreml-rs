use crate::{
    ffi::{modelWithPath, ComputePlatform, LoadError, Model},
    mlarray::{MLArray, MLDataType},
    mlbatchmodel::CoreMLBatchModelWithState,
};
use std::path::{Path, PathBuf};

pub use crate::ffi::MLModelOutput;

use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreMLError {
    #[error("CoreML IoError: {0}")]
    IoError(std::io::Error),
    #[error("BadInputShape: {0}")]
    BadInputShape(String),
    #[error("InvalidInput: {0}")]
    InvalidInput(String),
    #[error("InvalidOutput: {0}")]
    InvalidOutput(String),
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
    #[error("CoreML native error {domain} ({code}): {message}")]
    Native {
        domain: String,
        code: isize,
        message: String,
        model: CoreMLModelWithState,
    },
    #[error("CoreML native batch error {domain} ({code}): {message}")]
    NativeBatch {
        domain: String,
        code: isize,
        message: String,
        model: CoreMLBatchModelWithState,
    },
}

#[derive(Default, Clone)]
pub struct CoreMLModelOptions {
    pub compute_platform: ComputePlatform,
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
pub struct CoreMLModelLoader {
    /// Caller-owned compiled `.mlmodelc` directory, retained unchanged on reload.
    pub path: PathBuf,
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
    /// Creates an unloaded model from a caller-owned compiled bundle.
    ///
    /// `load` requires an existing `.mlmodelc` directory. CoreML validates the
    /// contents and performs any upstream encryption/key authorization flow.
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
        let mut model = CoreMLModel::load_compiled(&loader.path, &info);
        match model.model.load() {
            Ok(()) => Ok(Self::Loaded(model, info, loader)),
            Err(LoadError::Native {
                domain,
                code,
                message,
            }) => Err(CoreMLError::Native {
                domain,
                code,
                message,
                model: Self::Unloaded(info, loader),
            }),
            Err(error) => Err(CoreMLError::FailedToLoad(
                error.to_string(),
                Self::Unloaded(info, loader),
            )),
        }
    }

    /// Releases the loaded model without changing or taking ownership of its path.
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

    /// Discards every input and output backing belonging to the current request.
    pub fn clear_request(&mut self) {
        if let Self::Loaded(model, _, _) = self {
            model.clear_request();
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

/// The native model's accepted shapes, not merely its default shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeConstraint {
    /// Native CoreML declares no rank or dimension restriction.
    Unconstrained,
    Fixed(Vec<usize>),
    Enumerated(Vec<Vec<usize>>),
    /// Inclusive lower/upper bounds; `None` is an unbounded upper bound.
    Ranges(Vec<(usize, Option<usize>)>),
}

impl ShapeConstraint {
    pub fn accepts(&self, shape: &[usize]) -> bool {
        match self {
            Self::Unconstrained => true,
            Self::Fixed(expected) => expected == shape,
            Self::Enumerated(shapes) => shapes.iter().any(|expected| expected == shape),
            Self::Ranges(ranges) => {
                ranges.len() == shape.len()
                    && ranges
                        .iter()
                        .zip(shape)
                        .all(|(&(min, max), &dim)| dim >= min && max.map_or(true, |max| dim <= max))
            }
        }
    }
}

impl std::fmt::Display for ShapeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unconstrained => f.write_str("unconstrained"),
            Self::Fixed(shape) => write!(f, "fixed({shape:?})"),
            Self::Enumerated(shapes) => write!(f, "enumerated({shapes:?})"),
            Self::Ranges(ranges) => {
                f.write_str("ranges([")?;
                for (index, (min, max)) in ranges.iter().enumerate() {
                    if index != 0 {
                        f.write_str(", ")?;
                    }
                    match max {
                        Some(max) => write!(f, "{min}..={max}")?,
                        None => write!(f, "{min}..")?,
                    }
                }
                f.write_str("])")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// The model's default shape. Predictions use their actual returned shape.
    pub shape: Vec<usize>,
    pub dtype: MLDataType,
    pub constraint: ShapeConstraint,
    pub optional: bool,
}

impl TensorInfo {
    pub fn accepts_shape(&self, shape: &[usize]) -> bool {
        self.constraint.accepts(shape)
    }
}

impl std::fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: dtype={}, default_shape={:?}, constraint={}, optional={}",
            self.name, self.dtype, self.shape, self.constraint, self.optional
        )
    }
}

#[derive(Debug)]
pub struct CoreMLModel {
    model: Model,
}

unsafe impl Send for CoreMLModel {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

impl CoreMLModel {
    pub fn load_compiled(path: impl AsRef<Path>, info: &CoreMLModelInfo) -> Self {
        Self {
            model: modelWithPath(path, info.opts.compute_platform),
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
    ) -> Result<(), CoreMLError> {
        self.model
            .bind_input(tag.as_ref(), input.into())
            .map_err(CoreMLError::InvalidInput)
    }

    pub fn add_output_f32(
        &mut self,
        tag: impl AsRef<str>,
        out: impl Into<MLArray>,
    ) -> Result<(), CoreMLError> {
        let result = out
            .into()
            .extract_to_tensor::<f32>()
            .map_err(|error| CoreMLError::InvalidOutput(error.to_string()))
            .and_then(|array| {
                self.model
                    .bind_output(tag.as_ref(), array.into())
                    .map_err(CoreMLError::InvalidOutput)
            });
        if result.is_err() {
            self.clear_request();
        }
        result
    }

    pub fn add_output_f16(
        &mut self,
        tag: impl AsRef<str>,
        shape: Vec<usize>,
    ) -> Result<(), CoreMLError> {
        let result = crate::ffi::zero_array(MLDataType::F16, &shape)
            .and_then(|array| self.model.bind_output(tag.as_ref(), array))
            .map_err(CoreMLError::InvalidOutput);
        if result.is_err() {
            self.clear_request();
        }
        result
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        let output = self.model.predict();
        let specs = self
            .model
            .outputs_info()
            .map_err(CoreMLError::InvalidOutput)?;
        let outputs = output
            .into_outputs(specs)
            .map_err(CoreMLError::InvalidOutput)?;
        Ok(MLModelOutput { outputs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_constraints_check_rank_and_inclusive_bounds() {
        assert!(ShapeConstraint::Unconstrained.accepts(&[2, 3]));
        assert!(ShapeConstraint::Unconstrained.accepts(&[7]));
        let fixed = ShapeConstraint::Fixed(vec![2, 3]);
        assert!(fixed.accepts(&[2, 3]));
        assert!(!fixed.accepts(&[6]));
        let enumerated = ShapeConstraint::Enumerated(vec![vec![1, 3], vec![4, 3]]);
        assert!(enumerated.accepts(&[4, 3]));
        assert!(!enumerated.accepts(&[2, 3]));
        let ranges = ShapeConstraint::Ranges(vec![(1, Some(4)), (3, None)]);
        assert!(ranges.accepts(&[1, 3]));
        assert!(ranges.accepts(&[4, 100]));
        assert!(!ranges.accepts(&[5, 3]));
        assert!(!ranges.accepts(&[1, 2]));
        assert!(!ranges.accepts(&[1, 3, 1]));
    }

    #[test]
    fn unloaded_metadata_errors_and_request_clear_is_a_noop() {
        let mut model =
            CoreMLModelWithState::new_compiled("unused.mlmodelc", CoreMLModelOptions::default());
        model.clear_request();
        assert!(matches!(
            model.inputs_info(),
            Err(CoreMLError::ModelNotLoaded)
        ));
        assert!(matches!(
            model.outputs_info(),
            Err(CoreMLError::ModelNotLoaded)
        ));
        let mut batch = CoreMLBatchModelWithState::new_compiled(
            "unused.mlmodelc",
            CoreMLModelOptions::default(),
        );
        batch.clear_request();
        assert!(matches!(
            batch.inputs_info(),
            Err(CoreMLError::ModelNotLoaded)
        ));
        assert!(matches!(
            batch.outputs_info(),
            Err(CoreMLError::ModelNotLoaded)
        ));
    }
}
