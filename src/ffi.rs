//! CoreML inference layer implemented directly on top of Apple's Objective-C
//! CoreML framework through `objc2`.
//!
//! Models are loaded only from caller-owned compiled `.mlmodelc` directories.
//! The asynchronous URL loader supports Core ML's upstream encryption flow;
//! this crate never compiles, decrypts, rewrites, or removes model bundles.

#![allow(non_snake_case)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};

use block2::{DynBlock, RcBlock};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLArrayBatchProvider, MLBatchProvider, MLComputeUnits, MLDictionaryFeatureProvider,
    MLFeatureDescription, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration,
    MLModelDescription, MLMultiArray, MLMultiArrayDataType, MLMultiArrayShapeConstraintType,
    MLPredictionOptions,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

use crate::{
    mlarray::{MLArray, MLDataType},
    mlmodel::{ShapeConstraint, TensorInfo},
};

/// A load failure before it is paired with the public model's retry state.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("invalid compiled CoreML bundle: {0}")]
    InvalidPath(String),
    #[error("unsupported CoreML metadata: {0}")]
    Metadata(String),
    #[error("CoreML native error {domain} ({code}): {message}")]
    Native {
        domain: String,
        code: isize,
        message: String,
    },
    #[error("CoreML completed loading without a model or an NSError")]
    MissingModel,
}

/// The compute units a CoreML model is allowed to run on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputePlatform {
    Cpu,
    CpuAndANE,
    /// Matches the historical default: CPU + GPU, leaving the ANE out of the
    /// default path because it is opt-in on Apple silicon.
    #[default]
    CpuAndGpu,
}

impl ComputePlatform {
    fn compute_units(self) -> MLComputeUnits {
        match self {
            ComputePlatform::Cpu => MLComputeUnits::CPUOnly,
            ComputePlatform::CpuAndANE => MLComputeUnits::CPUAndNeuralEngine,
            ComputePlatform::CpuAndGpu => MLComputeUnits::CPUAndGPU,
        }
    }
}

/// Rust remains the sole owner of output allocations. Native arrays borrow them
/// only until prediction completes; no raw pointer transfers ownership.
enum BackingBuffer {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    I32(Vec<i32>),
}

impl BackingBuffer {
    fn dtype(&self) -> MLDataType {
        match self {
            Self::F32(_) => MLDataType::F32,
            Self::F16(_) => MLDataType::F16,
            Self::I32(_) => MLDataType::I32,
        }
    }

    fn pointer(&mut self) -> *mut std::ffi::c_void {
        match self {
            Self::F32(data) => data.as_mut_ptr().cast(),
            Self::F16(data) => data.as_mut_ptr().cast(),
            Self::I32(data) => data.as_mut_ptr().cast(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::F16(data) => data.len(),
            Self::I32(data) => data.len(),
        }
    }

    fn into_array(self, shape: &[usize]) -> Result<MLArray, String> {
        match self {
            Self::F32(data) => array_from_vec(shape, data),
            Self::F16(data) => array_from_vec(shape, data),
            Self::I32(data) => array_from_vec(shape, data),
        }
    }
}

struct Backing {
    // Drop native references before the allocation they borrow.
    array: Retained<MLMultiArray>,
    buffer: BackingBuffer,
    shape: Vec<usize>,
    used: bool,
}

type Backings = HashMap<String, Backing>;

/// A prediction retains its actual provider even when output backings were used.
/// The provider must be validated before any backing is handed back to Rust.
pub struct ModelOutput {
    provider: Option<Retained<ProtocolObject<dyn MLFeatureProvider>>>,
    backings: Backings,
    error: Option<String>,
}

impl ModelOutput {
    fn failed(message: impl Into<String>) -> Self {
        Self {
            provider: None,
            backings: HashMap::new(),
            error: Some(message.into()),
        }
    }

    fn from_provider(provider: Retained<ProtocolObject<dyn MLFeatureProvider>>) -> Self {
        Self {
            provider: Some(provider),
            backings: HashMap::new(),
            error: None,
        }
    }

    pub fn getError(&self) -> Option<String> {
        self.error.clone()
    }

    /// Extracts every declared tensor, checking actual names, dtype, shape, and
    /// storage. Dynamic outputs use their returned shape, never the default.
    pub fn into_outputs(
        mut self,
        specs: &[TensorInfo],
    ) -> Result<HashMap<String, MLArray>, String> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let provider = self.provider.as_ref().ok_or("missing output provider")?;
        let names = unsafe { provider.featureNames() }.allObjects();
        for index in 0..names.len() {
            let name = names.objectAtIndex(index).to_string();
            if !specs.iter().any(|spec| spec.name == name) {
                return Err(format!("unexpected output feature `{name}`"));
            }
        }
        let mut outputs = HashMap::with_capacity(specs.len());
        for spec in specs {
            let Some(array) = multi_array_value(provider, &spec.name)? else {
                if spec.optional {
                    continue;
                }
                return Err(format!("missing output feature `{}`", spec.name));
            };
            let dtype = native_dtype(unsafe { array.dataType() })
                .map_err(|error| format!("output `{}`: {error}", spec.name))?;
            if dtype != spec.dtype {
                return Err(format!(
                    "output `{}` has dtype {dtype}, expected {}",
                    spec.name, spec.dtype
                ));
            }
            let native_shape = unsafe { array.shape() };
            if let Some(backing) = self.backings.get_mut(&spec.name) {
                if dtype == backing.buffer.dtype()
                    && matches_native_shape(&native_shape, &backing.shape)
                    && spec.accepts_shape(&backing.shape)
                    && usize::try_from(unsafe { array.count() }).ok() == Some(backing.buffer.len())
                    && verify_backing(&array, &mut backing.buffer, &backing.shape)
                        .map_err(|error| format!("output `{}`: {error}", spec.name))?
                {
                    backing.used = true;
                    continue;
                }
            }
            // CoreML may legally ignore a proposed backing. Copy its actual
            // result instead of returning the unused buffer or rejecting success.
            let shape = array_to_dimensions(&native_shape)
                .map_err(|error| format!("output `{}`: {error}", spec.name))?;
            if !spec.accepts_shape(&shape) {
                return Err(format!(
                    "output `{}` shape {shape:?} violates {:?}",
                    spec.name, spec.constraint
                ));
            }
            let count = checked_count(&shape)?;
            if usize::try_from(unsafe { array.count() }).ok() != Some(count) {
                return Err(format!(
                    "output `{}` has an inconsistent element count",
                    spec.name
                ));
            }
            let value = match dtype {
                MLDataType::F32 => array_from_vec(&shape, copy_bytes::<f32>(&array, &shape)?)?,
                MLDataType::F16 => {
                    array_from_vec(&shape, copy_bytes::<half::f16>(&array, &shape)?)?
                }
                MLDataType::I32 => array_from_vec(&shape, copy_bytes::<i32>(&array, &shape)?)?,
                _ => return Err(format!("unsupported output dtype {dtype}")),
            };
            outputs.insert(spec.name.clone(), value);
        }
        // Drop every native reference before moving the borrowed allocations.
        drop(self.provider.take());
        for (name, backing) in std::mem::take(&mut self.backings) {
            drop(backing.array);
            if backing.used {
                outputs.insert(name, backing.buffer.into_array(&backing.shape)?);
            }
        }
        Ok(outputs)
    }
}

/// The results of a batch prediction.
pub struct BatchOutput {
    provider: Option<Retained<ProtocolObject<dyn MLBatchProvider>>>,
    error: Option<String>,
}

impl BatchOutput {
    fn failed(message: impl Into<String>) -> Self {
        Self {
            provider: None,
            error: Some(message.into()),
        }
    }

    pub fn getError(&self) -> Option<String> {
        self.error.clone()
    }

    pub fn count(&self) -> isize {
        self.provider
            .as_ref()
            .map_or(0, |provider| unsafe { provider.count() })
    }

    pub fn for_idx(&self, at: isize) -> ModelOutput {
        match &self.provider {
            Some(provider) if at >= 0 && at < unsafe { provider.count() } => {
                ModelOutput::from_provider(unsafe { provider.featuresAtIndex(at) })
            }
            _ => ModelOutput::failed(
                self.error
                    .clone()
                    .unwrap_or_else(|| format!("invalid batch output index {at}")),
            ),
        }
    }
}

/// Internal adapter for reading typed metadata during model loading.
struct ModelDescription {
    description: Option<Retained<MLModelDescription>>,
}

impl ModelDescription {
    fn new(description: Option<Retained<MLModelDescription>>) -> Self {
        Self { description }
    }

    fn inputs_by_name(&self) -> Option<Retained<NSDictionary<NSString, MLFeatureDescription>>> {
        self.description
            .as_ref()
            .map(|description| unsafe { description.inputDescriptionsByName() })
    }

    fn outputs_by_name(&self) -> Option<Retained<NSDictionary<NSString, MLFeatureDescription>>> {
        self.description
            .as_ref()
            .map(|description| unsafe { description.outputDescriptionsByName() })
    }

    fn inputs_info(&self) -> Result<Vec<TensorInfo>, String> {
        let descriptions = self.inputs_by_name().ok_or("model is not loaded")?;
        tensor_infos(&descriptions)
    }

    fn outputs_info(&self) -> Result<Vec<TensorInfo>, String> {
        let descriptions = self.outputs_by_name().ok_or("model is not loaded")?;
        tensor_infos(&descriptions)
    }
}

/// A loaded (or loadable) CoreML model with request-local tensor bindings.
pub struct Model {
    compiled_path: PathBuf,
    model: Option<Retained<MLModel>>,
    inputs: HashMap<String, Retained<MLFeatureValue>>,
    outputs: Backings,
    input_specs: Vec<TensorInfo>,
    output_specs: Vec<TensorInfo>,
    compute_units: MLComputeUnits,
    backings_unsupported: bool,
}

// SAFETY: `MLModel` is documented as safe for concurrent predictions, and every
// mutable field is only touched through `&mut self` (enforced by the wrappers in
// `mlmodel`).
unsafe impl Send for Model {}

impl Model {
    fn from_path(path: PathBuf, compute: ComputePlatform) -> Self {
        Self {
            compiled_path: path,
            model: None,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            compute_units: compute.compute_units(),
            backings_unsupported: false,
        }
    }

    pub fn load(&mut self) -> Result<(), LoadError> {
        if self.model.is_none() {
            let model = load_compiled_model(&self.compiled_path, self.compute_units)?;
            let description = ModelDescription::new(Some(unsafe { model.modelDescription() }));
            let inputs = description.inputs_info().map_err(LoadError::Metadata)?;
            let outputs = description.outputs_info().map_err(LoadError::Metadata)?;
            self.input_specs = inputs;
            self.output_specs = outputs;
            self.backings_unsupported = false;
            self.model = Some(model);
        }
        Ok(())
    }

    pub fn clear_request(&mut self) {
        self.outputs.clear();
        self.inputs.clear();
    }

    pub fn unload(&mut self) {
        self.clear_request();
        self.model = None;
        self.input_specs.clear();
        self.output_specs.clear();
        self.backings_unsupported = false;
    }

    pub fn inputs_info(&self) -> Result<&[TensorInfo], String> {
        self.model.as_ref().ok_or("model is not loaded")?;
        Ok(&self.input_specs)
    }

    pub fn outputs_info(&self) -> Result<&[TensorInfo], String> {
        self.model.as_ref().ok_or("model is not loaded")?;
        Ok(&self.output_specs)
    }

    /// Owns initialized Vec storage inside a retained deallocator block.
    pub fn bind_input(&mut self, name: &str, input: MLArray) -> Result<(), String> {
        let result = (|| {
            validate_tensor(self.inputs_info()?, name, &input)?;
            let value = input_value(input)?;
            self.inputs.insert(name.to_owned(), value);
            Ok(())
        })();
        if result.is_err() {
            self.clear_request();
        }
        result
    }

    pub fn bind_output(&mut self, name: &str, output: MLArray) -> Result<(), String> {
        let result = (|| {
            validate_tensor(self.outputs_info()?, name, &output)?;
            self.outputs
                .insert(name.to_owned(), output_backing(output)?);
            Ok(())
        })();
        if result.is_err() {
            self.clear_request();
        }
        result
    }

    pub fn predict(&mut self) -> ModelOutput {
        // Taking both maps before validation makes every exit terminal for the
        // request, including provider creation and native prediction failures.
        let inputs = std::mem::take(&mut self.inputs);
        let outputs = std::mem::take(&mut self.outputs);
        with_pool(|| self.predict_inner(inputs, outputs)).unwrap_or_else(ModelOutput::failed)
    }

    fn predict_inner(
        &mut self,
        inputs: HashMap<String, Retained<MLFeatureValue>>,
        mut outputs: Backings,
    ) -> Result<ModelOutput, String> {
        let model = self.model.as_ref().ok_or("model is not loaded")?;
        validate_required(&self.input_specs, &inputs)?;
        if !self.backings_unsupported {
            for spec in &self.output_specs {
                if let Some(shape) = fixed_output_shape(spec) {
                    if !outputs.contains_key(&spec.name) {
                        outputs.insert(spec.name.clone(), zero_backing(spec.dtype, shape)?);
                    }
                }
            }
        } else {
            outputs.clear();
        }
        let dictionary = feature_dictionary(&inputs);
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dictionary,
            )
        }
        .map_err(|error| error.localizedDescription().to_string())?;
        if !outputs.is_empty() {
            let options = unsafe { MLPredictionOptions::new() };
            let backings = backing_dictionary(&outputs);
            unsafe { options.setOutputBackings(&backings) };
            match unsafe {
                model.predictionFromFeatures_options_error(
                    ProtocolObject::from_ref(&*provider),
                    &options,
                )
            } {
                Ok(actual) => {
                    return Ok(ModelOutput {
                        provider: Some(actual),
                        backings: outputs,
                        error: None,
                    });
                }
                Err(error)
                    if error
                        .localizedDescription()
                        .to_string()
                        .to_lowercase()
                        .contains("output backings") =>
                {
                    outputs.clear();
                    self.backings_unsupported = true;
                }
                Err(error) => return Err(error.localizedDescription().to_string()),
            }
        }
        let options = unsafe { MLPredictionOptions::new() };
        let actual = unsafe {
            model.predictionFromFeatures_options_error(
                ProtocolObject::from_ref(&*provider),
                &options,
            )
        }
        .map_err(|error| error.localizedDescription().to_string())?;
        Ok(ModelOutput::from_provider(actual))
    }
}

/// A batch (multi-instance) variant of [`Model`].
pub struct BatchModel {
    compiled_path: PathBuf,
    model: Option<Retained<MLModel>>,
    inputs: BTreeMap<usize, HashMap<String, Retained<MLFeatureValue>>>,
    input_specs: Vec<TensorInfo>,
    output_specs: Vec<TensorInfo>,
    compute_units: MLComputeUnits,
}

// SAFETY: see `Model`; batch inputs are only mutated through `&mut self`.
unsafe impl Send for BatchModel {}

impl BatchModel {
    fn from_path(path: PathBuf, compute: ComputePlatform) -> Self {
        Self {
            compiled_path: path,
            model: None,
            inputs: BTreeMap::new(),
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            compute_units: compute.compute_units(),
        }
    }

    pub fn load(&mut self) -> Result<(), LoadError> {
        if self.model.is_none() {
            let model = load_compiled_model(&self.compiled_path, self.compute_units)?;
            let description = ModelDescription::new(Some(unsafe { model.modelDescription() }));
            let inputs = description.inputs_info().map_err(LoadError::Metadata)?;
            let outputs = description.outputs_info().map_err(LoadError::Metadata)?;
            self.input_specs = inputs;
            self.output_specs = outputs;
            self.model = Some(model);
        }
        Ok(())
    }

    pub fn clear_request(&mut self) {
        self.inputs.clear();
    }

    pub fn unload(&mut self) {
        self.clear_request();
        self.model = None;
        self.input_specs.clear();
        self.output_specs.clear();
    }

    pub fn inputs_info(&self) -> Result<&[TensorInfo], String> {
        self.model.as_ref().ok_or("model is not loaded")?;
        Ok(&self.input_specs)
    }

    pub fn outputs_info(&self) -> Result<&[TensorInfo], String> {
        self.model.as_ref().ok_or("model is not loaded")?;
        Ok(&self.output_specs)
    }

    pub fn bind_input(&mut self, name: &str, input: MLArray, idx: isize) -> Result<(), String> {
        let result = (|| {
            let idx = usize::try_from(idx).map_err(|_| "negative batch input index")?;
            validate_tensor(self.inputs_info()?, name, &input)?;
            let value = input_value(input)?;
            self.inputs
                .entry(idx)
                .or_default()
                .insert(name.to_owned(), value);
            Ok(())
        })();
        if result.is_err() {
            self.clear_request();
        }
        result
    }

    pub fn predict(&mut self) -> BatchOutput {
        let inputs = std::mem::take(&mut self.inputs);
        with_pool(|| self.predict_inner(inputs)).unwrap_or_else(BatchOutput::failed)
    }

    fn predict_inner(
        &self,
        inputs: BTreeMap<usize, HashMap<String, Retained<MLFeatureValue>>>,
    ) -> Result<BatchOutput, String> {
        let model = self.model.as_ref().ok_or("model is not loaded")?;
        if inputs.is_empty() {
            return Err("batch request has no inputs".into());
        }
        let mut providers = Vec::with_capacity(inputs.len());
        for (expected_index, (index, values)) in inputs.iter().enumerate() {
            if expected_index != *index {
                return Err(format!("missing batch input index {expected_index}"));
            }
            validate_required(&self.input_specs, values)
                .map_err(|error| format!("batch input {index}: {error}"))?;
            let dictionary = feature_dictionary(values);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dictionary,
                )
            }
            .map_err(|error| error.localizedDescription().to_string())?;
            providers.push(ProtocolObject::from_retained(provider));
        }
        let provider_refs: Vec<&ProtocolObject<dyn MLFeatureProvider>> =
            providers.iter().map(|provider| &**provider).collect();
        let batch = unsafe {
            MLArrayBatchProvider::initWithFeatureProviderArray(
                MLArrayBatchProvider::alloc(),
                &NSArray::from_slice(&provider_refs),
            )
        };
        let options = unsafe { MLPredictionOptions::new() };
        let actual = unsafe {
            model.predictionsFromBatch_options_error(ProtocolObject::from_ref(&*batch), &options)
        }
        .map_err(|error| error.localizedDescription().to_string())?;
        if usize::try_from(unsafe { actual.count() }).ok() != Some(inputs.len()) {
            return Err("batch output count does not match the request".into());
        }
        Ok(BatchOutput {
            provider: Some(actual),
            error: None,
        })
    }
}

/// Creates a model that loads a caller-owned compiled `.mlmodelc` directory.
pub fn modelWithPath(path: impl AsRef<Path>, compute: ComputePlatform) -> Model {
    Model::from_path(path.as_ref().to_path_buf(), compute)
}

/// Batch equivalent of [`modelWithPath`].
pub fn modelWithPathBatch(path: impl AsRef<Path>, compute: ComputePlatform) -> BatchModel {
    BatchModel::from_path(path.as_ref().to_path_buf(), compute)
}

/// Builds the configuration used for every model load.
///
/// The historical `experimentalMLE5EngineUsage` KVC switch is deliberately not
/// set: it trades a ~4x slower model load for a faster prediction engine, and
/// the single-model path never used it. If a workload wants that trade, set the
/// key here again.
fn model_configuration(compute_units: MLComputeUnits) -> Retained<MLModelConfiguration> {
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(compute_units) };
    config
}

/// Validates the filesystem container, then lets CoreML validate its contents.
/// A `.mlmodelc` suffix identifies the bundle format, not whether it is encrypted.
fn load_compiled_model(
    path: &Path,
    compute_units: MLComputeUnits,
) -> Result<Retained<MLModel>, LoadError> {
    if !path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mlmodelc"))
    {
        return Err(LoadError::InvalidPath(format!(
            "{} must have the .mlmodelc extension",
            path.display()
        )));
    }
    let metadata = std::fs::metadata(path)
        .map_err(|error| LoadError::InvalidPath(format!("{}: {error}", path.display())))?;
    if !metadata.is_dir() {
        return Err(LoadError::InvalidPath(format!(
            "{} is not a directory",
            path.display()
        )));
    }
    with_pool(|| {
        let url = file_url(path)?;
        let configuration = model_configuration(compute_units);
        load_url(&url, &configuration)
    })
}

/// Blocks only the calling thread; CoreML owns the asynchronous work and invokes
/// its completion on its own queue, without requiring a caller run loop.
fn load_url(
    url: &NSURL,
    configuration: &MLModelConfiguration,
) -> Result<Retained<MLModel>, LoadError> {
    struct SendModel(Retained<MLModel>);
    // SAFETY: MLModel supports use on arbitrary threads. The retained reference
    // has one owner after the mutex handoff; it is never accessed concurrently.
    unsafe impl Send for SendModel {}

    type Slot = Mutex<Option<Result<SendModel, LoadError>>>;
    let slot = Arc::new((Slot::new(None), Condvar::new()));
    let signal = Arc::clone(&slot);
    // The block captures only owned state. Retain the borrowed model (or copy
    // NSError fields) before the callback returns and CoreML releases them.
    let handler = RcBlock::new(move |model: *mut MLModel, error: *mut NSError| {
        let result = if let Some(error) = unsafe { error.as_ref() } {
            Err(LoadError::Native {
                domain: error.domain().to_string(),
                code: error.code(),
                message: error.localizedDescription().to_string(),
            })
        } else if let Some(model) = unsafe { Retained::retain(model) } {
            Ok(SendModel(model))
        } else {
            Err(LoadError::MissingModel)
        };
        let (lock, condvar) = &*signal;
        let mut guard = lock.lock().unwrap_or_else(|poison| poison.into_inner());
        *guard = Some(result);
        condvar.notify_one();
    });
    unsafe {
        MLModel::loadContentsOfURL_configuration_completionHandler(url, configuration, &handler)
    };
    let (lock, condvar) = &*slot;
    let mut guard = lock.lock().unwrap_or_else(|poison| poison.into_inner());
    while guard.is_none() {
        guard = condvar
            .wait(guard)
            .unwrap_or_else(|poison| poison.into_inner());
    }
    guard
        .take()
        .expect("completion published a result before waking the caller")
        .map(|model| model.0)
}

fn native_dtype(dtype: MLMultiArrayDataType) -> Result<MLDataType, String> {
    match dtype {
        MLMultiArrayDataType::Float32 => Ok(MLDataType::F32),
        MLMultiArrayDataType::Float16 => Ok(MLDataType::F16),
        MLMultiArrayDataType::Int32 => Ok(MLDataType::I32),
        _ => Err(format!("unsupported native multiarray dtype {dtype:?}")),
    }
}

fn native_type(dtype: MLDataType) -> Result<MLMultiArrayDataType, String> {
    match dtype {
        MLDataType::F32 => Ok(MLMultiArrayDataType::Float32),
        MLDataType::F16 => Ok(MLMultiArrayDataType::Float16),
        MLDataType::I32 => Ok(MLMultiArrayDataType::Int32),
        _ => Err(format!("unsupported native multiarray dtype {dtype}")),
    }
}

fn checked_count(shape: &[usize]) -> Result<usize, String> {
    if shape.iter().any(|&dim| dim > isize::MAX as usize) {
        return Err(format!("invalid native tensor shape {shape:?}"));
    }
    shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .filter(|&count| count <= isize::MAX as usize)
            .ok_or_else(|| format!("tensor shape {shape:?} overflows native element count"))
    })
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
    checked_count(shape)?;
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1]
            .checked_mul(shape[index + 1])
            .ok_or("tensor stride overflow")?;
    }
    Ok(strides)
}

fn shape_and_strides(
    shape: &[usize],
) -> Result<(Retained<NSArray<NSNumber>>, Retained<NSArray<NSNumber>>), String> {
    Ok((
        numbers_to_array(shape),
        numbers_to_array(&row_major_strides(shape)?),
    ))
}

fn numbers_to_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&value| NSNumber::new_usize(value))
        .collect();
    let refs: Vec<&NSNumber> = numbers.iter().map(|number| &**number).collect();
    NSArray::from_slice(&refs)
}

fn array_to_dimensions(array: &NSArray<NSNumber>) -> Result<Vec<usize>, String> {
    (0..array.len())
        .map(|index| {
            usize::try_from(array.objectAtIndex(index).as_isize())
                .map_err(|_| "negative native dimension or stride".to_string())
        })
        .collect()
}

fn matches_native_shape(native: &NSArray<NSNumber>, expected: &[usize]) -> bool {
    native.len() == expected.len()
        && expected.iter().enumerate().all(|(index, &dim)| {
            usize::try_from(native.objectAtIndex(index).as_isize()).ok() == Some(dim)
        })
}

fn shape_range(location: usize, length: usize) -> Result<(usize, Option<usize>), String> {
    if location > isize::MAX as usize || length == 0 {
        return Err("invalid native shape range".into());
    }
    // CoreML represents an unbounded upper limit using a maximal native range.
    let upper = if length >= isize::MAX as usize {
        None
    } else {
        let upper = location
            .checked_add(length - 1)
            .ok_or("shape range overflow")?;
        if upper >= isize::MAX as usize {
            None
        } else {
            Some(upper)
        }
    };
    Ok((location, upper))
}

fn tensor_infos(
    descriptions: &NSDictionary<NSString, MLFeatureDescription>,
) -> Result<Vec<TensorInfo>, String> {
    let (names, descriptions) = descriptions.to_vecs();
    let mut infos = Vec::with_capacity(names.len());
    for (name, description) in names.iter().zip(&descriptions) {
        let name = name.to_string();
        let native = unsafe { description.multiArrayConstraint() }
            .ok_or_else(|| format!("feature `{name}` is not a supported multiarray"))?;
        let dtype = native_dtype(unsafe { native.dataType() })
            .map_err(|error| format!("feature `{name}`: {error}"))?;
        let shape = array_to_dimensions(&*unsafe { native.shape() })?;
        checked_count(&shape)?;
        let native_shape = unsafe { native.shapeConstraint() };
        let constraint = match unsafe { native_shape.r#type() } {
            MLMultiArrayShapeConstraintType::Unspecified => ShapeConstraint::Unconstrained,
            MLMultiArrayShapeConstraintType::Enumerated => {
                let shapes = unsafe { native_shape.enumeratedShapes() };
                if shapes.is_empty() {
                    return Err(format!("feature `{name}` has no enumerated shapes"));
                }
                let mut accepted = Vec::with_capacity(shapes.len());
                for index in 0..shapes.len() {
                    let candidate = array_to_dimensions(&shapes.objectAtIndex(index))?;
                    checked_count(&candidate)?;
                    if candidate.len() != shape.len() {
                        return Err(format!("feature `{name}` has inconsistent shape ranks"));
                    }
                    accepted.push(candidate);
                }
                ShapeConstraint::Enumerated(accepted)
            }
            MLMultiArrayShapeConstraintType::Range => {
                let ranges = unsafe { native_shape.sizeRangeForDimension() };
                if ranges.len() != shape.len() {
                    return Err(format!("feature `{name}` has inconsistent range rank"));
                }
                let mut accepted = Vec::with_capacity(ranges.len());
                for index in 0..ranges.len() {
                    let range = unsafe { ranges.objectAtIndex(index).rangeValue() };
                    accepted.push(shape_range(range.location, range.length)?);
                }
                ShapeConstraint::Ranges(accepted)
            }
            kind => {
                return Err(format!(
                    "feature `{name}` has unsupported shape constraint {kind:?}"
                ))
            }
        };
        if !constraint.accepts(&shape) {
            return Err(format!(
                "feature `{name}` default shape violates its constraint"
            ));
        }
        infos.push(TensorInfo {
            name,
            shape,
            dtype,
            constraint,
            optional: unsafe { description.isOptional() },
        });
    }
    infos.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    Ok(infos)
}

fn validate_tensor(specs: &[TensorInfo], name: &str, array: &MLArray) -> Result<(), String> {
    let spec = specs
        .iter()
        .find(|spec| spec.name == name)
        .ok_or_else(|| format!("unknown tensor feature `{name}`"))?;
    if array.dtype() != spec.dtype {
        return Err(format!(
            "feature `{name}` expects dtype {}, found {}",
            spec.dtype,
            array.dtype()
        ));
    }
    if !spec.accepts_shape(array.shape()) {
        return Err(format!(
            "feature `{name}` shape {:?} violates {:?}",
            array.shape(),
            spec.constraint
        ));
    }
    checked_count(array.shape())?;
    native_type(array.dtype())?;
    Ok(())
}

fn validate_required(
    specs: &[TensorInfo],
    values: &HashMap<String, Retained<MLFeatureValue>>,
) -> Result<(), String> {
    for spec in specs {
        if !spec.optional && !values.contains_key(&spec.name) {
            return Err(format!("missing required input `{}`", spec.name));
        }
    }
    Ok(())
}

/// Preserve dense allocations (including excess capacity); normalize only
/// non-standard layouts or a nonzero logical offset.
fn owned_contiguous<T: Copy>(array: ndarray::ArrayD<T>) -> Result<(Vec<usize>, Vec<T>), String> {
    let shape = array.shape().to_vec();
    let count = checked_count(&shape)?;
    let array = if array.is_standard_layout() {
        array
    } else {
        array.as_standard_layout().into_owned()
    };
    let (mut data, offset) = array.into_raw_vec_and_offset();
    let offset = offset.unwrap_or(0);
    let end = offset.checked_add(count).ok_or("tensor offset overflow")?;
    if end > data.len() {
        return Err("tensor offset exceeds initialized storage".into());
    }
    if offset != 0 {
        data.copy_within(offset..end, 0);
    }
    data.truncate(count);
    Ok((shape, data))
}

fn owned_multi_array<T: crate::mlarray::MLType + Copy + 'static>(
    array: ndarray::ArrayD<T>,
) -> Result<Retained<MLMultiArray>, String> {
    let (shape, mut data) = owned_contiguous(array)?;
    let dtype = native_type(T::DATA_TYPE)?;
    let pointer = data.as_mut_ptr().cast();
    // The block owns the entire Vec, including its exact initialized length and
    // capacity, even if CoreML invokes a deallocator during initializer failure.
    // Releasing the final block reference is the only place the Vec is dropped.
    let deallocator = RcBlock::new(move |_pointer: NonNull<std::ffi::c_void>| {
        let _owner = &data;
    });
    multi_array(pointer, &shape, dtype, Some(&deallocator))
}

fn input_value(input: MLArray) -> Result<Retained<MLFeatureValue>, String> {
    let array = match input {
        MLArray::Float32Array(array) => owned_multi_array(array)?,
        MLArray::Float16Array(array) => owned_multi_array(array)?,
        MLArray::Int32Array(array) => owned_multi_array(array)?,
        other => return Err(format!("unsupported input dtype {}", other.dtype())),
    };
    Ok(unsafe { MLFeatureValue::featureValueWithMultiArray(&array) })
}

fn output_backing(output: MLArray) -> Result<Backing, String> {
    let (shape, buffer) = match output {
        MLArray::Float32Array(array) => {
            let (shape, data) = owned_contiguous(array)?;
            (shape, BackingBuffer::F32(data))
        }
        MLArray::Float16Array(array) => {
            let (shape, data) = owned_contiguous(array)?;
            (shape, BackingBuffer::F16(data))
        }
        MLArray::Int32Array(array) => {
            let (shape, data) = owned_contiguous(array)?;
            (shape, BackingBuffer::I32(data))
        }
        other => {
            return Err(format!(
                "unsupported output backing dtype {}",
                other.dtype()
            ))
        }
    };
    new_backing(shape, buffer)
}

/// Native enumerations/ranges can describe one fixed shape. An unconstrained
/// output's default dimensions are not a guarantee about its next result.
fn fixed_output_shape(spec: &TensorInfo) -> Option<&[usize]> {
    match &spec.constraint {
        ShapeConstraint::Fixed(shape) => Some(shape),
        ShapeConstraint::Enumerated(shapes) if shapes.len() == 1 => Some(&shapes[0]),
        ShapeConstraint::Ranges(ranges)
            if ranges.len() == spec.shape.len()
                && ranges
                    .iter()
                    .zip(&spec.shape)
                    .all(|(&(min, max), &dimension)| max == Some(min) && dimension == min) =>
        {
            Some(&spec.shape)
        }
        _ => None,
    }
}

fn zero_backing(dtype: MLDataType, shape: &[usize]) -> Result<Backing, String> {
    new_backing(shape.to_vec(), zero_buffer(dtype, shape)?)
}

fn new_backing(shape: Vec<usize>, mut buffer: BackingBuffer) -> Result<Backing, String> {
    let noop = RcBlock::new(|_pointer: NonNull<std::ffi::c_void>| {});
    let array = multi_array(
        buffer.pointer(),
        &shape,
        native_type(buffer.dtype())?,
        Some(&noop),
    )?;
    Ok(Backing {
        array,
        buffer,
        shape,
        used: false,
    })
}

fn array_from_vec<T: crate::mlarray::MLType>(
    shape: &[usize],
    values: Vec<T>,
) -> Result<MLArray, String> {
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), values)
        .map(MLArray::from)
        .map_err(|error| format!("output shape {shape:?} does not match storage: {error}"))
}

fn zero_buffer(dtype: MLDataType, shape: &[usize]) -> Result<BackingBuffer, String> {
    let count = checked_count(shape)?;
    let element_size = match dtype {
        MLDataType::F16 => 2,
        MLDataType::F32 | MLDataType::I32 => 4,
        _ => return Err(format!("unsupported output backing dtype {dtype}")),
    };
    if count
        .checked_mul(element_size)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .is_none()
    {
        return Err("output backing byte length overflow".into());
    }
    match dtype {
        MLDataType::F32 => Ok(BackingBuffer::F32(vec![0.0f32; count])),
        MLDataType::F16 => Ok(BackingBuffer::F16(vec![half::f16::ZERO; count])),
        MLDataType::I32 => Ok(BackingBuffer::I32(vec![0i32; count])),
        _ => Err(format!("unsupported output backing dtype {dtype}")),
    }
}

pub(crate) fn zero_array(dtype: MLDataType, shape: &[usize]) -> Result<MLArray, String> {
    zero_buffer(dtype, shape)?.into_array(shape)
}

/// Creates an `MLMultiArray` that borrows the caller's buffer.
fn multi_array(
    data: *mut std::ffi::c_void,
    shape: &[usize],
    data_type: MLMultiArrayDataType,
    deallocator: Option<&DynBlock<dyn Fn(NonNull<std::ffi::c_void>)>>,
) -> Result<Retained<MLMultiArray>, String> {
    let data = NonNull::new(data).ok_or_else(|| "null buffer pointer".to_string())?;
    let count = checked_count(shape)?;
    let element_size = match native_dtype(data_type)? {
        MLDataType::F16 => 2,
        MLDataType::F32 | MLDataType::I32 => 4,
        _ => unreachable!("native_dtype only returns supported native types"),
    };
    if count
        .checked_mul(element_size)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .is_none()
    {
        return Err("tensor buffer byte length overflow".into());
    }
    let (shape, strides) = shape_and_strides(shape)?;
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            data,
            &shape,
            data_type,
            &strides,
            deallocator,
        )
    }
    .map_err(|err| err.localizedDescription().to_string())
}

fn feature_dictionary(
    features: &HashMap<String, Retained<MLFeatureValue>>,
) -> Retained<NSDictionary<NSString, AnyObject>> {
    let keys: Vec<Retained<NSString>> = features
        .keys()
        .map(|name| NSString::from_str(name))
        .collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|key| &**key).collect();
    let value_refs: Vec<&MLFeatureValue> = features.values().map(|value| &**value).collect();
    let dictionary: Retained<NSDictionary<NSString, MLFeatureValue>> =
        NSDictionary::from_slices(&key_refs, &value_refs);
    // SAFETY: MLFeatureValue is an NSObject subclass, so the type parameter is
    // only a compile-time view of the same objects.
    unsafe { Retained::cast_unchecked(dictionary) }
}

fn backing_dictionary(outputs: &Backings) -> Retained<NSDictionary<NSString, AnyObject>> {
    let keys: Vec<Retained<NSString>> = outputs
        .keys()
        .map(|name| NSString::from_str(name))
        .collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|key| &**key).collect();
    let value_refs: Vec<&MLMultiArray> = outputs.values().map(|backing| &*backing.array).collect();
    let dictionary: Retained<NSDictionary<NSString, MLMultiArray>> =
        NSDictionary::from_slices(&key_refs, &value_refs);
    // SAFETY: MLMultiArray is an NSObject subclass.
    unsafe { Retained::cast_unchecked(dictionary) }
}

fn multi_array_value(
    provider: &ProtocolObject<dyn MLFeatureProvider>,
    name: &str,
) -> Result<Option<Retained<MLMultiArray>>, String> {
    let key = NSString::from_str(name);
    let Some(value) = (unsafe { provider.featureValueForName(&key) }) else {
        return Ok(None);
    };
    if unsafe { value.isUndefined() } {
        return Ok(None);
    }
    unsafe { value.multiArrayValue() }
        .map(Some)
        .ok_or_else(|| format!("output `{name}` is not a multiarray"))
}

fn dense_layout(
    shape: &[usize],
    strides: impl ExactSizeIterator<Item = Option<usize>> + DoubleEndedIterator,
) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1usize;
    for (&dim, stride) in shape.iter().zip(strides).rev() {
        if stride.is_none() || (dim != 1 && stride != Some(expected)) {
            return false;
        }
        let Some(next) = expected.checked_mul(dim) else {
            return false;
        };
        expected = next;
    }
    true
}

fn dense_native_layout(shape: &[usize], strides: &NSArray<NSNumber>) -> bool {
    dense_layout(
        shape,
        (0..strides.len())
            .map(|index| usize::try_from(strides.objectAtIndex(index).as_isize()).ok()),
    )
}

fn verify_backing(
    array: &MLMultiArray,
    buffer: &mut BackingBuffer,
    shape: &[usize],
) -> Result<bool, String> {
    let expected = buffer.pointer();
    let bytes = buffer
        .len()
        .checked_mul(match buffer.dtype() {
            MLDataType::F16 => 2,
            _ => 4,
        })
        .ok_or("backing byte length overflow")?;
    let result = std::cell::RefCell::new(Err("native byte handler did not run".to_string()));
    let block = RcBlock::new(|pointer: NonNull<std::ffi::c_void>, size: isize| {
        let check = (|| {
            let strides = unsafe { array.strides() };
            if pointer.as_ptr() != expected || !dense_native_layout(shape, &strides) {
                return Ok(false);
            }
            if usize::try_from(size)
                .ok()
                .filter(|&size| size >= bytes)
                .is_none()
            {
                return Err("native output backing is shorter than its tensor".into());
            }
            Ok(true)
        })();
        *result.borrow_mut() = check;
    });
    unsafe { array.getBytesWithHandler(&block) };
    drop(block);
    result.into_inner()
}

/// Reads the framework's materialized buffer using its actual element strides.
/// Dense outputs retain a single bulk-copy fast path.
fn copy_bytes<T: crate::mlarray::MLType + Copy + 'static>(
    array: &MLMultiArray,
    shape: &[usize],
) -> Result<Vec<T>, String> {
    if native_dtype(unsafe { array.dataType() })? != T::DATA_TYPE {
        return Err("native output extraction dtype mismatch".into());
    }
    let count = checked_count(shape)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let output = std::cell::RefCell::new(Err("native byte handler did not run".to_string()));
    let block = RcBlock::new(|pointer: NonNull<std::ffi::c_void>, size: isize| {
        let copy = (|| {
            let size = usize::try_from(size).map_err(|_| "negative native buffer size")?;
            // Query inside the handler: materialization may change the layout.
            let strides = unsafe { array.strides() };
            if dense_native_layout(shape, &strides) {
                // SAFETY: dtype and the dense native layout were checked.
                unsafe { copy_contiguous::<T>(pointer, size, count) }
            } else {
                let strides = array_to_dimensions(&strides)?;
                // SAFETY: CoreML lends initialized scalars with this layout
                // for the duration of the callback.
                unsafe { copy_layout::<T>(pointer, size, shape, &strides) }
            }
        })();
        *output.borrow_mut() = copy;
    });
    unsafe { array.getBytesWithHandler(&block) };
    drop(block);
    output.into_inner()
}

/// The first `count` scalars must be initialized values of T.
unsafe fn copy_contiguous<T: Copy>(
    pointer: NonNull<std::ffi::c_void>,
    bytes: usize,
    count: usize,
) -> Result<Vec<T>, String> {
    let required = count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or("native output byte length overflow")?;
    if required > bytes {
        return Err("native output storage is shorter than its dense shape".into());
    }
    if (pointer.as_ptr() as usize) % std::mem::align_of::<T>() == 0 {
        // SAFETY: alignment and length are checked; initialized, live storage
        // is guaranteed by the native callback.
        Ok(unsafe { std::slice::from_raw_parts(pointer.as_ptr().cast::<T>(), count) }.to_vec())
    } else {
        Ok((0..count)
            .map(|index| {
                // SAFETY: each addressed scalar lies within the checked storage.
                unsafe { pointer.as_ptr().cast::<T>().add(index).read_unaligned() }
            })
            .collect())
    }
}

/// The addressed elements must be initialized values of T for this call.
unsafe fn copy_layout<T: Copy>(
    pointer: NonNull<std::ffi::c_void>,
    bytes: usize,
    shape: &[usize],
    strides: &[usize],
) -> Result<Vec<T>, String> {
    let count = checked_count(shape)?;
    if shape.len() != strides.len() {
        return Err("native output stride rank mismatch".into());
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let last_offset = shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |offset, (&dim, &stride)| {
            (dim - 1)
                .checked_mul(stride)
                .and_then(|extent| offset.checked_add(extent))
        })
        .ok_or("native output stride overflow")?;
    let required = last_offset
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<T>()))
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or("native output byte length overflow")?;
    if required > bytes {
        return Err("native output storage is shorter than its strided shape".into());
    }
    if dense_layout(shape, strides.iter().copied().map(Some)) {
        // SAFETY: the caller supplies initialized scalars with this layout.
        return unsafe { copy_contiguous::<T>(pointer, bytes, count) };
    }
    let mut values = Vec::with_capacity(count);
    for linear in 0..count {
        let mut remaining = linear;
        let mut offset = 0usize;
        for (&dim, &stride) in shape.iter().zip(strides).rev() {
            offset += (remaining % dim) * stride;
            remaining /= dim;
        }
        // SAFETY: max addressed offset and byte length were checked above.
        values.push(unsafe { pointer.as_ptr().cast::<T>().add(offset).read_unaligned() });
    }
    Ok(values)
}

/// Builds a file URL without treating filesystem bytes as a URL or lossy UTF-8.
fn file_url(path: &Path) -> Result<Retained<NSURL>, LoadError> {
    let bytes = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| LoadError::InvalidPath("path contains a NUL byte".into()))?;
    // SAFETY: CString supplies a valid NUL-terminated filesystem representation
    // for the duration of this call. NSURL copies it into the returned URL.
    Ok(unsafe {
        NSURL::fileURLWithFileSystemRepresentation_isDirectory_relativeToURL(
            NonNull::new(bytes.as_ptr().cast_mut()).expect("CString pointer is non-null"),
            true,
            None,
        )
    })
}

/// Wraps a blocking CoreML call in an autorelease pool, as Swift did for us.
fn with_pool<T>(f: impl FnOnce() -> T) -> T {
    autoreleasepool(|_pool| f())
}

/// The ML array types this module can hand back.
pub struct MLModelOutput {
    pub outputs: HashMap<String, MLArray>,
}

/// One prediction's worth of batch outputs.
pub struct MLBatchModelOutput {
    pub outputs: Vec<HashMap<String, MLArray>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn spec(name: &str, dtype: MLDataType, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
            constraint: ShapeConstraint::Fixed(shape.to_vec()),
            optional: false,
        }
    }

    fn provider(
        name: &str,
        array: &MLMultiArray,
    ) -> Retained<ProtocolObject<dyn MLFeatureProvider>> {
        let value = unsafe { MLFeatureValue::featureValueWithMultiArray(array) };
        let values = HashMap::from([(name.to_string(), value)]);
        let dictionary = feature_dictionary(&values);
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dictionary,
            )
        }
        .expect("valid native provider");
        ProtocolObject::from_retained(provider)
    }

    #[test]
    fn shapes_use_checked_row_major_strides() {
        let (shape, strides) = shape_and_strides(&[2, 3, 4]).unwrap();
        assert_eq!(array_to_dimensions(&shape).unwrap(), vec![2, 3, 4]);
        assert_eq!(array_to_dimensions(&strides).unwrap(), vec![12, 4, 1]);
        assert!(shape_and_strides(&[isize::MAX as usize, 2]).is_err());
        assert!(zero_array(MLDataType::F32, &[isize::MAX as usize]).is_err());
        assert!(native_dtype(MLMultiArrayDataType::Double).is_err());
        assert!(native_type(MLDataType::U16).is_err());
    }

    #[test]
    fn native_shape_ranges_are_inclusive_and_can_be_unbounded() {
        assert_eq!(shape_range(2, 3).unwrap(), (2, Some(4)));
        assert_eq!(shape_range(1, usize::MAX).unwrap(), (1, None));
        assert_eq!(shape_range(0, 4).unwrap(), (0, Some(3)));
        assert!(shape_range(1, 0).is_err());
    }

    #[test]
    fn dense_normalization_preserves_initialized_length_capacity_and_allocation() {
        let mut data = Vec::with_capacity(19);
        data.extend_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let capacity = data.capacity();
        let pointer = data.as_ptr();
        let array = ArrayD::from_shape_vec(IxDyn(&[2, 2]), data).unwrap();
        let (shape, data) = owned_contiguous(array).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data.len(), 4);
        assert_eq!(data.capacity(), capacity);
        assert_eq!(data.as_ptr(), pointer);
    }

    #[test]
    fn offset_and_transposed_owned_inputs_follow_logical_order() {
        let array = ndarray::Array1::from_vec(vec![0i32, 1, 2, 3, 4, 5])
            .slice_move(ndarray::s![2..5])
            .into_dyn();
        let (_, data) = owned_contiguous(array).unwrap();
        assert_eq!(data, vec![2, 3, 4]);
        let array = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .reversed_axes()
            .into_dyn();
        let native = owned_multi_array(array).unwrap();
        assert_eq!(
            array_to_dimensions(&*unsafe { native.shape() }).unwrap(),
            vec![3, 2]
        );
        assert_eq!(
            copy_bytes::<f32>(&native, &[3, 2]).unwrap(),
            vec![1., 4., 2., 5., 3., 6.]
        );
    }

    #[test]
    fn native_provider_copyout_honors_actual_strides_and_dynamic_shape() {
        let mut storage = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let shape = numbers_to_array(&[2, 3]);
        let strides = numbers_to_array(&[1, 2]);
        let noop = RcBlock::new(|_pointer: NonNull<std::ffi::c_void>| {});
        // SAFETY: storage remains alive until after all native references are
        // dropped, and the strided shape addresses exactly its six elements.
        let array = unsafe {
            MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                MLMultiArray::alloc(),
                NonNull::new(storage.as_mut_ptr().cast()).unwrap(),
                &shape,
                MLMultiArrayDataType::Float32,
                &strides,
                Some(&noop),
            )
        }
        .unwrap();
        let mut output_spec = spec("result", MLDataType::F32, &[1, 3]);
        output_spec.constraint = ShapeConstraint::Ranges(vec![(1, Some(4)), (3, Some(3))]);
        let outputs = ModelOutput::from_provider(provider("result", &array))
            .into_outputs(&[output_spec])
            .unwrap();
        let tensor = outputs
            .into_iter()
            .next()
            .unwrap()
            .1
            .extract_to_tensor::<f32>()
            .unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_slice().unwrap(), &[1., 2., 3., 4., 5., 6.]);
        assert!(copy_bytes::<i32>(&array, &[2, 3]).is_err());
        let pointer = NonNull::new(storage.as_mut_ptr().cast()).unwrap();
        // SAFETY: the helper rejects the short byte range before reading.
        assert!(unsafe { copy_layout::<f32>(pointer, 4, &[2, 3], &[1, 2]) }.is_err());
    }

    #[test]
    fn native_output_extraction_reports_name_type_and_shape_mismatches() {
        let array = owned_multi_array(ndarray::arr1(&[1.0f32, 2.0]).into_dyn()).unwrap();
        assert!(ModelOutput::from_provider(provider("actual", &array))
            .into_outputs(&[spec("expected", MLDataType::F32, &[2])])
            .is_err());
        assert!(ModelOutput::from_provider(provider("actual", &array))
            .into_outputs(&[spec("actual", MLDataType::F16, &[2])])
            .is_err());
        assert!(ModelOutput::from_provider(provider("actual", &array))
            .into_outputs(&[spec("actual", MLDataType::F32, &[3])])
            .is_err());
    }

    #[test]
    fn declined_output_backings_copy_actual_values_instead_of_unused_storage() {
        let array = owned_multi_array(ndarray::arr1(&[7.0f32, -3.5]).into_dyn()).unwrap();
        let mut output = ModelOutput::from_provider(provider("actual", &array));
        output.backings.insert(
            "actual".into(),
            output_backing(zero_array(MLDataType::F32, &[2]).unwrap()).unwrap(),
        );
        let mut outputs = output
            .into_outputs(&[spec("actual", MLDataType::F32, &[2])])
            .unwrap();
        assert_eq!(
            outputs
                .remove("actual")
                .unwrap()
                .extract_to_tensor::<f32>()
                .unwrap()
                .as_slice()
                .unwrap(),
            &[7.0, -3.5]
        );
    }

    #[test]
    fn native_backed_outputs_move_f32_f16_and_i32_storage_without_copying() {
        for dtype in [MLDataType::F32, MLDataType::F16, MLDataType::I32] {
            let mut backing = zero_backing(dtype, &[2, 3]).unwrap();
            let pointer = backing.buffer.pointer();
            let mut output = ModelOutput::from_provider(provider("result", &backing.array));
            output.backings.insert("result".into(), backing);
            let mut outputs = output
                .into_outputs(&[spec("result", dtype, &[2, 3])])
                .unwrap();
            let result = outputs.remove("result").unwrap();
            assert_eq!(result.dtype(), dtype);
            assert_eq!(result.shape(), &[2, 3]);
            let actual = match result {
                MLArray::Float32Array(array) => array.as_ptr().cast::<std::ffi::c_void>(),
                MLArray::Float16Array(array) => array.as_ptr().cast::<std::ffi::c_void>(),
                MLArray::Int32Array(array) => array.as_ptr().cast::<std::ffi::c_void>(),
                _ => unreachable!(),
            };
            assert_eq!(actual, pointer.cast_const());
        }
    }

    #[test]
    fn required_optional_and_dtype_validation_happens_before_native_binding() {
        let mut optional = spec("optional", MLDataType::F16, &[2]);
        optional.optional = true;
        let required = spec("required", MLDataType::I32, &[2]);
        let specs = [required, optional];
        assert!(validate_required(&specs, &HashMap::new()).is_err());
        let values = HashMap::from([(
            "required".into(),
            input_value(ndarray::arr1(&[1i32, 2]).into_dyn().into()).unwrap(),
        )]);
        assert!(validate_required(&specs, &values).is_ok());
        let wrong: MLArray = ndarray::arr1(&[1.0f32, 2.0]).into_dyn().into();
        assert!(validate_tensor(&specs, "required", &wrong).is_err());
        assert!(validate_tensor(&specs, "unknown", &wrong).is_err());
    }

    #[test]
    fn validation_and_prediction_failures_discard_single_and_batch_requests() {
        let mut model = modelWithPath("unused.mlmodelc", ComputePlatform::Cpu);
        model.inputs.insert(
            "stale".into(),
            input_value(zero_array(MLDataType::F32, &[2]).unwrap()).unwrap(),
        );
        model.outputs.insert(
            "stale".into(),
            output_backing(zero_array(MLDataType::F32, &[2]).unwrap()).unwrap(),
        );
        assert!(model
            .bind_input("unknown", zero_array(MLDataType::F32, &[2]).unwrap())
            .is_err());
        assert!(model.inputs.is_empty() && model.outputs.is_empty());
        model.inputs.insert(
            "stale".into(),
            input_value(zero_array(MLDataType::F32, &[2]).unwrap()).unwrap(),
        );
        model.outputs.insert(
            "stale".into(),
            output_backing(zero_array(MLDataType::F32, &[2]).unwrap()).unwrap(),
        );
        assert!(model.predict().getError().is_some());
        assert!(model.inputs.is_empty() && model.outputs.is_empty());

        let mut batch = modelWithPathBatch("unused.mlmodelc", ComputePlatform::Cpu);
        let values = HashMap::from([(
            "stale".into(),
            input_value(zero_array(MLDataType::F32, &[2]).unwrap()).unwrap(),
        )]);
        batch.inputs.insert(0, values.clone());
        assert!(batch
            .bind_input("unknown", zero_array(MLDataType::F32, &[2]).unwrap(), -1)
            .is_err());
        assert!(batch.inputs.is_empty());
        batch.inputs.insert(0, values);
        assert!(batch.predict().getError().is_some());
        assert!(batch.inputs.is_empty());
    }

    #[test]
    fn native_provider_copyout_preserves_f16_and_i32_values() {
        let halves = vec![half::f16::from_f32(1.25), half::f16::from_f32(-2.5)];
        let array =
            owned_multi_array(ndarray::Array1::from_vec(halves.clone()).into_dyn()).unwrap();
        let mut outputs = ModelOutput::from_provider(provider("half", &array))
            .into_outputs(&[spec("half", MLDataType::F16, &[2])])
            .unwrap();
        assert_eq!(
            outputs
                .remove("half")
                .unwrap()
                .extract_to_tensor::<half::f16>()
                .unwrap()
                .as_slice()
                .unwrap(),
            halves.as_slice(),
        );

        let integers = vec![0x1020304i32, -7];
        let array =
            owned_multi_array(ndarray::Array1::from_vec(integers.clone()).into_dyn()).unwrap();
        let mut outputs = ModelOutput::from_provider(provider("integer", &array))
            .into_outputs(&[spec("integer", MLDataType::I32, &[2])])
            .unwrap();
        assert_eq!(
            outputs
                .remove("integer")
                .unwrap()
                .extract_to_tensor::<i32>()
                .unwrap()
                .as_slice()
                .unwrap(),
            integers.as_slice(),
        );
    }

    #[test]
    fn only_optional_outputs_may_be_absent() {
        let array = owned_multi_array(ndarray::arr1(&[1.0f32, 2.0]).into_dyn()).unwrap();
        let mut specs = [
            spec("actual", MLDataType::F32, &[2]),
            spec("missing", MLDataType::F32, &[2]),
        ];
        assert!(ModelOutput::from_provider(provider("actual", &array))
            .into_outputs(&specs)
            .is_err());
        specs[1].optional = true;
        let outputs = ModelOutput::from_provider(provider("actual", &array))
            .into_outputs(&specs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs.contains_key("actual"));
    }
}

#[cfg(test)]
mod backing_shape_regressions {
    use super::*;

    #[test]
    fn unconstrained_outputs_use_returned_rank_instead_of_default_dimensions() {
        let array = owned_multi_array(ndarray::arr2(&[[1.0_f32, 2., 3.], [4., 5., 6.]]).into_dyn())
            .unwrap();
        let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
        let dictionary = feature_dictionary(&HashMap::from([("output".to_owned(), value)]));
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dictionary,
            )
        }
        .unwrap();
        let mut spec = TensorInfo {
            name: "output".into(),
            shape: vec![4],
            dtype: MLDataType::F32,
            constraint: ShapeConstraint::Unconstrained,
            optional: false,
        };
        assert!(fixed_output_shape(&spec).is_none());
        let mut outputs = ModelOutput::from_provider(ProtocolObject::from_retained(provider))
            .into_outputs(&[spec.clone()])
            .unwrap();
        let output = outputs
            .remove("output")
            .unwrap()
            .extract_to_tensor::<f32>()
            .unwrap();
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.as_slice().unwrap(), &[1., 2., 3., 4., 5., 6.]);

        spec.constraint = ShapeConstraint::Enumerated(vec![vec![4]]);
        assert_eq!(fixed_output_shape(&spec), Some(&[4][..]));
        spec.constraint = ShapeConstraint::Enumerated(vec![vec![2], vec![4]]);
        assert!(fixed_output_shape(&spec).is_none());
        spec.constraint = ShapeConstraint::Ranges(vec![(4, Some(4))]);
        assert_eq!(fixed_output_shape(&spec), Some(&[4][..]));
        spec.constraint = ShapeConstraint::Ranges(vec![(2, Some(4))]);
        assert!(fixed_output_shape(&spec).is_none());
    }
}
