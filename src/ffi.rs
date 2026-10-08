//! CoreML inference layer implemented directly on top of Apple's Objective-C
//! CoreML framework through `objc2`.
//!
//! This module replaces the former Swift + `swift-bridge` shim. `objc2-core-ml`
//! already provides every framework class and method used here, so the only
//! hand-written binding is `+[MLModel compileModelAtURL:error:]`, which the
//! published bindings do not expose.
//!
//! Type and function names mirror the generated bindings that the rest of the
//! crate (and downstream users of `coreml_rs::ffi`) relied on, so switching the
//! implementation is invisible above this module. What used to live in
//! `swift-library/Sources/swift-library/swift_library.swift` — buffer-backed
//! `MLMultiArray`s, output backings, batch assembly, description mapping and
//! model lifecycle — now lives here.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use block2::{DynBlock, RcBlock};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{msg_send, AnyThread, ClassType};
use objc2_core_ml::{
    MLArrayBatchProvider, MLBatchProvider, MLComputeUnits, MLDictionaryFeatureProvider,
    MLFeatureDescription, MLFeatureProvider, MLFeatureValue, MLModel, MLModelAsset,
    MLModelConfiguration, MLModelDescription, MLMultiArray, MLMultiArrayDataType,
    MLPredictionOptions,
};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSError, NSNumber, NSString, NSURL};

use crate::mlarray::MLArray;

/// `-[MLModel compileModelAtURL:error:]` is not present in `objc2-core-ml`
/// (only the asynchronous completion-handler variant is). The synchronous form
/// has been available since macOS 10.13, so call it directly.
unsafe fn compile_model_at_url(model_url: &NSURL) -> Result<Retained<NSURL>, String> {
    let mut error: *mut NSError = std::ptr::null_mut();
    let compiled: *mut NSURL = msg_send![
        MLModel::class(),
        compileModelAtURL: model_url,
        error: &mut error
    ];
    if let Some(compiled) = Retained::retain(compiled) {
        return Ok(compiled);
    }
    Err(if let Some(error) = error.as_ref() {
        error.localizedDescription().to_string()
    } else {
        "MLModel.compileModelAtURL failed without reporting an error".to_string()
    })
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

/// An allocation handed to CoreML by the caller.
///
/// The caller `mem::forget`s its `Vec` once the backing is installed, so this
/// owns the allocation and frees it either when the prediction extracts the
/// data (`into_vec`) or when the backing is dropped without being read.
struct RawBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

impl<T> RawBuffer<T> {
    fn new(ptr: *mut T, len: usize, cap: usize) -> Option<Self> {
        // `len` is the number of initialized elements, `cap` the allocation's
        // capacity. `Vec::from_raw_parts` needs both to free correctly, which
        // is why the old shim's use of the element count as capacity was a bug
        // whenever the two differed.
        NonNull::new(ptr).map(|ptr| Self { ptr, len, cap })
    }

    fn into_vec(self) -> Vec<T> {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` owns the allocation and is not dropped afterwards.
        unsafe { Vec::from_raw_parts(this.ptr.as_ptr(), this.len, this.cap) }
    }
}

impl<T> Drop for RawBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `self` owns the allocation and this runs exactly once.
        unsafe {
            drop(Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap));
        }
    }
}

/// A caller-owned output buffer of one of the supported element types.
enum BackingBuffer {
    F32(RawBuffer<f32>),
    F16(RawBuffer<u16>),
}

/// A single prediction's outputs.
///
/// `Backed` outputs point straight at caller-provided buffers (zero copy on the
/// way out); `Provider` outputs come from a batch prediction and are copied out
/// of the framework's own arrays.
pub struct ModelOutput {
    inner: OutputInner,
    error: Option<String>,
}

enum OutputInner {
    Backed(HashMap<String, (Retained<MLMultiArray>, BackingBuffer)>),
    Provider(Retained<ProtocolObject<dyn MLFeatureProvider>>),
    Empty,
}

impl ModelOutput {
    fn failed(message: impl Into<String>) -> Self {
        Self {
            inner: OutputInner::Empty,
            error: Some(message.into()),
        }
    }

    pub fn getError(&self) -> Option<String> {
        self.error.clone()
    }

    /// Moves the f32 backing buffer for `name` out of the output, if present.
    pub fn outputF32(&mut self, name: String) -> Vec<f32> {
        match &mut self.inner {
            OutputInner::Backed(outputs) => match outputs.remove(&name) {
                Some((_, BackingBuffer::F32(buffer))) => buffer.into_vec(),
                Some((_, other)) => {
                    // Wrong element type; drop the backing to free it.
                    drop(other);
                    Vec::new()
                }
                None => Vec::new(),
            },
            OutputInner::Provider(provider) => {
                multi_array_value(provider, &name).map_or_else(Vec::new, |array| copy_f32(&array))
            }
            OutputInner::Empty => Vec::new(),
        }
    }

    /// Moves the f16 backing buffer for `name` out of the output, if present.
    pub fn outputU16(&mut self, name: String) -> Vec<u16> {
        match &mut self.inner {
            OutputInner::Backed(outputs) => match outputs.remove(&name) {
                Some((_, BackingBuffer::F16(buffer))) => buffer.into_vec(),
                Some((_, other)) => {
                    drop(other);
                    Vec::new()
                }
                None => Vec::new(),
            },
            OutputInner::Provider(provider) => {
                multi_array_value(provider, &name).map_or_else(Vec::new, |array| copy_u16(&array))
            }
            OutputInner::Empty => Vec::new(),
        }
    }

    pub fn outputI32(&mut self, name: String) -> Vec<i32> {
        match &mut self.inner {
            OutputInner::Backed(outputs) => {
                drop(outputs.remove(&name));
                Vec::new()
            }
            OutputInner::Provider(provider) => {
                multi_array_value(provider, &name).map_or_else(Vec::new, |array| copy_i32(&array))
            }
            OutputInner::Empty => Vec::new(),
        }
    }

    pub fn outputDescription(&self) -> Vec<String> {
        match &self.inner {
            OutputInner::Backed(outputs) => outputs
                .iter()
                .map(|(name, (array, _))| format!("{name}:{}", description_of(&**array)))
                .collect(),
            OutputInner::Provider(provider) => {
                let names = unsafe { provider.featureNames() }.allObjects();
                (0..names.len())
                    .filter_map(|index| {
                        let name = names.objectAtIndex(index);
                        let array = multi_array_value(provider, &name.to_string())?;
                        Some(format!("{}:{}", name, description_of(&*array)))
                    })
                    .collect()
            }
            OutputInner::Empty => Vec::new(),
        }
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
            Some(provider) => ModelOutput {
                inner: OutputInner::Provider(unsafe { provider.featuresAtIndex(at) }),
                error: None,
            },
            None => ModelOutput {
                inner: OutputInner::Empty,
                error: self.error.clone(),
            },
        }
    }
}

/// Model input/output descriptions, rendered the way CoreML renders them.
pub struct ModelDescription {
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

    pub fn inputs(&self) -> Vec<String> {
        self.inputs_by_name()
            .map(|descriptions| {
                descriptions
                    .to_vecs()
                    .1
                    .iter()
                    .map(|description| description_of(&**description))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn outputs(&self) -> Vec<String> {
        self.outputs_by_name()
            .map(|descriptions| {
                descriptions
                    .to_vecs()
                    .1
                    .iter()
                    .map(|description| description_of(&**description))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn output_names(&self) -> Vec<String> {
        self.outputs_by_name()
            .map(|descriptions| {
                descriptions
                    .to_vecs()
                    .0
                    .iter()
                    .map(|name| name.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn output_type(&self, name: String) -> String {
        let Some(description) = self
            .outputs_by_name()
            .and_then(|descriptions| descriptions.objectForKey(&NSString::from_str(&name)))
        else {
            return String::new();
        };
        let Some(constraint) = (unsafe { description.multiArrayConstraint() }) else {
            return String::new();
        };
        match unsafe { constraint.dataType() } {
            MLMultiArrayDataType::Float32 => "f32".to_string(),
            MLMultiArrayDataType::Float16 => "f16".to_string(),
            _ => String::new(),
        }
    }

    pub fn output_shape(&self, name: String) -> Vec<usize> {
        self.outputs_by_name()
            .and_then(|descriptions| descriptions.objectForKey(&NSString::from_str(&name)))
            .map(|description| feature_shape(&description))
            .unwrap_or_default()
    }

    pub fn input_shape(&self, name: String) -> Vec<usize> {
        self.inputs_by_name()
            .and_then(|descriptions| descriptions.objectForKey(&NSString::from_str(&name)))
            .map(|description| feature_shape(&description))
            .unwrap_or_default()
    }
}

/// A loaded (or loadable) single-input CoreML model.
pub struct Model {
    compiled_path: Option<PathBuf>,
    model_asset: Option<Retained<MLModelAsset>>,
    model: Option<Retained<MLModel>>,
    inputs: HashMap<String, Retained<MLFeatureValue>>,
    outputs: HashMap<String, (Retained<MLMultiArray>, BackingBuffer)>,
    compute_units: MLComputeUnits,
    failed_to_load: bool,
}

// SAFETY: `MLModel` is documented as safe for concurrent predictions, and every
// mutable field is only touched through `&mut self` (enforced by the wrappers in
// `mlmodel`).
unsafe impl Send for Model {}

impl Model {
    fn empty(compute: ComputePlatform) -> Self {
        Self {
            compiled_path: None,
            model_asset: None,
            model: None,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            compute_units: compute.compute_units(),
            failed_to_load: false,
        }
    }

    /// Loads a model from a path, compiling `.mlpackage`/`.mlmodel` sources when
    /// `compiled` is false, exactly like the Swift initializer used to.
    fn from_path(path: String, compute: ComputePlatform, compiled: bool) -> Self {
        let mut model = Self::empty(compute);
        if compiled {
            model.compiled_path = Some(PathBuf::from(path));
            return model;
        }
        match compile_model(Path::new(&path)) {
            Ok(compiled_path) => model.compiled_path = Some(compiled_path),
            Err(err) => {
                log(&format!("failed to compile {path}: {err}"));
                model.failed_to_load = true;
            }
        }
        model
    }

    /// Builds a model from an in-memory specification (`.mlmodel` bytes).
    ///
    /// The caller must pass an allocation whose length and capacity match and
    /// forget it afterwards; `mlmodel::CoreMLModel::load_buffer` guarantees this.
    fn from_asset(ptr: *mut u8, len: isize, compute: ComputePlatform) -> Self {
        let len = len.max(0) as usize;
        let mut model = Self::empty(compute);
        let buffer = if len == 0 {
            Vec::new()
        } else {
            // SAFETY: the caller hands over ownership of exactly `ptr..ptr+len`.
            unsafe { Vec::from_raw_parts(ptr, len, len) }
        };
        let data = NSData::from_vec(buffer);
        match unsafe { MLModelAsset::modelAssetWithSpecificationData_error(&data) } {
            Ok(asset) => model.model_asset = Some(asset),
            Err(err) => {
                log(&format!(
                    "failed to read model specification: {}",
                    err.localizedDescription()
                ));
                model.failed_to_load = true;
            }
        }
        model
    }

    pub fn load(&mut self) -> bool {
        if self.failed_to_load {
            return false;
        }
        if self.model.is_some() {
            return true;
        }
        let config = model_configuration(self.compute_units);
        if let Some(path) = &self.compiled_path {
            let url = file_url(path);
            return match unsafe {
                MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
            } {
                Ok(model) => {
                    self.model = Some(model);
                    true
                }
                Err(err) => {
                    log(&format!(
                        "failed to load compiled model at {}: {}",
                        path.display(),
                        err.localizedDescription()
                    ));
                    false
                }
            };
        }
        if let Some(asset) = &self.model_asset {
            return match load_asset(asset, &config) {
                Ok(model) => {
                    self.model = Some(model);
                    true
                }
                Err(err) => {
                    log(&format!("failed to load model asset: {err}"));
                    false
                }
            };
        }
        false
    }

    pub fn unload(&mut self) -> bool {
        if self.failed_to_load {
            return false;
        }
        self.model = None;
        true
    }

    /// Whether construction failed, matching the Swift `hasFailedToLoad`.
    pub fn failed(&self) -> bool {
        self.failed_to_load
    }

    pub fn compiled_path(&self) -> Option<String> {
        self.compiled_path.as_ref().map(|path| {
            file_url(path)
                .absoluteString()
                .map(|url| url.to_string())
                .unwrap_or_else(|| path.to_string_lossy().into_owned())
        })
    }

    pub fn description(&self) -> ModelDescription {
        ModelDescription::new(
            self.model
                .as_ref()
                .map(|model| unsafe { model.modelDescription() }),
        )
    }

    pub fn bindInputF32(
        &mut self,
        shape: Vec<usize>,
        feature_name: String,
        data: *mut f32,
        len: usize,
    ) -> bool {
        self.bind_input::<f32>(
            shape,
            feature_name,
            data,
            len,
            MLMultiArrayDataType::Float32,
        )
    }

    pub fn bindInputU16(
        &mut self,
        shape: Vec<usize>,
        feature_name: String,
        data: *mut u16,
        len: usize,
    ) -> bool {
        self.bind_input::<u16>(
            shape,
            feature_name,
            data,
            len,
            MLMultiArrayDataType::Float16,
        )
    }

    pub fn bindInputI32(
        &mut self,
        shape: Vec<usize>,
        feature_name: String,
        data: *mut i32,
        len: usize,
    ) -> bool {
        // The Swift shim declared `float32` here while handing over `i32` data;
        // that mis-declared every integer input to CoreML.
        self.bind_input::<i32>(shape, feature_name, data, len, MLMultiArrayDataType::Int32)
    }

    /// Creates an `MLMultiArray` over the caller's buffer. CoreML takes
    /// ownership and frees the original `Vec` through the deallocator block.
    fn bind_input<T: 'static>(
        &mut self,
        shape: Vec<usize>,
        feature_name: String,
        data: *mut T,
        len: usize,
        data_type: MLMultiArrayDataType,
    ) -> bool {
        let count: usize = shape.iter().product();
        if count > len {
            log(&format!(
                "input `{feature_name}` needs {count} elements but only {len} were provided"
            ));
            return false;
        }
        let cap = len;
        let deallocator = RcBlock::new(move |ptr: NonNull<std::ffi::c_void>| {
            // SAFETY: `T` matches the `Vec<T>` the caller handed over, so the
            // pointer and the full length/capacity pair are reconstructed
            // exactly. (Fixing the old shim's `Vec<f32>`-only free, which
            // corrupted the heap for `u16`/f16 inputs.)
            unsafe {
                drop(Vec::from_raw_parts(ptr.as_ptr().cast::<T>(), count, cap));
            }
        });
        match multi_array(data.cast(), &shape, data_type, Some(&deallocator)) {
            Ok(array) => {
                // SAFETY: the array is a valid MLMultiArray.
                let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
                self.inputs.insert(feature_name, value);
                true
            }
            Err(err) => {
                log(&format!("failed to bind input `{feature_name}`: {err}"));
                false
            }
        }
    }

    /// Registers a caller-owned f32 buffer as the backing store for an output.
    pub fn bindOutputF32(
        &mut self,
        shape: Vec<i32>,
        feature_name: String,
        data: *mut f32,
        len: usize,
    ) -> bool {
        let Some((shape, count)) = backing_shape(&shape, &feature_name) else {
            return false;
        };
        if count > len {
            log(&format!(
                "output `{feature_name}` needs {count} elements but only {len} were provided"
            ));
            return false;
        }
        let Some(buffer) = RawBuffer::new(data, count, len) else {
            return false;
        };
        self.bind_output(
            feature_name,
            shape,
            data.cast(),
            MLMultiArrayDataType::Float32,
            BackingBuffer::F32(buffer),
        )
    }

    /// Registers a caller-owned f16 buffer as the backing store for an output.
    pub fn bindOutputF16(
        &mut self,
        shape: Vec<i32>,
        feature_name: String,
        data: *mut u16,
        len: usize,
    ) -> bool {
        let Some((shape, count)) = backing_shape(&shape, &feature_name) else {
            return false;
        };
        if count > len {
            log(&format!(
                "output `{feature_name}` needs {count} elements but only {len} were provided"
            ));
            return false;
        }
        let Some(buffer) = RawBuffer::new(data, count, len) else {
            return false;
        };
        self.bind_output(
            feature_name,
            shape,
            data.cast(),
            MLMultiArrayDataType::Float16,
            BackingBuffer::F16(buffer),
        )
    }

    fn bind_output(
        &mut self,
        feature_name: String,
        shape: Vec<usize>,
        data: *mut std::ffi::c_void,
        data_type: MLMultiArrayDataType,
        buffer: BackingBuffer,
    ) -> bool {
        let noop = RcBlock::new(|_ptr: NonNull<std::ffi::c_void>| {});
        match multi_array(data, &shape, data_type, Some(&noop)) {
            Ok(array) => {
                self.outputs.insert(feature_name, (array, buffer));
                true
            }
            Err(err) => {
                log(&format!("failed to bind output `{feature_name}`: {err}"));
                // The caller still owns the buffer when we return false.
                std::mem::forget(buffer);
                false
            }
        }
    }

    pub fn predict(&mut self) -> ModelOutput {
        with_pool(|| self.predict_inner())
    }

    fn predict_inner(&mut self) -> ModelOutput {
        if self.failed_to_load {
            return ModelOutput::failed("Failed to load model; can't run predict");
        }
        let Some(model) = &self.model else {
            return ModelOutput::failed("ran predict without a model loaded into memory");
        };
        let inputs = feature_dictionary(&self.inputs);
        let provider = match unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &inputs,
            )
        } {
            Ok(provider) => provider,
            Err(err) => return ModelOutput::failed(err.localizedDescription().to_string()),
        };
        let options = unsafe { MLPredictionOptions::new() };
        let backings = backing_dictionary(&self.outputs);
        unsafe { options.setOutputBackings(&backings) };
        match unsafe {
            model.predictionFromFeatures_options_error(
                ProtocolObject::from_ref(&*provider),
                &options,
            )
        } {
            Ok(_) => {
                self.inputs.clear();
                ModelOutput {
                    inner: OutputInner::Backed(std::mem::take(&mut self.outputs)),
                    error: None,
                }
            }
            Err(err) => ModelOutput::failed(err.localizedDescription().to_string()),
        }
    }
}

/// A batch (multi-instance) variant of [`Model`].
pub struct BatchModel {
    compiled_path: Option<PathBuf>,
    model_asset: Option<Retained<MLModelAsset>>,
    model: Option<Retained<MLModel>>,
    inputs: Vec<HashMap<String, Retained<MLFeatureValue>>>,
    compute_units: MLComputeUnits,
    failed_to_load: bool,
}

// SAFETY: see `Model`; batch inputs are only mutated through `&mut self`.
unsafe impl Send for BatchModel {}

impl BatchModel {
    fn empty(compute: ComputePlatform) -> Self {
        Self {
            compiled_path: None,
            model_asset: None,
            model: None,
            inputs: Vec::new(),
            compute_units: compute.compute_units(),
            failed_to_load: false,
        }
    }

    fn from_path(path: String, compute: ComputePlatform, compiled: bool) -> Self {
        let mut model = Self::empty(compute);
        if compiled {
            model.compiled_path = Some(PathBuf::from(path));
            return model;
        }
        match compile_model(Path::new(&path)) {
            Ok(compiled_path) => model.compiled_path = Some(compiled_path),
            Err(err) => {
                log(&format!("failed to compile {path}: {err}"));
                model.failed_to_load = true;
            }
        }
        model
    }

    fn from_asset(ptr: *mut u8, len: isize, compute: ComputePlatform) -> Self {
        let len = len.max(0) as usize;
        let mut model = Self::empty(compute);
        let buffer = if len == 0 {
            Vec::new()
        } else {
            unsafe { Vec::from_raw_parts(ptr, len, len) }
        };
        let data = NSData::from_vec(buffer);
        match unsafe { MLModelAsset::modelAssetWithSpecificationData_error(&data) } {
            Ok(asset) => model.model_asset = Some(asset),
            Err(err) => {
                log(&format!(
                    "failed to read model specification: {}",
                    err.localizedDescription()
                ));
                model.failed_to_load = true;
            }
        }
        model
    }

    pub fn load(&mut self) -> bool {
        if self.failed_to_load {
            return false;
        }
        if self.model.is_some() {
            return true;
        }
        let config = model_configuration(self.compute_units);
        if let Some(path) = &self.compiled_path {
            let url = file_url(path);
            return match unsafe {
                MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
            } {
                Ok(model) => {
                    self.model = Some(model);
                    true
                }
                Err(err) => {
                    log(&format!(
                        "failed to load compiled model at {}: {}",
                        path.display(),
                        err.localizedDescription()
                    ));
                    false
                }
            };
        }
        if let Some(asset) = &self.model_asset {
            return match load_asset(asset, &config) {
                Ok(model) => {
                    self.model = Some(model);
                    true
                }
                Err(err) => {
                    log(&format!("failed to load model asset: {err}"));
                    false
                }
            };
        }
        false
    }

    pub fn unload(&mut self) -> bool {
        if self.failed_to_load {
            return false;
        }
        self.model = None;
        true
    }

    pub fn failed(&self) -> bool {
        self.failed_to_load
    }

    pub fn compiled_path(&self) -> Option<String> {
        self.compiled_path.as_ref().map(|path| {
            file_url(path)
                .absoluteString()
                .map(|url| url.to_string())
                .unwrap_or_else(|| path.to_string_lossy().into_owned())
        })
    }

    pub fn description(&self) -> ModelDescription {
        ModelDescription::new(
            self.model
                .as_ref()
                .map(|model| unsafe { model.modelDescription() }),
        )
    }

    pub fn bindInputF32(
        &mut self,
        shape: Vec<usize>,
        feature_name: String,
        data: *mut f32,
        len: usize,
        idx: isize,
    ) -> bool {
        let count: usize = shape.iter().product();
        if idx < 0 || count > len {
            log(&format!(
                "batch input `{feature_name}` has an invalid index or shape"
            ));
            return false;
        }
        let cap = len;
        let deallocator = RcBlock::new(move |ptr: NonNull<std::ffi::c_void>| unsafe {
            drop(Vec::from_raw_parts(ptr.as_ptr().cast::<f32>(), count, cap));
        });
        match multi_array(
            data.cast(),
            &shape,
            MLMultiArrayDataType::Float32,
            Some(&deallocator),
        ) {
            Ok(array) => {
                let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
                let idx = idx as usize;
                self.inputs.resize_with(idx + 1, HashMap::new);
                self.inputs[idx].insert(feature_name, value);
                true
            }
            Err(err) => {
                log(&format!(
                    "failed to bind batch input `{feature_name}`: {err}"
                ));
                false
            }
        }
    }

    pub fn predict(&mut self) -> BatchOutput {
        with_pool(|| self.predict_inner())
    }

    fn predict_inner(&mut self) -> BatchOutput {
        if self.failed_to_load {
            return BatchOutput::failed("Failed to load model; can't run predict");
        }
        let Some(model) = &self.model else {
            return BatchOutput::failed("ran predict without a model loaded into memory");
        };
        let mut providers = Vec::with_capacity(self.inputs.len());
        for inputs in &self.inputs {
            let dictionary = feature_dictionary(inputs);
            match unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dictionary,
                )
            } {
                Ok(provider) => providers.push(ProtocolObject::from_retained(provider)),
                Err(err) => {
                    return BatchOutput::failed(format!(
                        "failed to build a batch input: {}",
                        err.localizedDescription()
                    ))
                }
            }
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
        match unsafe {
            model.predictionsFromBatch_options_error(ProtocolObject::from_ref(&*batch), &options)
        } {
            Ok(result) => BatchOutput {
                provider: Some(result),
                error: None,
            },
            Err(err) => BatchOutput::failed(err.localizedDescription().to_string()),
        }
    }
}

/// Loads a model from a path. `compiled` skips compilation and treats the path
/// as an existing `.mlmodelc` (or a `file://` URL produced by [`Model::compiled_path`]).
pub fn modelWithPath(path: String, compute: ComputePlatform, compiled: bool) -> Model {
    Model::from_path(path, compute, compiled)
}

/// Loads a model from an in-memory specification. Takes ownership of
/// `ptr..ptr + len`; the caller must not touch or free it afterwards.
pub fn modelWithAssets(ptr: *mut u8, len: isize, compute: ComputePlatform) -> Model {
    Model::from_asset(ptr, len, compute)
}

/// Batch equivalent of [`modelWithPath`].
pub fn modelWithPathBatch(path: String, compute: ComputePlatform, compiled: bool) -> BatchModel {
    BatchModel::from_path(path, compute, compiled)
}

/// Batch equivalent of [`modelWithAssets`].
pub fn modelWithAssetsBatch(ptr: *mut u8, len: isize, compute: ComputePlatform) -> BatchModel {
    BatchModel::from_asset(ptr, len, compute)
}

/// Compiles a model package and atomically moves the result to `to`.
///
/// Returns `None` on success and the failure description otherwise, matching
/// the shape the old Swift function exposed.
pub fn compileModelTo(model: String, to: String) -> Option<String> {
    match compile_and_persist(Path::new(&model), Path::new(&to)) {
        Ok(()) => None,
        Err(err) => {
            log(&format!("failed to compile {model} to {to}: {err}"));
            Some(err)
        }
    }
}

fn compile_and_persist(package: &Path, destination: &Path) -> Result<(), String> {
    let start = Instant::now();
    let compiled = compile_model(package)?;
    log(&format!(
        "compiled {} in {}ms",
        package.display(),
        start.elapsed().as_millis()
    ));
    persist_compiled(&compiled, destination)?;
    log(&format!(
        "persisted compiled model to {} in {}ms total",
        destination.display(),
        start.elapsed().as_millis()
    ));
    Ok(())
}

fn compile_model(package: &Path) -> Result<PathBuf, String> {
    let url = file_url(package);
    let compiled = unsafe { compile_model_at_url(&url) }?;
    Ok(url_pathbuf(&compiled))
}

/// Moves `compiled` into place, never removing the existing model before its
/// replacement is available.
fn persist_compiled(compiled: &Path, destination: &Path) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    if !destination.exists() {
        return move_path(compiled, destination);
    }
    let backup = unique_sibling(destination, "backup");
    std::fs::rename(destination, &backup)
        .map_err(|err| format!("failed to move the existing model aside: {err}"))?;
    match move_path(compiled, destination) {
        Ok(()) => {
            let _ = remove_path(&backup);
            Ok(())
        }
        Err(err) => {
            let _ = std::fs::rename(&backup, destination);
            Err(err)
        }
    }
}

/// Prefers a rename, falling back to a recursive copy for cross-volume moves.
fn move_path(source: &Path, destination: &Path) -> Result<(), String> {
    match std::fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(rename_err) => match copy_recursive(source, destination) {
            Ok(()) => {
                let _ = remove_path(source);
                Ok(())
            }
            Err(_) => Err(format!(
                "failed to move {} to {}: {rename_err}",
                source.display(),
                destination.display()
            )),
        },
    }
}

fn copy_recursive(source: &Path, destination: &Path) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() {
        std::os::unix::fs::symlink(std::fs::read_link(source)?, destination)?;
    } else if metadata.is_dir() {
        std::fs::create_dir_all(destination)?;
        for entry in std::fs::read_dir(source)? {
            let entry = entry?;
            copy_recursive(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else {
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(source, destination)?;
    }
    Ok(())
}

fn remove_path(path: &Path) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

fn unique_sibling(path: &Path, tag: &str) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos())
        .unwrap_or_default();
    path.with_file_name(format!(".{name}.{tag}-{}-{nanos}", std::process::id()))
}

fn model_configuration(compute_units: MLComputeUnits) -> Retained<MLModelConfiguration> {
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(compute_units) };
    // CoreML's experimental MLE5 engine has historically crashed on newer
    // macOS releases; the Swift shim disabled it and we keep that behaviour.
    unsafe {
        let key = NSString::from_str("experimentalMLE5EngineUsage");
        let value = NSNumber::new_isize(1);
        let _: () = msg_send![&*config, setValue: &*value, forKey: &*key];
    }
    config
}

/// Loads an `MLModelAsset` through the asynchronous API and blocks until it
/// finishes, preserving the synchronous public interface.
fn load_asset(
    asset: &MLModelAsset,
    config: &MLModelConfiguration,
) -> Result<Retained<MLModel>, String> {
    // `MLModel` is safe to use from any thread, but `Retained` is not `Send` by
    // default. The completion handler may run on a CoreML thread, so hand the
    // model back through a wrapper that records that guarantee.
    struct SendModel(Option<Retained<MLModel>>);
    // SAFETY: see `Model`; `MLModel` predictions are documented as thread-safe.
    unsafe impl Send for SendModel {}

    type Slot = Mutex<Option<Result<SendModel, String>>>;
    let slot = Arc::new((Slot::new(None), Condvar::new()));
    let signal = Arc::clone(&slot);
    let handler = RcBlock::new(move |model: *mut MLModel, error: *mut NSError| {
        let result = if let Some(model) = unsafe { Retained::retain(model) } {
            Ok(SendModel(Some(model)))
        } else if let Some(error) = unsafe { error.as_ref() } {
            Err(error.localizedDescription().to_string())
        } else {
            Err("CoreML failed to load the model without reporting an error".to_string())
        };
        let (lock, condvar) = &*signal;
        if let Ok(mut guard) = lock.lock() {
            *guard = Some(result);
            condvar.notify_one();
        }
    });
    unsafe { MLModel::loadModelAsset_configuration_completionHandler(asset, config, &handler) };
    let (lock, condvar) = &*slot;
    let mut guard = lock
        .lock()
        .map_err(|_| "CoreML model load lock was poisoned".to_string())?;
    while guard.is_none() {
        guard = condvar
            .wait(guard)
            .map_err(|_| "CoreML model load wait was poisoned".to_string())?;
    }
    let result = guard.take().expect("guard is Some after the wait loop")?;
    result
        .0
        .ok_or_else(|| "CoreML reported a successful load without a model".to_string())
}

/// Validates an output shape and returns it alongside its element count.
fn backing_shape(shape: &[i32], feature_name: &str) -> Option<(Vec<usize>, usize)> {
    if shape.iter().any(|&dim| dim < 0) {
        log(&format!("output `{feature_name}` has a negative dimension"));
        return None;
    }
    let shape: Vec<usize> = shape.iter().map(|&dim| dim as usize).collect();
    let count = shape.iter().product();
    Some((shape, count))
}

/// Builds a shape and its row-major strides for `MLMultiArray`.
fn shape_and_strides(
    shape: &[usize],
) -> (Retained<NSArray<NSNumber>>, Retained<NSArray<NSNumber>>) {
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1].saturating_mul(shape[index + 1]);
    }
    (numbers_to_array(shape), numbers_to_array(&strides))
}

fn numbers_to_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&value| NSNumber::new_usize(value))
        .collect();
    let refs: Vec<&NSNumber> = numbers.iter().map(|number| &**number).collect();
    NSArray::from_slice(&refs)
}

fn array_to_shape(array: &NSArray<NSNumber>) -> Vec<usize> {
    (0..array.len())
        .map(|index| array.objectAtIndex(index).as_usize())
        .collect()
}

fn feature_shape(description: &MLFeatureDescription) -> Vec<usize> {
    let Some(constraint) = (unsafe { description.multiArrayConstraint() }) else {
        return Vec::new();
    };
    let shape = unsafe { constraint.shape() };
    array_to_shape(&shape)
}

/// Creates an `MLMultiArray` that borrows the caller's buffer.
fn multi_array(
    data: *mut std::ffi::c_void,
    shape: &[usize],
    data_type: MLMultiArrayDataType,
    deallocator: Option<&DynBlock<dyn Fn(NonNull<std::ffi::c_void>)>>,
) -> Result<Retained<MLMultiArray>, String> {
    let data = NonNull::new(data).ok_or_else(|| "null buffer pointer".to_string())?;
    let (shape, strides) = shape_and_strides(shape);
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
    let values: Vec<Retained<MLFeatureValue>> = features.values().cloned().collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|key| &**key).collect();
    let value_refs: Vec<&MLFeatureValue> = values.iter().map(|value| &**value).collect();
    let dictionary: Retained<NSDictionary<NSString, MLFeatureValue>> =
        NSDictionary::from_slices(&key_refs, &value_refs);
    // SAFETY: MLFeatureValue is an NSObject subclass, so the type parameter is
    // only a compile-time view of the same objects.
    unsafe { Retained::cast_unchecked(dictionary) }
}

fn backing_dictionary(
    outputs: &HashMap<String, (Retained<MLMultiArray>, BackingBuffer)>,
) -> Retained<NSDictionary<NSString, AnyObject>> {
    let keys: Vec<Retained<NSString>> = outputs
        .keys()
        .map(|name| NSString::from_str(name))
        .collect();
    let values: Vec<Retained<MLMultiArray>> =
        outputs.values().map(|(array, _)| array.clone()).collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|key| &**key).collect();
    let value_refs: Vec<&MLMultiArray> = values.iter().map(|value| &**value).collect();
    let dictionary: Retained<NSDictionary<NSString, MLMultiArray>> =
        NSDictionary::from_slices(&key_refs, &value_refs);
    // SAFETY: MLMultiArray is an NSObject subclass.
    unsafe { Retained::cast_unchecked(dictionary) }
}

fn multi_array_value(
    provider: &ProtocolObject<dyn MLFeatureProvider>,
    name: &str,
) -> Option<Retained<MLMultiArray>> {
    let key = NSString::from_str(name);
    let value = unsafe { provider.featureValueForName(&key) }?;
    unsafe { value.multiArrayValue() }
}

fn description_of<T: objc2::Message>(object: &T) -> String {
    let description: Retained<NSString> = unsafe { msg_send![object, description] };
    description.to_string()
}

fn copy_f32(array: &MLMultiArray) -> Vec<f32> {
    let count = unsafe { array.count().max(0) as usize };
    copy_bytes::<f32>(array, count)
}

fn copy_u16(array: &MLMultiArray) -> Vec<u16> {
    let count = unsafe { array.count().max(0) as usize };
    copy_bytes::<u16>(array, count)
}

fn copy_i32(array: &MLMultiArray) -> Vec<i32> {
    let count = unsafe { array.count().max(0) as usize };
    copy_bytes::<i32>(array, count)
}

/// Copies `count` elements out of an `MLMultiArray` through the framework's
/// byte handler. Unlike the deprecated `dataPointer`, this materialises arrays
/// whose backing storage is lazy (for example pixel-buffer backed outputs).
fn copy_bytes<T: Copy + 'static>(array: &MLMultiArray, count: usize) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    let output = std::cell::RefCell::new(Vec::new());
    let block = RcBlock::new(|ptr: NonNull<std::ffi::c_void>, size: isize| {
        let available = (size.max(0) as usize) / std::mem::size_of::<T>();
        let available = available.min(count);
        // SAFETY: CoreML owns a buffer valid for the duration of the handler.
        let elements = unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<T>(), available) };
        *output.borrow_mut() = elements.to_vec();
    });
    // SAFETY: the handler only reads the framework-owned buffer.
    unsafe { array.getBytesWithHandler(&block) };
    drop(block);
    output.into_inner()
}

/// Builds an `NSURL` from a filesystem path or an existing `file://` URL.
fn file_url(path: &Path) -> Retained<NSURL> {
    let raw = path.to_string_lossy();
    if raw.starts_with("file:") {
        if let Some(url) = NSURL::URLWithString(&NSString::from_str(&raw)) {
            return url;
        }
    }
    NSURL::fileURLWithPath(&NSString::from_str(&raw))
}

fn url_pathbuf(url: &NSURL) -> PathBuf {
    url.path()
        .map(|path| PathBuf::from(path.to_string()))
        .unwrap_or_else(|| {
            PathBuf::from(
                url.absoluteString()
                    .map(|url| url.to_string())
                    .unwrap_or_default(),
            )
        })
}

fn log(message: &str) {
    eprintln!("[coreml-rs] {message}");
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

    #[test]
    fn shapes_use_row_major_strides() {
        let (shape, strides) = shape_and_strides(&[2, 3, 4]);
        assert_eq!(array_to_shape(&shape), vec![2, 3, 4]);
        assert_eq!(array_to_shape(&strides), vec![12, 4, 1]);
    }

    #[test]
    fn raw_buffer_preserves_length_and_capacity() {
        let mut data = Vec::with_capacity(8);
        data.extend_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let capacity = data.capacity();
        let pointer = data.as_mut_ptr();
        std::mem::forget(data);

        let buffer = RawBuffer::new(pointer, 4, capacity).expect("non-null buffer");
        let restored = buffer.into_vec();
        assert_eq!(restored, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(restored.capacity(), capacity);
    }

    #[test]
    fn backing_shape_rejects_negative_dimensions() {
        assert!(backing_shape(&[1, -4], "out").is_none());
        let (shape, count) = backing_shape(&[1, 4], "out").expect("valid shape");
        assert_eq!(shape, vec![1, 4]);
        assert_eq!(count, 4);
    }
}
