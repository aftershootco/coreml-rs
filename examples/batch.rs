use coreml_rs::{mlbatchmodel::CoreMLBatchModelWithState, ComputePlatform, CoreMLModelOptions};
use ndarray::{Array, Array4};

pub fn main() {
    let path = std::env::args_os()
        .nth(1)
        .expect("pass a compiled .mlmodelc directory");

    let model_options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
    };

    let mut model = CoreMLBatchModelWithState::new_compiled(path, model_options)
        .load()
        .unwrap();

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    // Add multiple inputs for batch processing
    for i in 0..10 {
        model
            .add_input("image", input.clone().into_dyn(), i)
            .unwrap();
    }

    let output = model.predict().unwrap();
    for output in output.outputs {
        for (_out, v) in output {
            let _output: Array<f32, _> = v.extract_to_tensor().expect("expected f32 output");
        }
    }
}
