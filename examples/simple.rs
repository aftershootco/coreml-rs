use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::{Array, Array4};

pub fn main() {
    let path = std::env::args_os()
        .nth(1)
        .expect("pass a compiled .mlmodelc directory");

    let model_options = CoreMLModelOptions {
        compute_platform: ComputePlatform::CpuAndANE,
    };

    let mut model = CoreMLModelWithState::new_compiled(path, model_options)
        .load()
        .unwrap();

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap();
    for (_out, v) in v.outputs {
        let _output: Array<f32, _> = v.extract_to_tensor().expect("expected f32 output");
    }

    // Use output as needed
}
