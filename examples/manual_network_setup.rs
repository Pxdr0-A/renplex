use std::collections::HashMap;

use num_complex::Complex32;
use renplex_core::activations::Tanh;
use renplex_core::init::UniformXG;
use renplex_core::modules::{Module, StaticLinearLayer, StaticLinearModule};
use renplex_core::optimization::{Loss, Optimizer, CSGD, MSE};
use renplex_core::tensor::{StaticTensor, Tensor};

fn main() {
    type Precision = Complex32;
    type Initializer = UniformXG;

    const LENW1: usize = 64 * 32;
    let mut layer1 =
        StaticLinearLayer::<Precision, Initializer, Tanh, LENW1, 64, 64>::init(&HashMap::new());

    const LENW2: usize = 128 * 64;
    let mut layer2 =
        StaticLinearModule::<Precision, Initializer, LENW2, 128, 128>::init(&HashMap::new());

    // forward pass
    let input = StaticTensor::<Precision, 32>::new(Vec::new());
    let out1 = layer1.forward(&input);
    let out2 = layer2.forward(&out1);

    let target = StaticTensor::<Precision, 128>::new(Vec::new());
    let _loss_vals = MSE::forward(&out2, &target);

    // backward pass
    let mut opt = CSGD::<Precision>::init(&HashMap::new());
    let loss_gradx = MSE::backward(&out2, &target);

    let layer2_gradx = layer2.backward(&out1, &loss_gradx);
    let (layer2_gradw, layer2_gradb) = layer2.gradient(&out1, &loss_gradx);
    let (weights, bias) = layer2.get_mut_params();
    opt.step(weights, bias, &layer2_gradw, &layer2_gradb);

    let _layer1_gradx = layer1.backward(&input, &layer2_gradx);
    let (layer1_gradw, layer1_gradb) = layer1.gradient(&out1, &layer2_gradx);
    let (weights, bias) = layer1.get_mut_params();
    opt.step(weights, bias, &layer1_gradw, &layer1_gradb);
}
