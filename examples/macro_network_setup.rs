use std::collections::HashMap;

use num_complex::Complex32;
use renplex_core::activations::Tanh;
use renplex_core::init::UniformXG;
use renplex_core::modules::{Module, StaticLinearLayer, StaticLinearModule};
use renplex_core::optimization::Optimizer;
use renplex_core::optimization::{CSGD, MSE};
use renplex_core::tensor::{StaticTensor, Tensor};
use renplex_core::InitArgs;
use renplex_core::Network;
use renplex_derive::ImplNetwork;

fn main() {
    // needs to be later detected by the macro
    type Precision = Complex32;
    type MyLossFunc = MSE;
    type MyOptimizer = CSGD<Precision>;

    type Initializer = UniformXG;

    const LENW1: usize = 64 * 32;
    type Layer1 = StaticLinearLayer<Precision, Initializer, Tanh, LENW1, 64, 64>;

    const LENW2: usize = 128 * 64;
    type Layer2 = StaticLinearModule<Precision, Initializer, LENW2, 128, 128>;

    #[derive(ImplNetwork)]
    pub struct MyNetwork {
        _layer1: Layer1,
        _layer2: Layer2,
    }

    let mut my_network = MyNetwork::init(&HashMap::new());

    let input = StaticTensor::<Precision, 32>::new(Vec::new());
    let _prediction = my_network.predict(&input);

    let target = <MyNetwork as Network>::Output::new(Vec::new());
    let mut opt = <MyNetwork as Network>::Opt::init(&HashMap::new());
    my_network.train(&input, &target, &mut opt);
}
