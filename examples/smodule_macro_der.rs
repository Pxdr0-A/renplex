use renplex_core::Module;
use renplex_core::{module::Linear, tensor::StaticTensor};
use renplex_derive::SuperModuleMacro;
use std::collections::HashMap;

fn main() {
    type Prec = f32;
    type Module1 = Linear<
        StaticTensor<Prec, 16>,
        StaticTensor<Prec, 4>,
        StaticTensor<Prec, 32>,
        StaticTensor<Prec, 16>,
    >;
    type Module2 = Linear<
        StaticTensor<Prec, 16>,
        StaticTensor<Prec, 4>,
        StaticTensor<Prec, 16>,
        StaticTensor<Prec, 64>,
    >;

    #[derive(Debug, SuperModuleMacro)]
    pub struct Network {
        pub _module1: Module1,
        pub _module2: Module2,
    }

    let network = Network::init(HashMap::new(), HashMap::new());

    println!("{:?}", network);
}
