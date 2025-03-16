use num_complex::Complex32;
use renplex_core::{
    activations::Tanh,
    init::UniformXG,
    modules::{Module, StaticLinearLayer, StaticLinearModule},
    optimization::{Loss, Optimizer, CSGD, MSE},
    tensor::Tensor,
    InitArgs, Network,
};

fn main() {
    type Precision = Complex32;
    type Initializer = UniformXG;
    type LossFunc = MSE;
    type Opt = CSGD<Precision>;

    const LENW1: usize = 64 * 32;
    type Layer1 = StaticLinearLayer<Precision, Initializer, Tanh, LENW1, 64, 64>;

    const LENW2: usize = 128 * 64;
    type Layer2 = StaticLinearModule<Precision, Initializer, LENW2, 128, 128>;

    struct MyNetwork(Layer1, Layer2);

    impl Network for MyNetwork {
        type Precision = Precision;
        type Output = <Layer2 as Module>::Output;
        type LossFunc = LossFunc;
        type Opt = Opt;

        fn init(args: &InitArgs) -> Self {
            let layer1 = Layer1::init(args);
            let layer2 = Layer2::init(args);
            MyNetwork(layer1, layer2)
        }

        fn predict<T: Tensor<Self::Precision>>(&self, input: &T) -> Self::Output {
            let MyNetwork(layer1, layer2) = self;
            let out1 = layer1.forward(input);
            layer2.forward(&out1)
        }

        fn train<T: Tensor<Self::Precision>>(
            &mut self,
            input: &T,
            target: &Self::Output,
            opt: &mut Self::Opt,
        ) {
            // forward pass
            let MyNetwork(layer1, layer2) = self;
            let out1 = layer1.forward(input);
            let out2 = layer2.forward(&out1);
            let _loss_vals = Self::LossFunc::forward(&out2, target);

            // backward pass
            let loss_gradx = Self::LossFunc::backward(&out2, target);

            let layer2_gradx = layer2.backward(&out1, &loss_gradx);
            let (layer2_gradw, layer2_gradb) = layer2.gradient(&out1, &loss_gradx);
            let (weights, bias) = layer2.get_mut_params();
            opt.step(weights, bias, &layer2_gradw, &layer2_gradb);

            let _layer1_gradx = layer1.backward(input, &layer2_gradx);
            let (layer1_gradw, layer1_gradb) = layer1.gradient(&out1, &layer2_gradx);
            let (weights, bias) = layer1.get_mut_params();
            opt.step(weights, bias, &layer1_gradw, &layer1_gradb);
        }
    }
}
