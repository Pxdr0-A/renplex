pub struct Neuron<W> {
  pub id: usize,
  pub weights: Vec<W>,
  pub bias: W,
  pub activation: ActivationFunction
}

pub enum ActivationFunction {
  SIGMOID,
  TANH,
  RELU
}