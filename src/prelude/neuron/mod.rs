pub mod activation;

use activation::ActivationFunction;

pub struct Neuron<W> {
  pub id: usize,
  pub weights: Vec<W>,
  pub bias: W,
  pub activation: ActivationFunction
}