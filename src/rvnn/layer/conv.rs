use crate::math::matrix::Matrix;
use crate::input::{InputType, OutputType};

use super::{InitError, LayerError, LayerLike};

pub struct ConvLayer<T> {
  _weights: Matrix<T>,
  _biases: Vec<T>
}

impl<T> LayerLike<T> for ConvLayer<T> {
  fn init(inputs: usize, units: usize, func: crate::act::ActFunc, method: super::InitMethod, seed: &mut u128) -> Result<Self, InitError> {
    unimplemented!()
  }

  fn forward(&self, _input_type: InputType<T>) -> Result<OutputType<T>, LayerError> {
    unimplemented!()
  }

  fn wrap(self) -> super::Layer<T> {
    unimplemented!()
  }
}