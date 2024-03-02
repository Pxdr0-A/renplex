use crate::math::matrix::Matrix;
use crate::input::{InputShape, InputType, OutputType};
use crate::act::ActFunc;
use super::{LayerInitError, LayerForwardError, LayerLike};

pub struct ConvLayer<T> {
  _weights: Matrix<T>,
  _biases: Vec<T>
}

impl<T> LayerLike<T> for ConvLayer<T> {
  fn is_empty(&self) -> bool {
    unimplemented!()
  }
  
  fn get_input_shape(&self) -> crate::input::InputShape {
    unimplemented!()
  }

  fn get_output_shape(&self) -> crate::input::OutputShape {
    unimplemented!()
  }

  fn new(func: ActFunc) -> Self {
    unimplemented!()
  }

  fn init(input_shape: InputShape, units: usize, func: crate::act::ActFunc, method: super::InitMethod, seed: &mut u128) -> Result<Self, LayerInitError> {
    unimplemented!()
  }

  fn init_mut(&mut self, input_shape: InputShape, units: usize, method: super::InitMethod, seed: &mut u128) -> Result<(), LayerInitError> {
    unimplemented!()
  }

  fn forward(&self, _input_type: InputType<T>) -> Result<OutputType<T>, LayerForwardError> {
    unimplemented!()
  }

  fn wrap(self) -> super::Layer<T> {
    unimplemented!()
  }
}