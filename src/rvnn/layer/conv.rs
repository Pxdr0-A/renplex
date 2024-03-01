use crate::math::matrix::Matrix;
use crate::input::{InputType, OutputType};

use super::{LayerError, LayerLike};

pub struct ConvLayer<T> {
  _weights: Matrix<T>,
  _biases: Vec<T>
}

impl<T> LayerLike<T> for ConvLayer<T> {

  fn forward(&self, _input_type: InputType<T>) -> Result<OutputType<T>, LayerError> {
    unimplemented!()
  }
}