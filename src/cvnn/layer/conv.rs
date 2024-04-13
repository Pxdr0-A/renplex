use crate::{act::ComplexActFunc, err::LayerForwardError, input::{IOShape, IOType}, math::{matrix::Matrix, BasicOperations, Complex}};

#[derive(Debug)]
pub struct ConvCLayer<T> {
  weights: Vec<Matrix<T>>,
  biases: Matrix<T>,
  func: ComplexActFunc
}

impl<T: Complex + BasicOperations<T>> ConvCLayer<T> {
  pub fn is_empty(&self) -> bool {
    if self.weights.len() == 0 { true }
    else { false }
  }

  pub fn is_trainable(&self) -> bool {
    true
  }

  pub fn params_len(&self) -> (usize, usize) {
    let mut params: usize = 0;
    for kernel in self.weights.iter() {
      let kernel_shape = kernel.get_shape();
      params += kernel_shape[0] * kernel_shape[1];
    }

    let bias_shape = self.biases.get_shape();

    (params, bias_shape[0]*bias_shape[1])
  }

  pub fn get_input_shape(&self) -> IOShape {
    let bias_shape = self.biases.get_shape();
    IOShape::Matrix([bias_shape[0], bias_shape[1]])
  }

  pub fn get_output_shape(&self) -> IOShape {
    let bias_shape = self.biases.get_shape();
    IOShape::Matrix([bias_shape[0], bias_shape[1]])
  }

  pub fn init() {
    unimplemented!()
  }

  pub fn init_mut() {
    unimplemented!()
  }

  pub fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    unimplemented!()
  }

  pub fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    unimplemented!()
  }
}