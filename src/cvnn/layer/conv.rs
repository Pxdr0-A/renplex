use crate::{act::ComplexActFunc, math::{matrix::Matrix, BasicOperations, Complex}};

#[derive(Debug)]
pub struct ConvCLayer<T> {
  weights: Vec<Matrix<T>>,
  bias: Matrix<T>,
  func: ComplexActFunc
}

impl<T: Complex + BasicOperations<T>> ConvCLayer<T> {
  pub fn is_empty(&self) {
    unimplemented!()
  }
}