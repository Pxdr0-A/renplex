use crate::{act::ComplexActFunc, math::matrix::Matrix};

#[derive(Debug)]
pub struct ConvCLayer<T> {
  weights: Vec<Matrix<T>>,
  bias: Matrix<T>,
  func: ComplexActFunc
}