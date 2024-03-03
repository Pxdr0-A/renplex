use std::marker::PhantomData;

use crate::input::{IOShape, IOType};
use crate::act::ActFunc;
use super::{LayerForwardError, LayerLike, LayerInitError};

#[derive(Debug)]
pub struct Flatten<T> {
  _hidden: PhantomData<T> 
}

impl<T> LayerLike<T> for Flatten<T> {
  fn is_empty(&self) -> bool {
    unimplemented!()
  }

  fn get_input_shape(&self) -> IOShape {
    unimplemented!()
  }

  fn get_output_shape(&self) -> IOShape {
    unimplemented!()
  }

  fn new(func: ActFunc) -> Self {
    unimplemented!()
  }

  fn init(input_shape: IOShape, units: usize, func: crate::act::ActFunc, method: super::InitMethod, seed: &mut u128) -> Result<Self, LayerInitError> {
    unimplemented!()
  }

  fn init_mut(&mut self, input_shape: IOShape, units: usize, method: super::InitMethod, seed: &mut u128) -> Result<(), LayerInitError> {
    unimplemented!()
  }

  fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    unimplemented!()
  }

  fn forward(&self, _input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    unimplemented!()
  }

  fn wrap(self) -> super::Layer<T> {
    unimplemented!()
  }
}