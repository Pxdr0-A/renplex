use std::marker::PhantomData;

use crate::input::{InputShape, InputType, OutputShape, OutputType};
use crate::act::ActFunc;
use super::{LayerForwardError, LayerLike, LayerInitError};

pub struct Flatten<T> {
  _hidden: PhantomData<T> 
}

impl<T> LayerLike<T> for Flatten<T> {
  fn is_empty(&self) -> bool {
    unimplemented!()
  }

  fn get_input_shape(&self) -> InputShape {
    unimplemented!()
  }

  fn get_output_shape(&self) -> OutputShape {
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