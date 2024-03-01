use std::marker::PhantomData;

use crate::input::{InputType, OutputType};

use super::{LayerError, LayerLike, InitError};

pub struct Flatten<T> {
  _hidden: PhantomData<T> 
}

impl<T> LayerLike<T> for Flatten<T> {
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