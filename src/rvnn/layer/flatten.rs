use std::marker::PhantomData;

use crate::input::{InputType, OutputType};

use super::{LayerError, LayerLike};

pub struct Flatten<T> {
  _hidden: PhantomData<T> 
}

impl<T> LayerLike<T> for Flatten<T> {

  fn forward(&self, _input_type: InputType<T>) -> Result<OutputType<T>, LayerError> {
    unimplemented!()
  }
}