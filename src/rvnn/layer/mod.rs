use crate::{input::{InputType, OutputType}, math::{BasicOperations, Real}};

use self::{dense::DenseLayer, flatten::Flatten, conv::ConvLayer};

pub mod dense;
pub mod flatten;
pub mod conv;

#[derive(Debug)]
pub enum LayerError {
  InvalidInput,
  UnimplementedLayer
}

pub trait LayerLike<T> {

  fn forward(&self, input: InputType<T>) -> Result<OutputType<T>, LayerError>;
}

pub enum Layer<T> {
  Dense(DenseLayer<T>),
  Flatten(Flatten<T>),
  Convolutional(ConvLayer<T>)
}

impl<T: Real + BasicOperations<T>> Layer<T> {
  pub fn foward(&self, input_type: InputType<T>) -> Result<OutputType<T>, LayerError> {
    /* deconstruct what type of layer it is */
    match self {
      Layer::Dense(l) => { l.forward(input_type) },
      _ => { Err(LayerError::UnimplementedLayer) }
    }
  }
}

