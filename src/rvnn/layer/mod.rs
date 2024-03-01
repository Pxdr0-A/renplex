use crate::input::{InputType, OutputType}; 
use crate::math::{BasicOperations, Real};
use crate::act::ActFunc;

use self::{dense::DenseLayer, flatten::Flatten, conv::ConvLayer};

pub mod dense;
pub mod flatten;
pub mod conv;

#[derive(Debug)]
pub enum LayerError {
  InvalidInput,
  UnimplementedLayer
}

pub enum InitMethod {
  Random,
  Distribution
}

pub enum InitError {
  InvalidMethod
}

pub trait LayerLike<T> where Self: Sized {

  fn init(inputs: usize, units: usize, func: ActFunc, method: InitMethod, seed: &mut u128) -> Result<Self, InitError>;

  fn forward(&self, input: InputType<T>) -> Result<OutputType<T>, LayerError>;

  fn wrap(self) -> Layer<T>;
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

