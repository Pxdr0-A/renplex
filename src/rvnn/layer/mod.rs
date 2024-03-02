use crate::input::{InputShape, InputType, OutputShape, OutputType}; 
use crate::math::{BasicOperations, Real};
use crate::act::ActFunc;

use self::{dense::DenseLayer, flatten::Flatten, conv::ConvLayer};

pub mod dense;
pub mod flatten;
pub mod conv;

#[derive(Debug)]
pub enum LayerForwardError {
  InvalidInput,
  UnimplementedLayer
}

pub enum InitMethod {
  Random,
  Distribution
}

#[derive(Debug)]
pub enum LayerInitError {
  InvalidInputShape,
  UnimplementedInitMethod,
  AlreadyInitialized
}

pub trait LayerLike<T> where Self: Sized {
  fn is_empty(&self) -> bool;

  fn get_input_shape(&self) -> InputShape;

  fn get_output_shape(&self) -> OutputShape;

  /// Initializes an empty layer
  fn new(func: ActFunc) -> Self;

  fn init(input_shape: InputShape, units: usize, func: ActFunc, method: InitMethod, seed: &mut u128) -> Result<Self, LayerInitError>;

  fn init_mut(&mut self, input_shape: InputShape, units: usize, method: InitMethod, seed: &mut u128) -> Result<(), LayerInitError>;

  fn forward(&self, input: InputType<T>) -> Result<OutputType<T>, LayerForwardError>;

  fn wrap(self) -> Layer<T>;
}

/// Just a general interface for a [`Network<T>`] that allows for a static personaliztion.
pub enum Layer<T> {
  Dense(DenseLayer<T>),
  Flatten(Flatten<T>),
  Convolutional(ConvLayer<T>)
}

impl<T: Real + BasicOperations<T>> Layer<T> {
  pub fn is_empty(&self) -> bool {
    match self {
      Layer::Dense(l) => { l.is_empty() },
      Layer::Flatten(l) => { l.is_empty() },
      Layer::Convolutional(l) => { l.is_empty() }
    }
  }

  pub fn get_input_shape(&self) -> InputShape {
    match self {
      Layer::Dense(l) => { l.get_input_shape() },
      Layer::Flatten(l) => { l.get_input_shape() },
      Layer::Convolutional(l) => { l.get_input_shape() }
    }
  }

  pub fn get_output_shape(&self) -> OutputShape {
    match self {
      Layer::Dense(l) => { l.get_output_shape() },
      Layer::Flatten(l) => { l.get_output_shape() },
      Layer::Convolutional(l) => { l.get_output_shape() }
    }
  }

  pub fn foward(&self, input_type: InputType<T>) -> Result<OutputType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      Layer::Dense(l) => { l.forward(input_type) },
      _ => { Err(LayerForwardError::UnimplementedLayer) }
    }
  }
}

