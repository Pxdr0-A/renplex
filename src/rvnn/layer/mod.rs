use crate::input::{IOShape, IOType}; 
use crate::math::{BasicOperations, Real};
use crate::act::ActFunc;
use crate::init::InitMethod;
use crate::err::{LayerInitError, LayerForwardError};

use self::dense::DenseLayer;

pub mod dense;

pub trait LayerLike<T> where Self: Sized {
  fn is_empty(&self) -> bool;

  fn get_input_shape(&self) -> IOShape;

  fn get_output_shape(&self) -> IOShape;

  /// Creates an empty layer
  fn new(func: ActFunc) -> Self;

  fn init(input_shape: IOShape, units: usize, func: ActFunc, method: InitMethod, seed: &mut u128) -> Result<Self, LayerInitError>;

  fn init_mut(&mut self, input_shape: IOShape, units: usize, method: InitMethod, seed: &mut u128) -> Result<(), LayerInitError>;

  fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError>;

  fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError>;

  fn wrap(self) -> Layer<T>;
}

/// Just a general interface for a [`Network<T>`] that allows for a static personaliztion.
#[derive(Debug)]
pub enum Layer<T> {
  Dense(DenseLayer<T>)
}

impl<T: Real + BasicOperations<T>> Layer<T> {
  pub fn is_empty(&self) -> bool {
    match self {
      Layer::Dense(l) => { l.is_empty() }
    }
  }

  pub fn get_input_shape(&self) -> IOShape {
    match self {
      Layer::Dense(l) => { l.get_input_shape() }
    }
  }

  pub fn get_output_shape(&self) -> IOShape {
    match self {
      Layer::Dense(l) => { l.get_output_shape() }
    }
  }

  pub fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      Layer::Dense(l) => { l.trigger(input_type) }
    }  
  }

  pub fn foward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      Layer::Dense(l) => { l.forward(input_type) }
    }
  }
}

