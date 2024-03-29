use crate::input::{IOShape, IOType}; 
use crate::math::matrix::Matrix;
use crate::math::{BasicOperations, Complex};
use crate::act::ComplexActFunc;
use crate::init::InitMethod;
use crate::err::{LayerInitError, LayerForwardError};
use crate::err::GradientError;

use self::dense::DenseCLayer;

pub mod dense;

pub trait CLayerLike<T> where Self: Sized {
  fn is_empty(&self) -> bool;

  fn is_trainable(&self) -> bool;

  fn get_input_shape(&self) -> IOShape;

  fn get_output_shape(&self) -> IOShape;

  /// Creates an empty layer
  fn new(func: ComplexActFunc) -> Self;

  fn init(input_shape: IOShape, units: usize, func: ComplexActFunc, method: InitMethod, seed: &mut u128) -> Result<Self, LayerInitError>;

  fn init_mut(&mut self, input_shape: IOShape, units: usize, method: InitMethod, seed: &mut u128) -> Result<(), LayerInitError>;

  fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError>;

  fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError>;

  fn compute_derivatives(&self, is_input: bool, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<(Matrix<T>, Matrix<T>, Vec<T>, Vec<T>), GradientError>;

  fn gradient_adjustment(&mut self, dldw: Matrix<T>, dldb: Matrix<T>) -> Result<(), GradientError>;

  fn wrap(self) -> CLayer<T>;
}


/// Just a general interface for a [`Network<T>`] that allows for a static personaliztion.
#[derive(Debug)]
pub enum CLayer<T> {
  Dense(DenseCLayer<T>)
}


impl<T: Complex + BasicOperations<T>> CLayer<T> {
  pub fn is_empty(&self) -> bool {
    match self {
      CLayer::Dense(l) => { l.is_empty() }
    }
  }

  pub fn is_trainable(&self) -> bool {
    match self {
      CLayer::Dense(l) => { l.is_trainable() }
    }
  }

  pub fn get_input_shape(&self) -> IOShape {
    match self {
      CLayer::Dense(l) => { l.get_input_shape() }
    }
  }

  pub fn get_output_shape(&self) -> IOShape {
    match self {
      CLayer::Dense(l) => { l.get_output_shape() }
    }
  }

  pub fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      CLayer::Dense(l) => { l.trigger(input_type) }
    }  
  }

  pub fn foward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      CLayer::Dense(l) => { l.forward(input_type) }
    }
  }

  pub fn compute_derivatives(&self, is_input: bool, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<(Matrix<T>, Matrix<T>, Vec<T>, Vec<T>), GradientError> {
    match self {
      CLayer::Dense(l) => { l.compute_derivatives(is_input, previous_act, dlda, dlda_conj) }
    }
  }

  pub fn gradient_adjustment(&mut self, dldw: Matrix<T>, dldb: Matrix<T>) -> Result<(), GradientError> {
    match self {
      CLayer::Dense(l) => { l.gradient_adjustment(dldw, dldb) }
    }
  }
}

