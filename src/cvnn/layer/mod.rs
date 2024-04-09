use crate::input::{IOShape, IOType};
use crate::math::matrix::Matrix;
use crate::math::{BasicOperations, Complex};
use crate::act::ComplexActFunc;
use crate::init::InitMethod;
use crate::err::{LayerInitError, LayerForwardError};
use crate::err::GradientError;

use self::dense::DenseCLayer;
use self::conv::ConvCLayer;

pub mod dense;
pub mod conv;


/// Just a general interface for a [`Network<T>`] that allows for a static personaliztion.
#[derive(Debug)]
pub enum CLayer<T> {
  Dense(DenseCLayer<T>),
  Convolutional(ConvCLayer<T>)
}

impl<T: Complex + BasicOperations<T>> CLayer<T> {
  /// Method that checks if the layer was already initialized.
  /// Its logic has room for improvement.
  pub fn is_empty(&self) -> bool {
    match self {
      CLayer::Dense(l) => { l.is_empty() },
      CLayer::Convolutional(l) => { unimplemented!() }
    }
  }

  pub fn is_trainable(&self) -> bool {
    match self {
      CLayer::Dense(l) => { l.is_trainable() },
      CLayer::Convolutional(l) => { unimplemented!() }
    }
  }

  pub fn get_input_shape(&self) -> IOShape {
    match self {
      CLayer::Dense(l) => { l.get_input_shape() },
      CLayer::Convolutional(l) => { unimplemented!() }
    }
  }

  pub fn get_output_shape(&self) -> IOShape {
    match self {
      CLayer::Dense(l) => { l.get_output_shape() },
      CLayer::Convolutional(l) => { unimplemented!() }
    }
  }

  pub fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      CLayer::Dense(l) => { l.trigger(input_type) },
      CLayer::Convolutional(l) => { unimplemented!() }
    }  
  }

  pub fn foward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      CLayer::Dense(l) => { l.forward(input_type) },
      CLayer::Convolutional(l) => { unimplemented!() }
    }
  }

  pub fn compute_derivatives(&self, is_input: bool, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<(Matrix<T>, Matrix<T>, Vec<T>, Vec<T>), GradientError> {
    match self {
      CLayer::Dense(l) => { l.compute_derivatives(is_input, previous_act, dlda, dlda_conj) },
      CLayer::Convolutional(l) => { unimplemented!() }
    }
  }

  pub fn gradient_adjustment(&mut self, dldw: Matrix<T>, dldb: Matrix<T>) -> Result<(), GradientError> {
    match self {
      CLayer::Dense(l) => { l.gradient_adjustment(dldw, dldb) },
      CLayer::Convolutional(l) => { unimplemented!() }
    }
  }
}
