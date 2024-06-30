use crate::input::{IOShape, IOType};
use crate::math::{BasicOperations, Complex};
use crate::err::{LayerInitError, LayerForwardError};
use crate::err::GradientError;

use self::dense::DenseCLayer;
use self::conv::ConvCLayer;
use self::reduce::Reduce;
use self::flatten::Flatten;

pub mod dense;
pub mod conv;
pub mod reduce;
pub mod flatten;

pub type ComplexDerivatives<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);

/// Just a general interface for a [`Network<T>`] that allows for a static personaliztion.
#[derive(Debug)]
pub enum CLayer<T> {
  Dense(DenseCLayer<T>),
  Convolutional(ConvCLayer<T>),
  Reduce(Reduce<T>),
  Flatten(Flatten)
}

unsafe impl<T: Send> Send for CLayer<T> {}
unsafe impl<T: Sync> Sync for CLayer<T> {}

impl<T: Complex + BasicOperations<T>> CLayer<T> {
  /// Method that checks if the layer was already initialized.
  /// Its logic has room for improvement.
  pub fn is_empty(&self) -> bool {
    match self {
      CLayer::Dense(l) => { l.is_empty() },
      CLayer::Convolutional(l) => { l.is_empty() },
      CLayer::Reduce(l) => { l.is_empty() },
      CLayer::Flatten(l) => { l.is_empty() }
    }
  }

  pub fn propagates(&self) -> bool {
    match self {
      CLayer::Dense(l) => { l.propagates() },
      CLayer::Convolutional(l) => { l.propagates() },
      CLayer::Reduce(l) => { l.propagates() },
      CLayer::Flatten(l) => { l.propagates() }
    }
  }

  pub fn get_input_shape(&self) -> IOShape {
    match self {
      CLayer::Dense(l) => { l.get_input_shape() },
      CLayer::Convolutional(l) => { l.get_input_shape() },
      CLayer::Reduce(l) => { l.get_input_shape() },
      CLayer::Flatten(l) => { l.get_input_shape() }
    }
  }

  pub fn get_output_shape(&self) -> IOShape {
    match self {
      CLayer::Dense(l) => { l.get_output_shape() },
      CLayer::Convolutional(l) => { l.get_output_shape() },
      CLayer::Reduce(l) => { l.get_output_shape() },
      CLayer::Flatten(l) =>{ l.get_output_shape() }
    }
  }

  pub fn params_len(&self) -> (usize, usize) {
    match self {
      CLayer::Dense(l) => { l.params_len() },
      CLayer::Convolutional(l) => { l.params_len() },
      CLayer::Reduce(l) => { l.params_len() },
      CLayer::Flatten(_l) => { (0, 0) }
    }
  }

  pub fn foward(&self, input_type: &IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    /* deconstruct what type of layer it is */
    match self {
      CLayer::Dense(l) => { l.forward(input_type) },
      CLayer::Convolutional(l) => { l.forward(input_type) },
      CLayer::Reduce(l) => { l.foward(input_type) },
      CLayer::Flatten(l) => { l.foward(input_type) }
    }
  }

  pub fn compute_derivatives(&self, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    match self {
      CLayer::Dense(l) => { l.compute_derivatives(previous_act, dlda, dlda_conj) },
      CLayer::Convolutional(l) => { l.compute_derivatives(previous_act, dlda, dlda_conj) },
      CLayer::Reduce(l) => { l.compute_derivatives(previous_act, dlda, dlda_conj) },
      CLayer::Flatten(_l) => { panic!("Flatten layer has no derivatives, since it is non-trainable.") }
    }
  }

  pub fn neg_conj_adjustment(&mut self, dldw: Vec<T>, dldb: Vec<T>) -> Result<(), GradientError> {
    match self {
      CLayer::Dense(l) => { l.neg_conj_adjustment(dldw, dldb) },
      CLayer::Convolutional(l) => { l.neg_conj_adjustment(dldw, dldb) },
      CLayer::Reduce(l) => { l.neg_conj_adjustment(dldw, dldb) },
      CLayer::Flatten(_l) => { panic!("Flaatten layer has no parameters to be adjusted.") }
    }
  }
}
