use std::ops::{AddAssign, SubAssign, Add, Mul};
use std::default::Default;

use crate::cvnn::Criteria;
use crate::err::PredicionError;
use crate::init::PredictModel;

use self::cfloat::{Cf32, Cf64};
use self::random::{lcgf32, lcgf64};

pub mod matrix;
pub mod random;
pub mod cfloat;

pub trait BasicOperations<T>: AddAssign + SubAssign + Add<Output=T> + Mul<Output=T> + Default + Copy {}

impl<T, U> BasicOperations<T> for U where U: AddAssign + SubAssign + Add<Output=T> + Mul<Output=T> + Default + Copy {}

/// Trait containing utilities for RVNNs
pub trait Real 
  where Self: Sized {

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn gen_pred(size: usize, index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;

  fn sigmoid(self) -> Self;
}

impl Real for f32 {
  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f32;

    2.0 * float_scale * lcgf32(seed) - float_scale
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size > critical_index { return Err(PredicionError::CriticalIndexOverflow) }

    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![0.0; size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }

  fn sigmoid(self) -> Self {
    self.exp() / (1.0 + self.exp())
  }
}
impl Real for f64 {
  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f64;

    2.0 * float_scale * lcgf64(seed) - float_scale
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size > critical_index { return Err(PredicionError::CriticalIndexOverflow) }
    
    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![0.0; size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }

  fn sigmoid(self) -> Self {
    self.exp() / (1.0 + self.exp())
  }
}

pub trait Complex where Self: Sized {
  type Precision: Real + BasicOperations<Self::Precision>;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn rit_sigmoid(self) -> Self;

  fn cost(pred: &[Self], target: &[Self::Precision], criteria: &Criteria) -> Vec<Self::Precision>;
}

impl Complex for Cf32 {
  type Precision = f32;

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f32;

    Cf32 { 
      x: 2.0 * float_scale * lcgf32(seed) - float_scale,
      y: 2.0 * float_scale * lcgf32(seed) - float_scale
    }
  }

  fn rit_sigmoid(self) -> Self {
    Cf32 {
      x: self.x.exp() / (1.0 + self.x.exp()),
      y: self.y.exp() / (1.0 + self.y.exp())
    }
  }

  fn cost(prediction: &[Self], target: &[Self::Precision], criteria: &Criteria) -> Vec<Self::Precision> {    
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match criteria {
          Criteria::Norm => { (pred.norm() - targ).powi(2) },
          Criteria::Phase => { (pred.phase() - targ).powi(2) },
          Criteria::Real => { (pred.re() - targ).powi(2) },
          Criteria::Imaginary => { (pred.im() - targ).powi(2) }
        }
      })
      .collect()
  }
}

impl Complex for Cf64 {
  type Precision = f64;

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f64;

    Cf64 { 
      x: 2.0 * float_scale * lcgf64(seed) - float_scale,
      y: 2.0 * float_scale * lcgf64(seed) - float_scale
    }
  }

  fn rit_sigmoid(self) -> Self {
    Cf64 {
      x: self.x.exp() / (1.0 + self.x.exp()),
      y: self.y.exp() / (1.0 + self.y.exp())
    }
  }

  fn cost(prediction: &[Self], target: &[Self::Precision], criteria: &Criteria) -> Vec<Self::Precision> {
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match criteria {
          Criteria::Norm => { (pred.norm() - targ).powi(2) },
          Criteria::Phase => { (pred.phase() - targ).powi(2) },
          Criteria::Real => { (pred.re() - targ).powi(2) },
          Criteria::Imaginary => { (pred.im() - targ).powi(2) }
        }
      })
      .collect()
  }
}