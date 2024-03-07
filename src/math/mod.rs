use std::fmt::Debug;
use std::ops::{AddAssign, SubAssign, Add, Mul};
use std::default::Default;

use crate::rvnn::CostModel;
use crate::cvnn::{ComplexCostModel, Criteria};
use crate::err::PredicionError;
use crate::init::PredictModel;

use self::cfloat::{Cf32, Cf64};
use self::random::{lcgf32, lcgf64};

pub mod matrix;
pub mod random;
pub mod cfloat;

const SIGMOID_THRESHOLD_F32: f32 = 15.0;
const SIGMOID_THRESHOLD_F64: f64 = 30.0;

pub trait BasicOperations<T>: AddAssign + SubAssign + Add<Output=T> + Mul<Output=T> + Default + Debug + Copy {}

impl<T, U> BasicOperations<T> for U where U: AddAssign + SubAssign + Add<Output=T> + Mul<Output=T> + Default + Debug + Copy {}

/// Trait containing utilities for RVNNs
pub trait Real 
  where Self: Sized {

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn gen_pred(size: usize, index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;

  fn cost(preditcion: &[Self], target: &[Self], cost_model: &CostModel) -> Vec<Self>;

  fn sigmoid(self) -> Self;
}

impl Real for f32 {
  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f32;

    2.0 * float_scale * lcgf32(seed) - float_scale
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size < critical_index { return Err(PredicionError::CriticalIndexOverflow) }

    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![0.0; size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }

  fn cost(prediction: &[Self], target: &[Self], cost_model: &CostModel) -> Vec<Self> {
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match cost_model {
          CostModel::Conventional => { (pred - targ).powi(2) }
        }
      })
      .collect()
  }

  fn sigmoid(self) -> Self {
    if self >= SIGMOID_THRESHOLD_F32 {
      1.0
    } else if self <= -SIGMOID_THRESHOLD_F32 {
      0.0
    } else {
      self.exp() / (1.0 + self.exp())
    }
  }
}
impl Real for f64 {
  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f64;

    2.0 * float_scale * lcgf64(seed) - float_scale
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size < critical_index { return Err(PredicionError::CriticalIndexOverflow) }
    
    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![0.0; size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }

  fn cost(prediction: &[Self], target: &[Self], cost_model: &CostModel) -> Vec<Self> {
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match cost_model {
          CostModel::Conventional => { (pred - targ).powi(2) }
        }
      })
      .collect()
  }

  fn sigmoid(self) -> Self {
    if self >= SIGMOID_THRESHOLD_F64 {
      1.0
    } else if self <= -SIGMOID_THRESHOLD_F64 {
      0.0
    } else {
      self.exp() / (1.0 + self.exp())
    }
  }
}

pub trait Complex where Self: Sized {
  type Precision: Real + BasicOperations<Self::Precision>;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn rit_sigmoid(self) -> Self;

  fn cost(prediction: &[Self], target: &[Self::Precision], cost_model: &ComplexCostModel, criteria: &Criteria) -> Vec<Self::Precision>;

  fn raw_cost(prediction: &[Self], target: &[Self], cost_model: &ComplexCostModel, criteria: &Criteria) -> Vec<Self::Precision>;
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
      x: self.x.sigmoid(),
      y: self.y.sigmoid()
    }
  }

  fn cost(prediction: &[Self], target: &[Self::Precision], cost_model: &ComplexCostModel, criteria: &Criteria) -> Vec<Self::Precision> {    
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match cost_model {
          ComplexCostModel::Conventional => {
            match criteria {
              Criteria::Norm => { (pred.norm_sq() - targ).powi(2) },
              Criteria::Phase => { (pred.phase() - targ).powi(2) },
              Criteria::Real => { (pred.re() - targ).powi(2) },
              Criteria::Imaginary => { (pred.im() - targ).powi(2) }
            }
          }
          ComplexCostModel::Group => {
            let complex_targ = Cf32::new(*targ, 0.0);
            match criteria {
              Criteria::Norm => { (pred - &complex_targ).norm_sq().powi(2) },
              Criteria::Phase => { (pred - &complex_targ).phase().powi(2) },
              Criteria::Real => { (pred - &complex_targ).re().powi(2) },
              Criteria::Imaginary => { (pred - &complex_targ).im().powi(2) }
            }
          }
        }
      })
      .collect()
  }

  fn raw_cost(prediction: &[Self], target: &[Self], cost_model: &ComplexCostModel, criteria: &Criteria) -> Vec<Self::Precision> {    
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match cost_model {
          ComplexCostModel::Conventional => {
            match criteria {
              Criteria::Norm => { (pred.norm() - targ.norm()).powi(2) },
              Criteria::Phase => { (pred.phase() - targ.phase()).powi(2) },
              Criteria::Real => { (pred.re() - targ.re()).powi(2) },
              Criteria::Imaginary => { (pred.im() - targ.im()).powi(2) }
            }
          }
          ComplexCostModel::Group => {
            match criteria {
              Criteria::Norm => { (pred - targ).norm_sq() },
              Criteria::Phase => { (pred - targ).phase().powi(2) },
              Criteria::Real => { (pred - targ).re().powi(2) },
              Criteria::Imaginary => { (pred - targ).im().powi(2) }
            }
          }
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
      x: self.x.sigmoid(),
      y: self.y.sigmoid()
    }
  }

  fn cost(prediction: &[Self], target: &[Self::Precision], cost_model: &ComplexCostModel, criteria: &Criteria) -> Vec<Self::Precision> {
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match cost_model {
          ComplexCostModel::Conventional => {
            match criteria {
              Criteria::Norm => { (pred.norm() - targ).powi(2) },
              Criteria::Phase => { (pred.phase() - targ).powi(2) },
              Criteria::Real => { (pred.re() - targ).powi(2) },
              Criteria::Imaginary => { (pred.im() - targ).powi(2) }
            }
          }
          ComplexCostModel::Group => {
            let complex_targ = Cf64::new(*targ, 0.0);
            match criteria {
              Criteria::Norm => { (pred - &complex_targ).norm_sq() },
              Criteria::Phase => { (pred - &complex_targ).phase().powi(2) },
              Criteria::Real => { (pred - &complex_targ).re().powi(2) },
              Criteria::Imaginary => { (pred - &complex_targ).im().powi(2) }
            }
          }
        }
      })
      .collect()
  }

  fn raw_cost(prediction: &[Self], target: &[Self], cost_model: &ComplexCostModel, criteria: &Criteria) -> Vec<Self::Precision> {    
    prediction
      .iter()
      .zip(target)
      .map(|(pred, targ)| {
        match cost_model {
          ComplexCostModel::Conventional => {
            match criteria {
              Criteria::Norm => { (pred.norm() - targ.norm()).powi(2) },
              Criteria::Phase => { (pred.phase() - targ.phase()).powi(2) },
              Criteria::Real => { (pred.re() - targ.re()).powi(2) },
              Criteria::Imaginary => { (pred.im() - targ.im()).powi(2) }
            }
          }
          ComplexCostModel::Group => {
            match criteria {
              Criteria::Norm => { (pred - targ).norm_sq() },
              Criteria::Phase => { (pred - targ).phase().powi(2) },
              Criteria::Real => { (pred - targ).re().powi(2) },
              Criteria::Imaginary => { (pred - targ).im().powi(2) }
            }
          }
        }
      })
      .collect()
  }
}