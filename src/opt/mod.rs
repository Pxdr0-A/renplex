//! Module related to optimization procedures.

use crate::err::LossCalcError;
use crate::math::cfloat::{Cf32, Cf64};
use crate::math::Complex;
use crate::input::IOType;

/* Loss Functions. */

/* Complex Valued (complex input -> real output) */

/* Old version

- 0 is predicted in opt module

*/

/* New Approach */
/* This can be all simplified with macros */

/// Purely an utility
fn sq_error_cf32(targ: Cf32, pred: Cf32) -> f32 {
  ( targ - pred ).norm_sq()
}
/// Purely an utility (to compute the derivative)
fn d_sq_error_cf32(targ: Cf32, pred: Cf32) -> Cf32 {
  // the derivative inverts
  ( pred - targ ).conj()
}
/// Purely an utility
fn sq_error_cf64(targ: Cf64, pred: Cf64) -> f64 {
  ( targ - pred ).norm_sq()
}
/// Purely an utility (to compute the derivative)
fn d_sq_error_cf64(targ: Cf64, pred: Cf64) -> Cf64 {
  // the derivative inverts
  ( pred - targ ).conj()
}

/// Loss function that represents the mean squared error.
/// Complex Loss Functions defined here always return a real number.
/// Used for regression type task since it requires a complex target.
fn mean_sq_loss_cf32<'a, T1: Iterator<Item = &'a Cf32>, T2: Iterator<Item = &'a Cf32>>(targ: T1, pred: T2) -> f32 {
  let mut len = 0;
  let sum_err = targ
    .zip(pred)
    .enumerate()
    .fold(0.0,|acc, (id, (t, p))| {
      len = id;
      acc + sq_error_cf32(*t, *p)
    });
  
  sum_err / ( len as f32 )
}

fn d_mean_sq_loss_cf32<'a, T1: Iterator<Item = &'a Cf32>, T2: Iterator<Item = &'a Cf32>>(targ: T1, pred: T2) -> Vec<Cf32> {
  let mean_err = targ
    .zip(pred)
    .map(|(t, p)| {
      d_sq_error_cf32(*t, *p)
    }).collect();
  
  mean_err
}

/// Loss function that represents the mean squared error.
/// Complex Loss Functions defined here always return a real number.
/// Used for regression type task since it requires a complex target.
fn mean_sq_loss_cf64<'a, T1: Iterator<Item = &'a Cf64>, T2: Iterator<Item = &'a Cf64>>(targ: T1, pred: T2) -> f64 {
  let mut len = 0;
  let sum_err = targ
    .zip(pred)
    .enumerate()
    .fold(0.0,|acc, (id, (t, p))| {
      len = id;
      acc + sq_error_cf64(*t, *p)
    });
  
  sum_err / ( len as f64 )
}

fn d_mean_sq_loss_cf64<'a, T1: Iterator<Item = &'a Cf64>, T2: Iterator<Item = &'a Cf64>>(targ: T1, pred: T2) -> Vec<Cf64> {
  let mean_err = targ
    .zip(pred)
    .map(|(t, p)| {
      d_sq_error_cf64(*t, *p)
    }).collect();
  
  mean_err
}

/// Real Cross Entropy Loss Function.
/// Suitable for classification tasks so it requires a "real" target (in complex format)
fn norm_ce_loss_cf32<'a, T1: Iterator<Item = &'a Cf32>, T2: Iterator<Item = &'a Cf32>>(targ: T1, pred: T2) -> f32 {
  // The real part should contain the one-hot-encoding
  let real_targ = targ.map(|elm| { elm.re() });

  // Compute softmax of the prediction
  let exp_map = pred
    .map(|elm| { elm.norm_sq().exp() })
    .collect::<Vec<_>>();
  let exp_sum = exp_map.iter()
    .fold(0.0, |acc, elm| { acc + *elm });

  // Compute Cross entropy with the softmax values
  let loss = real_targ
    .zip(exp_map)
    .fold(0.0,|acc, (t, exp)| {
      let s = exp / exp_sum;
      acc - t * s.ln() 
    });

  loss
}

/// Derivative of Real Cross Entropy Loss Function
fn d_norm_ce_loss_cf32<'a, T1: Iterator<Item = &'a Cf32>, T2: Iterator<Item = &'a Cf32>>(targ: T1, pred: T2) -> Vec<Cf32> {
  // for now until I figure out what is wrong
  d_mean_sq_loss_cf32(targ, pred)
}

/// Real Cross Entropy Loss Function.
/// Suitable for classification tasks so it requires a real target
fn norm_ce_loss_cf64<'a, T1: Iterator<Item = &'a Cf64>, T2: Iterator<Item = &'a Cf64>>(targ: T1, pred: T2) -> f64 {
  // The real part should contain the one-hot-encoding
  let real_targ = targ.map(|elm| { elm.re() });

  // Compute softmax of the prediction
  let exp_map = pred
    .map(|elm| { elm.norm_sq().exp() })
    .collect::<Vec<_>>();
  let exp_sum = exp_map.iter()
    .fold(0.0, |acc, elm| { acc + *elm });

  // Compute Cross entropy with the softmax values
  let loss = real_targ
    .zip(exp_map)
    .fold(0.0,|acc, (t, exp)| {
      let s = exp / exp_sum;
      acc - t * s.ln() 
    });

  loss
}

/// Derivative of Real Cross Entropy Loss Function
fn d_norm_ce_loss_cf64<'a, T1: Iterator<Item = &'a Cf64>, T2: Iterator<Item = &'a Cf64>>(targ: T1, pred: T2) -> Vec<Cf64> {
  // for now until I figure out what is wrong
  d_mean_sq_loss_cf64(targ, pred)
}

/// List of possible loss function to use.
#[derive(Debug)]
pub enum ComplexLossFunc {
  MeanSquare,
  NormCrossEntropy
}

impl ComplexLossFunc {

  /* Provide the Loss function. */

  /// Returns a function related to the current loss function option 
  /// for complex numbers with 64-bit precision.
  pub fn release_func_cf32<'a, T1: Iterator<Item = &'a Cf32>, T2: Iterator<Item = &'a Cf32>>(&self) -> impl Fn(T1, T2) -> f32 {
    match self {
      Self::MeanSquare => { mean_sq_loss_cf32 },
      Self::NormCrossEntropy => { norm_ce_loss_cf32 }
    }
  }

  /// Returns a function related to the current loss function derivative option 
  /// for complex numbers with 64-bit precision.
  pub fn release_dfunc_cf32<'a, T1: Iterator<Item = &'a Cf32>, T2: Iterator<Item = &'a Cf32>>(&self) -> impl Fn(T1, T2) -> Vec<Cf32> {
    match self {
      Self::MeanSquare => { d_mean_sq_loss_cf32 },
      Self::NormCrossEntropy => { d_norm_ce_loss_cf32 }
    }
  }

  /// Returns a function related to the current loss function option 
  /// for complex numbers with 128-bit precision.
  pub fn release_func_cf64<'a, T1: Iterator<Item = &'a Cf64>, T2: Iterator<Item = &'a Cf64>>(&self) -> impl Fn(T1, T2) -> f64 {
    match self {
      Self::MeanSquare => { mean_sq_loss_cf64 },
      Self::NormCrossEntropy => { norm_ce_loss_cf64 }
    }
  }

  /// Returns a function related to the current loss function derivative option 
  /// for complex numbers with 128-bit precision.
  pub fn release_dfunc_cf64<'a, T1: Iterator<Item = &'a Cf64>, T2: Iterator<Item = &'a Cf64>>(&self) -> impl Fn(T1, T2) -> Vec<Cf64> {
    match self {
      Self::MeanSquare => { d_mean_sq_loss_cf64 },
      Self::NormCrossEntropy => { d_norm_ce_loss_cf64 }
    }
  }

  /* Loss function computation. */

  /// Returns the loss value related to a certain loss function option with 64-bit precision.
  /// 
  /// # Arguments
  /// 
  /// * `target` - [IOType] related to the target output of the network.
  /// * `prediction` - [IOType] related to the prediction of the network.
  pub fn compute_cf32(&self, target: &IOType<Cf32>, prediction: &IOType<Cf32>) -> Result<f32, LossCalcError> {
    match prediction {
      IOType::Scalar(pred) => {
        match target {
          IOType::Scalar(targ) => {
            let func = self.release_func_cf32::<_,_>();
            Ok(func(targ.iter(), pred.iter()))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            let pred_flatten = pred
              .iter().flat_map(|elm| { elm.get_body() });
            let targ_flatten = targ
              .iter().flat_map(|elm| { elm.get_body() });

            let func = self.release_func_cf32::<_,_>();
            Ok(func(targ_flatten, pred_flatten))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  /// Returns the loss value related to a certain loss function option with 128-bit precision.
  /// 
  /// # Arguments
  /// 
  /// * `target` - [IOType] related to the target output of the network.
  /// * `prediction` - [IOType] related to the prediction of the network.
  pub fn compute_cf64(&self, target: &IOType<Cf64>, prediction: &IOType<Cf64>) -> Result<f64, LossCalcError> {
    match prediction {
      IOType::Scalar(pred) => {
        match target {
          IOType::Scalar(targ) => {
            let func = self.release_func_cf64::<_,_>();
            Ok(func(targ.iter(), pred.iter()))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            let pred_flatten = pred
              .iter().flat_map(|elm| { elm.get_body() });
            let targ_flatten = targ
              .iter().flat_map(|elm| { elm.get_body() });

            let func = self.release_func_cf64::<_,_>();
            Ok(func(targ_flatten, pred_flatten))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  /// Returns the derivative loss value related to a certain loss function option with 64-bit precision.
  /// 
  /// # Arguments
  /// 
  /// * `target` - [IOType] related to the target output of the network.
  /// * `prediction` - [IOType] related to the prediction of the network.
  pub fn compute_d_cf32(&self, target: &IOType<Cf32>, prediction: &IOType<Cf32>) -> Result<Vec<Cf32>, LossCalcError> {
    match prediction {
      IOType::Scalar(pred) => {
        match target {
          IOType::Scalar(targ) => {
            let func = self.release_dfunc_cf32::<_,_>();
            Ok(func(targ.iter(), pred.iter()))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            let pred_flatten = pred
              .iter().flat_map(|elm| { elm.get_body() });
            let targ_flatten = targ
              .iter().flat_map(|elm| { elm.get_body() });
            
            let func = self.release_dfunc_cf32::<_,_>();
            Ok(func(targ_flatten, pred_flatten))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  /// Returns the conjugate derivative loss value related to a certain loss function option with 128-bit precision.
  /// 
  /// # Arguments
  /// 
  /// * `target` - [IOType] related to the target output of the network.
  /// * `prediction` - [IOType] related to the prediction of the network.
  pub fn compute_d_cf64(&self, target: &IOType<Cf64>, prediction: &IOType<Cf64>) -> Result<Vec<Cf64>, LossCalcError> {
    match prediction {
      IOType::Scalar(pred) => {
        match target {
          IOType::Scalar(targ) => {
            let func = self.release_dfunc_cf64::<_,_>();
            Ok(func(targ.iter(), pred.iter()))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            let pred_flatten = pred
              .iter().flat_map(|elm| { elm.get_body() });
            let targ_flatten = targ
              .iter().flat_map(|elm| { elm.get_body() });
            
            let func = self.release_dfunc_cf64::<_,_>();
            Ok(func(targ_flatten, pred_flatten))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }
}
