use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::default::Default;

use crate::act::ComplexActFunc;
use crate::input::IOType;
use crate::opt::ComplexLossFunc;
use crate::err::{LossCalcError, PredicionError};
use crate::init::PredictModel;

use self::cfloat::{Cf32, Cf64};
use self::random::{lcgf32, lcgf64};

pub mod matrix;
pub mod random;
pub mod cfloat;


pub trait BasicOperations<T>: AddAssign + SubAssign + MulAssign + DivAssign + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + PartialOrd + Default + Debug + Copy + Send + Sync + 'static {}

impl<T, U> BasicOperations<T> for U where U: AddAssign + SubAssign + MulAssign + DivAssign + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + PartialOrd + Default + Debug + Copy + Send + Sync + 'static {}

/// Trait containing utilities for RVNNs
pub trait Real 
  where Self: Sized {
  
  fn unit() -> Self;

  fn usize_to_real(num: usize) -> Self;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;
}

impl Real for f32 {
  fn unit() -> Self {
    1.0
  }

  fn usize_to_real(num: usize) -> Self {
    num as Self
  }

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f32;

    2.0 * float_scale * lcgf32(seed) - float_scale
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size < critical_index { return Err(PredicionError::CriticalIndexOverflow) }

    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![Self::default(); size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }
}

impl Real for f64 {
  fn unit() -> Self {
    1.0
  }

  fn usize_to_real(num: usize) -> Self {
    num as Self
  }

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as f64;

    2.0 * float_scale * lcgf64(seed) - float_scale
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size < critical_index { return Err(PredicionError::CriticalIndexOverflow) }

    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![Self::default(); size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }
}

pub trait Complex where Self: Sized {
  type Precision: Real + BasicOperations<Self::Precision>;

  fn new(re: Self::Precision, im: Self::Precision) -> Self;

  fn usize_to_complex(num: usize) -> Self;

  fn unit() -> Self;

  fn re(&self) -> Self::Precision;

  fn im(&self) -> Self::Precision;

  fn norm(&self) -> Self::Precision;

  fn norm_sq(&self) -> Self::Precision;

  fn phase(&self) -> Self::Precision;

  fn conj(&self) -> Self;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;

  fn activate(&self, func: &ComplexActFunc) -> Self;

  fn d_activate(&self, func: &ComplexActFunc) -> Self;

  fn d_conj_activate(&self, func: &ComplexActFunc) -> Self;

  fn activate_mut(vals: &mut [Self], func: &ComplexActFunc);

  fn d_activate_mut(vals: &mut [Self], func: &ComplexActFunc);

  fn d_conj_activate_mut(vals: &mut [Self], func: &ComplexActFunc);
  
  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<Self::Precision, LossCalcError>;

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<IOType<Self>, LossCalcError>;
}

impl Complex for Cf32 {
  type Precision = f32;

  fn new(re: Self::Precision, im: Self::Precision) -> Self {
    Self { x: re, y: im }
  }

  fn usize_to_complex(num: usize) -> Self {
    Self { x: num as Self::Precision, y: 0.0 }
  }

  fn unit() -> Self {
    Self { x: 1.0, y: 0.0 }
  }

  fn re(&self) -> Self::Precision {
    self.x
  }

  fn im(&self) -> Self::Precision {
    self.y
  }

  fn norm(&self) -> Self::Precision {
    (self.x.powi(2) + self.y.powi(2)).sqrt()
  }

  fn norm_sq(&self) -> Self::Precision {
    self.x.powi(2) + self.y.powi(2)
  }

  fn phase(&self) -> Self::Precision {
    (self.y / self.x).atan()
  }

  fn conj(&self) -> Self {
    Self {
      x: self.x,
      y: -self.y
    }
  }

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as Self::Precision;

    Self { 
      x: 2.0 * float_scale * lcgf32(seed) - float_scale,
      y: 2.0 * float_scale * lcgf32(seed) - float_scale
    }
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size < critical_index { return Err(PredicionError::CriticalIndexOverflow) }

    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![Self::default(); size];
        one_hot_vec[critical_index] += Self::unit();

        Ok(one_hot_vec)
      }
    }
  }

  fn activate(&self, func: &ComplexActFunc) -> Self {
    func.compute_val_cf32(self)
  }

  fn d_activate(&self, func: &ComplexActFunc) -> Self {
    func.compute_d_val_cf32(self)
  }

  fn d_conj_activate(&self, func: &ComplexActFunc) -> Self {
    func.compute_d_conj_val_cf32(self)
  }

  fn activate_mut(vals: &mut [Self], func: &ComplexActFunc) {
    func.compute_cf32(vals);
  }

  fn d_activate_mut(vals: &mut [Self], func: &ComplexActFunc) {
    func.compute_d_cf32(vals);
  }

  fn d_conj_activate_mut(vals: &mut [Self], func: &ComplexActFunc) {
    func.compute_d_conj_cf32(vals)
  }

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<Self::Precision, LossCalcError> {
    loss_func.compute_cf32(prediction, target)
  }

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<IOType<Self>, LossCalcError> {
    loss_func.compute_d_cf32(prediction, target)
  }
}

impl Complex for Cf64 {
  type Precision = f64;

  fn new(re: Self::Precision, im: Self::Precision) -> Self {
    Self { x: re, y: im }
  }

  fn usize_to_complex(num: usize) -> Self {
    Self { x: num as Self::Precision, y: 0.0 }
  }

  fn unit() -> Self {
    Self { x: 1.0, y: 0.0 }
  }

  fn re(&self) -> Self::Precision {
    self.x
  }

  fn im(&self) -> Self::Precision {
    self.y
  }

  fn norm(&self) -> Self::Precision {
    (self.x.powi(2) + self.y.powi(2)).sqrt()
  }

  fn norm_sq(&self) -> Self::Precision {
    self.x.powi(2) + self.y.powi(2)
  }

  fn phase(&self) -> Self::Precision {
    (self.y / self.x).atan()
  }

  fn conj(&self) -> Self {
    Self {
      x: self.x,
      y: -self.y
    }
  }

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as Self::Precision;

    Self { 
      x: 2.0 * float_scale * lcgf64(seed) - float_scale,
      y: 2.0 * float_scale * lcgf64(seed) - float_scale
    }
  }

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError> {
    if size < critical_index { return Err(PredicionError::CriticalIndexOverflow) }

    match pred_method {
      PredictModel::Sparse => { 
        let mut one_hot_vec = vec![Self::default(); size];
        one_hot_vec[critical_index] += Self::unit();

        Ok(one_hot_vec)
      }
    }
  }

  fn activate(&self, func: &ComplexActFunc) -> Self {
    func.compute_val_cf64(self)
  }

  fn d_activate(&self, func: &ComplexActFunc) -> Self {
    func.compute_d_val_cf64(self)
  }

  fn d_conj_activate(&self, func: &ComplexActFunc) -> Self {
    func.compute_d_conj_val_cf64(self)
  }

  fn activate_mut(vals: &mut [Self], func: &ComplexActFunc) {
    func.compute_cf64(vals);
  }

  fn d_activate_mut(vals: &mut [Self], func: &ComplexActFunc) {
    func.compute_d_cf64(vals);
  }

  fn d_conj_activate_mut(vals: &mut [Self], func: &ComplexActFunc) {
    func.compute_d_conj_cf64(vals);
  }

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<Self::Precision, LossCalcError> {
    loss_func.compute_cf64(prediction, target)
  }

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<IOType<Self>, LossCalcError> {
    loss_func.compute_d_cf64(prediction, target)
  }
}