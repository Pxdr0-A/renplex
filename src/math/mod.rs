use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::default::Default;

use crate::act::{ActFunc, ComplexActFunc};
use crate::input::IOType;
use crate::opt::ComplexLossFunc;
use crate::opt::LossFunc;
use crate::err::{LossCalcError, PredicionError};
use crate::init::PredictModel;

use self::cfloat::{Cf32, Cf64};
use self::random::{lcgf32, lcgf64};

pub mod matrix;
pub mod random;
pub mod cfloat;


pub trait BasicOperations<T>: AddAssign + SubAssign + MulAssign + DivAssign + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Default + Debug + Copy {}

impl<T, U> BasicOperations<T> for U where U: AddAssign + SubAssign + MulAssign + DivAssign + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Default + Debug + Copy {}

/// Trait containing utilities for RVNNs
pub trait Real 
  where Self: Sized {
  
  fn unit() -> Self;

  fn usize_to_real(num: usize) -> Self;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;

  fn activate_mut(vals: &mut [Self], func: &ActFunc);

  fn d_activate_mut(vals: &mut [Self], func: &ActFunc);

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<Self, LossCalcError>;

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<IOType<Self>, LossCalcError>;
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
        let mut one_hot_vec = vec![0.0; size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }

  fn activate_mut(vals: &mut [Self], func: &ActFunc) {
    func.compute_f32(vals);
  }

  fn d_activate_mut(vals: &mut [Self], func: &ActFunc) {
    func.compute_d_f32(vals)
  }

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<Self, LossCalcError> {
    loss_func.compute_f32(prediction, target)
  }

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<IOType<Self>, LossCalcError> {
    loss_func.compute_d_f32(prediction, target)
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
        let mut one_hot_vec = vec![0.0; size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }

  fn activate_mut(vals: &mut [Self], func: &ActFunc) {
    func.compute_f64(vals);
  }

  fn d_activate_mut(vals: &mut [Self], func: &ActFunc) {
    func.compute_d_f64(vals);
  }

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<Self, LossCalcError> {
    loss_func.compute_f64(prediction, target)
  }

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<IOType<Self>, LossCalcError> {
    loss_func.compute_d_f64(prediction, target)
  }
}

pub trait Complex where Self: Sized {
  type Precision: Real + BasicOperations<Self::Precision>;

  fn new(re: Self::Precision, im: Self::Precision) -> Self;

  fn unit() -> Self;

  fn re(&self) -> Self::Precision;

  fn im(&self) -> Self::Precision;

  fn norm(&self) -> Self::Precision;

  fn norm_sq(&self) -> Self::Precision;

  fn phase(&self) -> Self::Precision;

  fn conj(&self) -> Self;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;

  fn activate_mut(vals: &mut [Self], func: &ComplexActFunc);

  fn d_activate_mut(vals: &mut [Self], func: &ComplexActFunc);

  fn d_conj_activate_mut(vals: &mut [Self], func: &ComplexActFunc);
  
  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<Self::Precision, LossCalcError>;

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<IOType<Self>, LossCalcError>;

  fn d_conj_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<IOType<Self>, LossCalcError>;
}

impl Complex for Cf32 {
  type Precision = f32;

  fn new(re: Self::Precision, im: Self::Precision) -> Self {
    Self { x: re, y: im }
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
    (self.y / self.x).tan()
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
        one_hot_vec[critical_index] += Self { x: 1.0, y: 1.0 };

        Ok(one_hot_vec)
      }
    }
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

  fn d_conj_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<IOType<Self>, LossCalcError> {
    loss_func.compute_d_conj_cf32(prediction, target)
  }
}

impl Complex for Cf64 {
  type Precision = f64;

  fn new(re: Self::Precision, im: Self::Precision) -> Self {
    Self { x: re, y: im }
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
    (self.y / self.x).tan()
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
        one_hot_vec[critical_index] += Self { x: 1.0, y: 1.0 };

        Ok(one_hot_vec)
      }
    }
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

  fn d_conj_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<IOType<Self>, LossCalcError> {
    loss_func.compute_d_conj_cf64(prediction, target)
  }
}