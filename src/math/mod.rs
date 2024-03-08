use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};
use std::default::Default;

use crate::opt::ComplexLossFunc;
use crate::opt::LossFunc;
use crate::err::PredicionError;
use crate::init::PredictModel;

use self::cfloat::{Cf32, Cf64};
use self::random::{lcgf32, lcgf64};

pub mod matrix;
pub mod random;
pub mod cfloat;

const SIGMOID_THRESHOLD_F32: f32 = 15.0;
const SIGMOID_THRESHOLD_F64: f64 = 30.0;

pub trait BasicOperations<T>: AddAssign + SubAssign + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Default + Debug + Copy {}

impl<T, U> BasicOperations<T> for U where U: AddAssign + SubAssign + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Default + Debug + Copy {}

/// Trait containing utilities for RVNNs
pub trait Real 
  where Self: Sized {

  fn pow(&self, n: i32) -> Self;

  fn log(&self) -> Self;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn gen_pred(size: usize, index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;

  fn loss(preditcion: &[Self], target: &[Self], loss_func: &LossFunc) -> Self;

  fn sigmoid(self) -> Self;
}

impl Real for f32 {
  fn pow(&self, n: i32) -> Self {
    self.powi(n)
  }

  fn log(&self) -> Self {
    self.ln()
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

  fn loss(prediction: &[Self], target: &[Self], loss_func: &LossFunc) -> Self {
    let func = match loss_func {
      LossFunc::Conventional => {
        |data: (&Self, &Self)| { 
          ( data.0 - data.1 ).powi(2)
        }
      }
    };

    prediction
      .iter()
      .zip(target)
      .map(func)
      .fold(Self::default(), |acc, elm| { acc + elm })
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
  fn pow(&self, n: i32) -> Self {
    self.powi(n)
  }

  fn log(&self) -> Self {
    self.ln()
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

  fn loss(prediction: &[Self], target: &[Self], loss_func: &LossFunc) -> Self {
    let func = match loss_func {
      LossFunc::Conventional => {
        |data: (&Self, &Self)| { 
          ( data.0 - data.1 ).powi(2)
        }
      }
    };

    prediction
      .iter()
      .zip(target)
      .map(func)
      .fold(Self::default(), |acc, elm| { acc + elm })
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

  fn new(re: Self::Precision, im: Self::Precision) -> Self;

  fn re(&self) -> Self::Precision;

  fn im(&self) -> Self::Precision;

  fn norm(&self) -> Self::Precision;

  fn norm_sq(&self) -> Self::Precision;

  fn phase(&self) -> Self::Precision;

  fn gen(seed: &mut u128, scale: usize) -> Self;

  fn rit_sigmoid(self) -> Self;

  fn loss(prediction: &[Self], target: &[Self], loss_func: &ComplexLossFunc) -> Self::Precision;
}

impl Complex for Cf32 {
  type Precision = f32;

  fn new(re: Self::Precision, im: Self::Precision) -> Self {
    Self { x: re, y: im }
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

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as Self::Precision;

    Self { 
      x: 2.0 * float_scale * lcgf32(seed) - float_scale,
      y: 2.0 * float_scale * lcgf32(seed) - float_scale
    }
  }

  fn rit_sigmoid(self) -> Self {
    Self {
      x: self.x.sigmoid(),
      y: self.y.sigmoid()
    }
  }

  fn loss(prediction: &[Self], target: &[Self], loss_func: &ComplexLossFunc) -> Self::Precision {
    let func = match loss_func {
      ComplexLossFunc::Conventional => {
        |acc: Self::Precision, data: (&Self, &Self)| {
          acc + ( data.0 - data.1 ).norm_sq()
        }
      },
      ComplexLossFunc::Log => {
        |acc: Self::Precision, data: (&Self, &Self)| { 
          acc + ((data.0.norm_sq() / data.1.norm_sq()).ln() + (data.0.phase() - data.1.phase()).powi(2)) * 0.5
        }
      }
    };

    prediction
      .iter()
      .zip(target)
      .fold(Self::Precision::default(), func)
  }
}

impl Complex for Cf64 {
  type Precision = f64;

  fn new(re: Self::Precision, im: Self::Precision) -> Self {
    Self { x: re, y: im }
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

  fn gen(seed: &mut u128, scale: usize) -> Self {
    let float_scale = scale as Self::Precision;

    Self { 
      x: 2.0 * float_scale * lcgf64(seed) - float_scale,
      y: 2.0 * float_scale * lcgf64(seed) - float_scale
    }
  }

  fn rit_sigmoid(self) -> Self {
    Self {
      x: self.x.sigmoid(),
      y: self.y.sigmoid()
    }
  }

  fn loss(prediction: &[Self], target: &[Self], loss_func: &ComplexLossFunc) -> Self::Precision {
    let func = match loss_func {
      ComplexLossFunc::Conventional => {
        |data: (&Self, &Self)| { 
          ( data.0 - data.1 ).norm_sq() 
        }
      },
      ComplexLossFunc::Log => {
        |data: (&Self, &Self)| { 
          ((data.0.norm_sq() / data.1.norm_sq()).ln() + (data.0.phase() - data.1.phase()).powi(2)) * 0.5
        }
      }
    };

    prediction
      .iter()
      .zip(target)
      .map(func)
      .fold(Self::Precision::default(), |acc, elm| { acc + elm })
  }
}