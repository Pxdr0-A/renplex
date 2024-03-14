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


/* Activation functions. Will soon be moved do act module. */
const SIGMOID_THRESHOLD_F32: f32 = 15.0;
const SIGMOID_THRESHOLD_F64: f64 = 30.0;

fn sigmoid_f32(val: f32) -> f32 {
  if val >= SIGMOID_THRESHOLD_F32 { 1.0 } 
  else if val <= -SIGMOID_THRESHOLD_F32 { 0.0 } 
  else { val.exp() / (1.0 + val.exp()) }
}

fn sigmoid_f64(val: f64) -> f64 {
  if val >= SIGMOID_THRESHOLD_F64 { 1.0 } 
  else if val <= -SIGMOID_THRESHOLD_F64 { 0.0 } 
  else { val.exp() / (1.0 + val.exp()) }
}

fn d_sigmoid_f32(val: f32) -> f32 {
  sigmoid_f32(val) * (1.0 - sigmoid_f32(val))
}

fn d_sigmoid_f64(val: f64) -> f64 {
  sigmoid_f64(val) * (1.0 - sigmoid_f64(val))
}

fn ritsigmoid_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: sigmoid_f32(val.x), 
    y: sigmoid_f32(val.y)
  }
}

fn ritsigmoid_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: sigmoid_f64(val.x), 
    y: sigmoid_f64(val.y)
  }
}

/* Loss Functions. Will soon be moved to opt module. */

fn conv_err_f32(data: (f32, f32)) -> f32 { ( data.0 - data.1 ).powi(2) }
fn conv_err_f64(data: (f64, f64)) -> f64 { ( data.0 - data.1 ).powi(2) }
fn d_conv_err_f32(data: (f32, f32)) -> f32 { 2.0 * ( data.0 - data.1 ) }
fn d_conv_err_f64(data: (f64, f64)) -> f64 { 2.0 * ( data.0 - data.1 ) }
fn conv_err_cf32(data: (Cf32, Cf32)) -> f32 { ( data.0 - data.1 ).norm_sq() }
fn log_err_cf32(data: (Cf32, Cf32)) -> f32 { ((data.0.norm_sq() / data.1.norm_sq()).ln() + (data.0.phase() - data.1.phase()).powi(2)) * 0.5 }
fn log_err_cf64(data: (Cf64, Cf64)) -> f64 { ((data.0.norm_sq() / data.1.norm_sq()).ln() + (data.0.phase() - data.1.phase()).powi(2)) * 0.5 }
fn conv_err_cf64(data: (Cf64, Cf64)) -> f64 { ( data.0 - data.1 ).norm_sq() }

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
    let act_func = match func {
      ActFunc::Sigmoid => {
        sigmoid_f32
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  fn d_activate_mut(vals: &mut [Self], func: &ActFunc) {
    let act_func = match func {
      ActFunc::Sigmoid => {
        d_sigmoid_f32
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<Self, LossCalcError> {
    let func = match loss_func {
      LossFunc::Conventional => {
        conv_err_f32
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ)
                .fold(Self::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ.into_iter())
                .fold(Self::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<IOType<Self>, LossCalcError> {
    let func = match loss_func {
      LossFunc::Conventional => {
        d_conv_err_f32
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            let vec = pred
              .into_iter()
              .zip(targ)
              .map(func)
              .collect::<Vec<Self>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            let vec = pred
              .into_iter()
              .zip(targ.into_iter())
              .map(func)
              .collect::<Vec<Self>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
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
        let mut one_hot_vec = vec![0.0; size];
        one_hot_vec[critical_index] += 1.0;

        Ok(one_hot_vec)
      }
    }
  }

  fn activate_mut(vals: &mut [Self], func: &ActFunc) {
    let act_func = match func {
      ActFunc::Sigmoid => {
        sigmoid_f64
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  fn d_activate_mut(vals: &mut [Self], func: &ActFunc) {
    let act_func = match func {
      ActFunc::Sigmoid => {
        d_sigmoid_f64
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<Self, LossCalcError> {
    let func = match loss_func {
      LossFunc::Conventional => {
        conv_err_f64
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ)
                .fold(Self::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ.into_iter())
                .fold(Self::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  fn d_loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &LossFunc) -> Result<IOType<Self>, LossCalcError> {
    let func = match loss_func {
      LossFunc::Conventional => {
        d_conv_err_f64
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            let vec = pred
              .into_iter()
              .zip(targ)
              .map(func)
              .collect::<Vec<Self>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            let vec = pred
              .into_iter()
              .zip(targ.into_iter())
              .map(func)
              .collect::<Vec<Self>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
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

  fn gen_pred(size: usize, critical_index: usize, pred_method: &PredictModel) -> Result<Vec<Self>, PredicionError>;

  fn activate_mut(vals: &mut [Self], func: &ComplexActFunc);

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<Self::Precision, LossCalcError>;
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
    let act_func = match func {
      ComplexActFunc::RITSigmoid => {
        ritsigmoid_cf32
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  } 

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<Self::Precision, LossCalcError> {
    let func = match loss_func {
      ComplexLossFunc::Conventional => {
        conv_err_cf32
      },
      ComplexLossFunc::Log => {
        log_err_cf32
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ)
                .fold(Self::Precision::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ.into_iter())
                .fold(Self::Precision::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
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
    let act_func = match func {
      ComplexActFunc::RITSigmoid => {
        ritsigmoid_cf64
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val)
    }
  } 

  fn loss(prediction: IOType<Self>, target: IOType<Self>, loss_func: &ComplexLossFunc) -> Result<Self::Precision, LossCalcError> {
    let func = match loss_func {
      ComplexLossFunc::Conventional => {
        conv_err_cf64
      },
      ComplexLossFunc::Log => {
        log_err_cf64
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ)
                .fold(Self::Precision::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::Matrix(pred) => {
        match target {
          IOType::Matrix(targ) => {
            Ok(
              pred
                .into_iter()
                .zip(targ.into_iter())
                .fold(Self::Precision::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }
}