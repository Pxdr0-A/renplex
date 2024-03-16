use crate::err::LossCalcError;
use crate::math::cfloat::{Cf32, Cf64};
use crate::math::Complex;
use crate::input::IOType;

/* Loss Functions. */
fn conv_err_f32(data: (f32, f32)) -> f32 { ( data.0 - data.1 ).powi(2) }
fn conv_err_f64(data: (f64, f64)) -> f64 { ( data.0 - data.1 ).powi(2) }
fn d_conv_err_f32(data: (f32, f32)) -> f32 { 2.0 * ( data.0 - data.1 ) }
fn d_conv_err_f64(data: (f64, f64)) -> f64 { 2.0 * ( data.0 - data.1 ) }
fn conv_err_cf32(data: (Cf32, Cf32)) -> f32 { ( data.0 - data.1 ).norm_sq() }
fn log_err_cf32(data: (Cf32, Cf32)) -> f32 { ((data.0.norm_sq() / data.1.norm_sq()).ln() + (data.0.phase() - data.1.phase()).powi(2)) * 0.5 }
fn log_err_cf64(data: (Cf64, Cf64)) -> f64 { ((data.0.norm_sq() / data.1.norm_sq()).ln() + (data.0.phase() - data.1.phase()).powi(2)) * 0.5 }
fn conv_err_cf64(data: (Cf64, Cf64)) -> f64 { ( data.0 - data.1 ).norm_sq() }

pub enum LossFunc {
  Conventional
}

impl LossFunc {
  pub fn compute_f32(&self, prediction: IOType<f32>, target: IOType<f32>) -> Result<f32, LossCalcError> {
    let func = match self {
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
                .fold(0.0,|acc, data| {
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
                .fold(0.0,|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  pub fn compute_f64(&self, prediction: IOType<f64>, target: IOType<f64>) -> Result<f64, LossCalcError> {
    let func = match self {
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
                .fold(0.0,|acc, data| {
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
                .fold(0.0,|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  pub fn compute_d_f32(&self, prediction: IOType<f32>, target: IOType<f32>) -> Result<IOType<f32>, LossCalcError> {
    let func = match self {
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
              .collect::<Vec<f32>>();
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
              .collect::<Vec<f32>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  pub fn compute_d_f64(&self, prediction: IOType<f64>, target: IOType<f64>) -> Result<IOType<f64>, LossCalcError> {
    let func = match self {
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
              .collect::<Vec<f64>>();
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
              .collect::<Vec<f64>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }
}

pub enum ComplexLossFunc {
  Conventional,
  Log
}

impl ComplexLossFunc {
  pub fn compute_f32(&self, prediction: IOType<Cf32>, target: IOType<Cf32>) -> Result<f32, LossCalcError> {
    let func = match self {
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
                .fold(f32::default(),|acc, data| {
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
                .fold(f32::default(),|acc, data| {
                  acc + func(data)
                })
            )
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  pub fn compute_f64(&self, prediction: IOType<Cf64>, target: IOType<Cf64>) -> Result<f64, LossCalcError> {
    let func = match self {
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
                .fold(f64::default(),|acc, data| {
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
                .fold(f64::default(),|acc, data| {
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
