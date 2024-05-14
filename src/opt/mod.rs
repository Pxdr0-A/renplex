use crate::err::LossCalcError;
use crate::math::cfloat::{Cf32, Cf64};
use crate::math::Complex;
use crate::input::IOType;

/* Loss Functions. */

/* Complex Valued (complex input -> real output) */

/* Square Error */
/* For some reason, I cannot flip the predicted with the target. Investigate! */
fn conv_err_cf32(data: (Cf32, Cf32)) -> f32 { ( data.0 - data.1 ).norm_sq() }
fn conv_err_cf64(data: (Cf64, Cf64)) -> f64 { ( data.0 - data.1 ).norm_sq() }
fn d_conv_err_cf32(data: (Cf32, Cf32)) -> Cf32 { (data.0 - data.1).conj() }
fn d_conv_err_cf64(data: (Cf64, Cf64)) -> Cf64 { (data.0 - data.1).conj() }

/* Categorical Cross-Entropy */
fn cross_entropy_f32(data: (f32, f32)) -> f32 {
  -data.1 * data.0.ln()
}
fn cross_entropy_f64(data: (f64, f64)) -> f64 { 
  -data.1 * data.0.ln()
}
fn d_cross_entropy_f32(data: (f32, f32)) -> f32 {
  -data.1 / data.0
}
fn d_cross_entropy_f64(data: (f64, f64)) -> f64 { 
  -data.1 / data.0
}
/* Complex cross entropy (NOT READY!!!) */
/* maybe you need the softmax */
fn ce_err_cf32(data: (Cf32, Cf32)) -> f32 { 
  let (pred_re, pred_im, targ_re, targ_im) = (data.0.re(), data.0.im(), data.1.re(), data.1.im());

  0.5 * ( cross_entropy_f32((pred_re, targ_re)) + cross_entropy_f32((pred_im, targ_im)) )
}
fn ce_err_cf64(data: (Cf64, Cf64)) -> f64 { 
  let (pred_re, pred_im, targ_re, targ_im) = (data.0.re(), data.0.im(), data.1.re(), data.1.im());

  0.5 * ( cross_entropy_f64((pred_re, targ_re)) + cross_entropy_f64((pred_im, targ_im)) )
}
fn d_ce_err_cf32(data: (Cf32, Cf32)) -> Cf32 {
  let (pred_re, pred_im, targ_re, targ_im) = (data.0.re(), data.0.im(), data.1.re(), data.1.im());

  Cf32 {
    x: 0.25 * d_cross_entropy_f32((pred_re, targ_re)),
    y: - 0.25 * d_cross_entropy_f32((pred_im, targ_im))
  }  
}
fn d_ce_err_cf64(data: (Cf64, Cf64)) -> Cf64 {
  let (pred_re, pred_im, targ_re, targ_im) = (data.0.re(), data.0.im(), data.1.re(), data.1.im());

  Cf64 {
    x: 0.25 * d_cross_entropy_f64((pred_re, targ_re)),
    y: - 0.25 * d_cross_entropy_f64((pred_im, targ_im))
  }  
}

pub enum ComplexLossFunc {
  Conventional,
  CCrossEntropy
}

impl ComplexLossFunc {
  pub fn compute_cf32(&self, prediction: IOType<Cf32>, target: IOType<Cf32>) -> Result<f32, LossCalcError> {
    type TargetType = Cf32;
    type SubTargetType = f32;

    let func = match self {
      ComplexLossFunc::Conventional => {
        conv_err_cf32
      },
      ComplexLossFunc::CCrossEntropy => {
        ce_err_cf32
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            let pred_len = pred.len();
            if pred_len != targ.len() { return Err(LossCalcError::InconsistentIO) }

            let mean_err = pred
              .into_iter()
              .zip(targ)
              .fold(SubTargetType::default(),|acc, data| {
                acc + func(data)
              }) / ( pred_len as SubTargetType );
            
            Ok(mean_err)
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::FeatureMaps(pred) => {
        match target {
          IOType::FeatureMaps(targ) => {
            let pred_flatten = pred.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            let targ_flatten = targ.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            
            let pred_len = pred_flatten.len();
            let targ_len = targ_flatten.len();
            if pred_len != targ_len { return Err(LossCalcError::InconsistentIO) }

            let mean_err = pred_flatten
              .into_iter()
              .zip(targ_flatten)
              .fold(SubTargetType::default(),|acc, data| {
                acc + func(data)
              }) / ( pred_len as SubTargetType );
            
            Ok(mean_err)
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  pub fn compute_cf64(&self, prediction: IOType<Cf64>, target: IOType<Cf64>) -> Result<f64, LossCalcError> {
    type TargetType = Cf64;
    type SubTargetType = f64;

    let func = match self {
      ComplexLossFunc::Conventional => {
        conv_err_cf64
      },
      ComplexLossFunc::CCrossEntropy => {
        ce_err_cf64
      }
    };

    match prediction {
      IOType::Vector(pred) => {
        match target {
          IOType::Vector(targ) => {
            let pred_len = pred.len();
            if pred_len != targ.len() { return Err(LossCalcError::InconsistentIO) }

            let mean_err = pred
              .into_iter()
              .zip(targ)
              .fold(SubTargetType::default(),|acc, data| {
                acc + func(data)
              }) / ( pred_len as SubTargetType );
            
            Ok(mean_err)
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::FeatureMaps(pred) => {
        match target {
          IOType::FeatureMaps(targ) => {
            let pred_flatten = pred.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            let targ_flatten = targ.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            
            let pred_len = pred_flatten.len();
            let targ_len = targ_flatten.len();
            if pred_len != targ_len { return Err(LossCalcError::InconsistentIO) }

            let mean_err = pred_flatten
              .into_iter()
              .zip(targ_flatten)
              .fold(SubTargetType::default(),|acc, data| {
                acc + func(data)
              }) / ( pred_len as SubTargetType );
            
            Ok(mean_err)
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  pub fn compute_d_cf32(&self, prediction: IOType<Cf32>, target: IOType<Cf32>) -> Result<IOType<Cf32>, LossCalcError> {
    type TargetType = Cf32;

    let func = match self {
      ComplexLossFunc::Conventional => {
        d_conv_err_cf32
      },
      ComplexLossFunc::CCrossEntropy => {
        d_ce_err_cf32
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
              .collect::<Vec<TargetType>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::FeatureMaps(pred) => {
        match target {
          IOType::FeatureMaps(targ) => {
            let pred_flatten = pred.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            let targ_flatten = targ.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            
            let pred_len = pred_flatten.len();
            let targ_len = targ_flatten.len();
            if pred_len != targ_len { return Err(LossCalcError::InconsistentIO) }

            let error_der = pred_flatten
              .into_iter()
              .zip(targ_flatten)
              .map(func)
              .collect::<Vec<TargetType>>();
            
            Ok(IOType::Vector(error_der))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }

  pub fn compute_d_cf64(&self, prediction: IOType<Cf64>, target: IOType<Cf64>) -> Result<IOType<Cf64>, LossCalcError> {
    type TargetType = Cf64;

    let func = match self {
      ComplexLossFunc::Conventional => {
        d_conv_err_cf64
      },
      ComplexLossFunc::CCrossEntropy => {
        d_ce_err_cf64
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
              .collect::<Vec<TargetType>>();
            Ok(IOType::Vector(vec))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      },
      IOType::FeatureMaps(pred) => {
        match target {
          IOType::FeatureMaps(targ) => {
            let pred_flatten = pred.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            let targ_flatten = targ.into_iter().map(|elm| { elm.export_body() }).flatten().collect::<Vec<TargetType>>();
            
            let pred_len = pred_flatten.len();
            let targ_len = targ_flatten.len();
            if pred_len != targ_len { return Err(LossCalcError::InconsistentIO) }

            let error_der = pred_flatten
              .into_iter()
              .zip(targ_flatten)
              .map(func)
              .collect::<Vec<TargetType>>();
            
            Ok(IOType::Vector(error_der))
          },
          _ => { Err(LossCalcError::InconsistentIO) }
        }
      }
    }
  }
}
