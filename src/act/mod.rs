//! Module containing everything related to activation functions and respective derivatives.

use crate::math::{cfloat::{Cf32, Cf64}, Complex};

/* Activation functions. */

/* Sigmoid */
const SIGMOID_THRESHOLD_F32: f32 = 15.0;
const SIGMOID_THRESHOLD_F64: f64 = 30.0;

/* Real function utilitis for RIT */
fn sigmoid_f32(val: f32) -> f32 {
  if val >= SIGMOID_THRESHOLD_F32 { 0.999999 } 
  else if val <= -SIGMOID_THRESHOLD_F32 { 0.000001 } 
  else { val.exp() / (1.0 + val.exp()) }
}

fn sigmoid_f64(val: f64) -> f64 {
  if val >= SIGMOID_THRESHOLD_F64 { 0.99999999999999 } 
  else if val <= -SIGMOID_THRESHOLD_F64 { 0.00000000000001 } 
  else { val.exp() / (1.0 + val.exp()) }
}

fn d_sigmoid_f32(val: f32) -> f32 {
  sigmoid_f32(val) * (1.0 - sigmoid_f32(val))
}

fn d_sigmoid_f64(val: f64) -> f64 {
  sigmoid_f64(val) * (1.0 - sigmoid_f64(val))
}

fn relu_f32(val: f32) -> f32 {
  if val.is_sign_positive() {
    val
  } else {
    f32::default()
  }
}
fn relu_f64(val: f64) -> f64 {
  if val.is_sign_positive() {
    val
  } else {
    f64::default()
  }
}

fn d_relu_f32(val: f32) -> f32 {
  if val.is_sign_positive() {
    1.0
  } else {
    f32::default()
  }
}

fn d_relu_f64(val: f64) -> f64 {
  if val.is_sign_positive() {
    1.0
  } else {
    f64::default()
  }
}

/* RITSigmoid */
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

fn d_ritsigmoid_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: (d_sigmoid_f32(val.x) + d_sigmoid_f32(val.y)) * 0.5, 
    y: 0.0
  }
}

fn d_ritsigmoid_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: (d_sigmoid_f64(val.x) + d_sigmoid_f64(val.y)) * 0.5, 
    y: 0.0
  }
}

fn d_conj_ritsigmoid_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: (d_sigmoid_f32(val.x) - d_sigmoid_f32(val.y)) * 0.5, 
    y: 0.0
  }
}

fn d_conj_ritsigmoid_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: (d_sigmoid_f64(val.x) - d_sigmoid_f64(val.y)) * 0.5, 
    y: 0.0
  }
}

/* RITTanh */
fn rittanh_cf32(val: Cf32) -> Cf32 {
  Cf32 { x: val.x.tanh(), y: val.y.tanh() }
}

fn rittanh_cf64(val: Cf64) -> Cf64 {
  Cf64 { x: val.x.tanh(), y: val.y.tanh() }
}

fn d_rittanh_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: ( (1.0 / val.x.cosh().powi(2)) + (1.0 / val.y.cosh().powi(2)) ) * 0.5,
    y: 0.0
  }
}

fn d_rittanh_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: ( (1.0 / val.x.cosh().powi(2)) + (1.0 / val.y.cosh().powi(2)) ) * 0.5,
    y: 0.0
  }
}

fn d_conj_rittanh_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: ( (1.0 / val.x.cosh().powi(2)) - (1.0 / val.y.cosh().powi(2)) ) * 0.5,
    y: 0.0
  }
}

fn d_conj_rittanh_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: ( (1.0 / val.x.cosh().powi(2)) - (1.0 / val.y.cosh().powi(2)) ) * 0.5,
    y: 0.0
  }
}

/* RITReLU */
fn ritrelu_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: relu_f32(val.x),
    y: relu_f32(val.y)
  }
}

fn ritrelu_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: relu_f64(val.x),
    y: relu_f64(val.y)
  }
}

fn d_ritrelu_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: ( d_relu_f32(val.x) + d_relu_f32(val.y) ) * 0.5,
    y: 0.0
  }
}

fn d_ritrelu_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: ( d_relu_f64(val.x) + d_relu_f64(val.y) ) * 0.5,
    y: 0.0
  }
}

fn d_conj_ritrelu_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: ( d_relu_f32(val.x) - d_relu_f32(val.y) ) * 0.5,
    y: 0.0
  }
}

fn d_conj_ritrelu_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: ( d_relu_f64(val.x) - d_relu_f64(val.y) ) * 0.5,
    y: 0.0
  }
}

/* zReLU */
fn zrelu_cf32(val: Cf32) -> Cf32 {
  if val.re().is_sign_negative() || val.im().is_sign_negative() {
    Cf32 { x: 0.0, y: 0.0 }
  } else {
    val
  }
}

fn zrelu_cf64(val: Cf64) -> Cf64 {
  if val.re().is_sign_negative() || val.im().is_sign_negative() {
    Cf64 { x: 0.0, y: 0.0 }
  } else {
    val
  }
}

fn d_zrelu_cf32(val: Cf32) -> Cf32 {
  if val.re().is_sign_negative() || val.im().is_sign_negative() {
    Cf32 { x: 0.0, y: 0.0 }
  } else {
    Cf32 { x: 1.0, y: 0.0 }
  }
}

fn d_zrelu_cf64(val: Cf64) -> Cf64 {
  if val.re().is_sign_negative() || val.im().is_sign_negative() {
    Cf64 { x: 0.0, y: 0.0 }
  } else {
    Cf64 { x: 1.0, y: 0.0 }
  }
}

fn d_conj_zrelu_cf32(_val: Cf32) -> Cf32 {
  Cf32 { x: 0.0, y: 0.0 }
}

fn d_conj_zrelu_cf64(_val: Cf64) -> Cf64 {
  Cf64 { x: 0.0, y: 0.0 }
}

/* Cardioid */
fn card_cf32(val: Cf32) -> Cf32 {
  let phase = val.phase();
  
  let cos = phase.cos();
  let sin = phase.sin();

  Cf32 { x: 2.0 * (1.0 + cos) * cos, y: 2.0 * (1.0 + cos) * sin }
}

fn card_cf64(val: Cf64) -> Cf64 {
  let phase = val.phase();
  
  let cos = phase.cos();
  let sin = phase.sin();

  Cf64 { x: 2.0 * (1.0 + cos) * cos, y: 2.0 * (1.0 + cos) * sin }
}

fn d_card_cf32(val: Cf32) -> Cf32 {
  let norm = val.norm();
  let phase = val.phase();
  
  let cos = phase.cos();
  let sin = phase.sin();
  let cos2 = cos.powi(2);
  let sin_2x = (2.0 * phase).sin();
  let i = Cf32::iunit();

  let unfactor_re = sin + sin_2x;
  let unfactor_im = 2.0 * cos2 + cos - 1.0;
  let x_factor = sin / norm;
  let y_factor = cos / norm;

  let dfdx = Cf32::new(
    2.0 * x_factor * unfactor_re,
    -2.0 * x_factor * unfactor_im
  );

  let dfdy = Cf32::new(
    2.0 * y_factor * unfactor_re,
    -2.0 * y_factor * unfactor_im
  );

  dfdx - i * dfdy
}

fn d_card_cf64(val: Cf64) -> Cf64 {
  let norm = val.norm();
  let phase = val.phase();
  
  let cos = phase.cos();
  let sin = phase.sin();
  let cos2 = cos.powi(2);
  let sin_2x = (2.0 * phase).sin();
  let i = Cf64::iunit();

  let unfactor_re = sin + sin_2x;
  let unfactor_im = 2.0 * cos2 + cos - 1.0;
  let x_factor = sin / norm;
  let y_factor = cos / norm;

  let dfdx = Cf64::new(
    2.0 * x_factor * unfactor_re,
    -2.0 * x_factor * unfactor_im
  );

  let dfdy = Cf64::new(
    2.0 * y_factor * unfactor_re,
    -2.0 * y_factor * unfactor_im
  );

  dfdx - i * dfdy
}

fn d_conj_card_cf32(val: Cf32) -> Cf32 {
  let norm = val.norm();
  let phase = val.phase();
  
  let cos = phase.cos();
  let sin = phase.sin();
  let cos2 = cos.powi(2);
  let sin_2x = (2.0 * phase).sin();
  let i = Cf32::iunit();

  let unfactor_re = sin + sin_2x;
  let unfactor_im = 2.0 * cos2 + cos - 1.0;
  let x_factor = sin / norm;
  let y_factor = cos / norm;

  let dfdx = Cf32::new(
    2.0 * x_factor * unfactor_re,
    -2.0 * x_factor * unfactor_im
  );

  let dfdy = Cf32::new(
    2.0 * y_factor * unfactor_re,
    -2.0 * y_factor * unfactor_im
  );

  dfdx + i * dfdy
}

fn d_conj_card_cf64(val: Cf64) -> Cf64 {
  let norm = val.norm();
  let phase = val.phase();
  
  let cos = phase.cos();
  let sin = phase.sin();
  let cos2 = cos.powi(2);
  let sin_2x = (2.0 * phase).sin();
  let i = Cf64::iunit();

  let unfactor_re = sin + sin_2x;
  let unfactor_im = 2.0 * cos2 + cos - 1.0;
  let x_factor = sin / norm;
  let y_factor = cos / norm;

  let dfdx = Cf64::new(
    2.0 * x_factor * unfactor_re,
    -2.0 * x_factor * unfactor_im
  );

  let dfdy = Cf64::new(
    2.0 * y_factor * unfactor_re,
    -2.0 * y_factor * unfactor_im
  );

  dfdx + i * dfdy
}

/* modz */
fn mod_z_cf32(val: Cf32) -> Cf32 {
  let norm = val.norm();

  val / Cf32::new(norm, 0.0)
}

fn d_mod_z_cf32(val: Cf32) -> Cf32 {
  let norm2 = 2.0 * val.norm();

  Cf32::new(1.0 / norm2, 0.0) 
}

fn d_conj_mod_z_cf32(val: Cf32) -> Cf32 {
  let norm2 = 2.0 * val.norm();

  (val / val.conj()) / Cf32::new(-norm2, 0.0)
}

fn mod_z_cf64(val: Cf64) -> Cf64 {
  let norm = val.norm();

  val / Cf64::new(norm, 0.0)
}

fn d_mod_z_cf64(val: Cf64) -> Cf64 {
  let norm2 = 2.0 * val.norm();

  Cf64::new(1.0 / norm2, 0.0) 
}

fn d_conj_mod_z_cf64(val: Cf64) -> Cf64 {
  let norm2 = 2.0 * val.norm();

  (val / val.conj()) / Cf64::new(-norm2, 0.0)
}

fn none_cf32(val: Cf32) -> Cf32 {
  val
}

fn none_cf64(val: Cf64) -> Cf64 {
  val
}

fn d_none_cf32(_val: Cf32) -> Cf32 {
  Cf32::unit()
}

fn d_none_cf64(_val: Cf64) -> Cf64 {
  Cf64::unit()
}

fn d_conj_none_cf32(_val: Cf32) -> Cf32 {
  Cf32::default()
}

fn d_conj_none_cf64(_val: Cf64) -> Cf64 {
  Cf64::default()
}

/// Enumeration containing all the possibilities of complex activation functions.
#[derive(Debug, Clone, Copy)]
pub enum ComplexActFunc {
  None,
  ModZ,
  RITSigmoid,
  RITTanh,
  RITReLU,
  ZReLU,
  Cardioid
}

impl ComplexActFunc {
  /// Returns a function related to the current complex activation function option 
  /// for complex numbers with 64-bit precision.
  pub fn release_func_cf32(&self) -> fn(Cf32) -> Cf32 {
    match self {
      Self::None => { none_cf32 },
      Self::ModZ => { mod_z_cf32 },
      Self::RITSigmoid => { ritsigmoid_cf32 },
      Self::RITTanh => { rittanh_cf32 }
      Self::RITReLU => { ritrelu_cf32 },
      Self::ZReLU => { zrelu_cf32 },
      Self::Cardioid => { card_cf32 }
    }
  }

  /// Returns a function related to the current complex activation function option 
  /// for complex numbers with 128-bit precision.
  pub fn release_func_cf64(&self) -> fn(Cf64) -> Cf64 {
    match self {
      Self::None => { none_cf64 },
      Self::ModZ => { mod_z_cf64 },
      Self::RITSigmoid => { ritsigmoid_cf64 },
      Self::RITTanh => { rittanh_cf64 }
      Self::RITReLU => { ritrelu_cf64 },
      Self::ZReLU => { zrelu_cf64 },
      Self::Cardioid => { card_cf64 }
    }
  }

  /// Returns a function related to the derivative of 
  /// current complex activation function option 
  /// for complex numbers with 64-bit precision.
  pub fn release_dfunc_cf32(&self) -> fn(Cf32) -> Cf32 {
    match self {
      Self::None => { d_none_cf32 },
      Self::ModZ => { d_mod_z_cf32 },
      Self::RITSigmoid => { d_ritsigmoid_cf32 },
      Self::RITTanh => { d_rittanh_cf32 },
      Self::RITReLU => { d_ritrelu_cf32 },
      Self::ZReLU => { d_zrelu_cf32 },
      Self::Cardioid => { d_card_cf32 }
    }
  }

  /// Returns a function related to the derivative of 
  /// current complex activation function option 
  /// for complex numbers with 128-bit precision.
  pub fn release_dfunc_cf64(&self) -> fn(Cf64) -> Cf64 {
    match self {
      Self::None => { d_none_cf64 },
      Self::ModZ => { d_mod_z_cf64 },
      Self::RITSigmoid => { d_ritsigmoid_cf64 },
      Self::RITTanh => { d_rittanh_cf64 },
      Self::RITReLU => { d_ritrelu_cf64 },
      Self::ZReLU => { d_zrelu_cf64 },
      Self::Cardioid => { d_card_cf64 }
    }
  }

  /// Returns a function related to the conjugate derivative of 
  /// current complex activation function option 
  /// for complex numbers with 64-bit precision.
  pub fn release_dfunc_conj_cf32(&self) -> fn(Cf32) -> Cf32 {
    match self {
      Self::None => { d_conj_none_cf32 },
      Self::ModZ => { d_conj_mod_z_cf32 },
      Self::RITSigmoid => { d_conj_ritsigmoid_cf32 },
      Self::RITTanh => { d_conj_rittanh_cf32 },
      Self::RITReLU => { d_conj_ritrelu_cf32 },
      Self::ZReLU => { d_conj_zrelu_cf32 },
      Self::Cardioid => { d_conj_card_cf32 }
    }
  }

  /// Returns a function related to the conjugate derivative of 
  /// current complex activation function option 
  /// for complex numbers with 128-bit precision.
  pub fn release_dfunc_conj_cf64(&self) -> fn(Cf64) -> Cf64 {
    match self {
      Self::None => { d_conj_none_cf64 },
      Self::ModZ => { d_conj_mod_z_cf64 },
      Self::RITSigmoid => { d_conj_ritsigmoid_cf64 },
      Self::RITTanh => { d_conj_rittanh_cf64 },
      Self::RITReLU => { d_conj_ritrelu_cf64 },
      Self::ZReLU => { d_conj_zrelu_cf64 },
      Self::Cardioid => { d_conj_card_cf64 }
    }
  }

  /* multiple values */

  /// Modifies a slice of complex values with 64-bit precision by 
  /// computing the respective complex activation option defined.
  /// 
  /// # Arguments
  /// 
  /// * `vals` - values to be replaced by the respective activations.
  pub fn compute_cf32(&self, vals: &mut [Cf32]) {
    let act_func = self.release_func_cf32();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  /// Modifies a slice of complex values with 128-bit precision by 
  /// computing the respective complex activation option defined.
  /// 
  /// # Arguments
  /// 
  /// * `vals` - values to be replaced by the respective activations.
  pub fn compute_cf64(&self, vals: &mut [Cf64]) {
    let act_func = self.release_func_cf64();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  /// Modifies a slice of complex values with 64-bit precision by 
  /// computing the respective derivative of the complex activation option defined.
  /// 
  /// # Arguments
  /// 
  /// * `vals` - values to be replaced by the respective activations.
  pub fn compute_d_cf32(&self, vals: &mut [Cf32]) {
    let act_func = self.release_dfunc_cf32();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  /// Modifies a slice of complex values with 128-bit precision by 
  /// computing the respective derivative of the complex activation option defined.
  /// 
  /// # Arguments
  /// 
  /// * `vals` - values to be replaced by the respective activations.
  pub fn compute_d_cf64(&self, vals: &mut [Cf64]) {
    let act_func = self.release_dfunc_cf64();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  /// Modifies a slice of complex values with 64-bit precision by 
  /// computing the respective conjugate derivative of the complex activation option defined.
  /// 
  /// # Arguments
  /// 
  /// * `vals` - values to be replaced by the respective activations.
  pub fn compute_d_conj_cf32(&self, vals: &mut [Cf32]) {
    let act_func = self.release_dfunc_conj_cf32();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  /// Modifies a slice of complex values with 128-bit precision by 
  /// computing the respective conjugate derivative of the complex activation option defined.
  /// 
  /// # Arguments
  /// 
  /// * `vals` - values to be replaced by the respective activations.
  pub fn compute_d_conj_cf64(&self, vals: &mut [Cf64]) {
    let act_func = self.release_dfunc_conj_cf64();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  /* single value */

  /// Return the activation of a compelx value with 64-bit precision given current
  /// complex activation function option.
  /// 
  /// # Arguments
  /// 
  /// * `val` - value to be given as an argument replaced for the activation.
  pub fn compute_val_cf32(&self, val: &Cf32) -> Cf32 {
    let act_func = self.release_func_cf32();
    act_func(*val)
  }

  /// Return the activation of a compelx value with 128-bit precision given current
  /// complex activation function option.
  /// 
  /// # Arguments
  /// 
  /// * `val` - value to be given as an argument replaced for the activation.
  pub fn compute_val_cf64(&self, val: &Cf64) -> Cf64 {
    let act_func = self.release_func_cf64();
    act_func(*val)
  }

  /// Return the derivative of the activation of a compelx value with 64-bit precision given current
  /// complex activation function option.
  /// 
  /// # Arguments
  /// 
  /// * `val` - value to be given as an argument replaced for the activation.
  pub fn compute_d_val_cf32(&self, val: &Cf32) -> Cf32 {
    let act_func = self.release_dfunc_cf32();
    act_func(*val)
  }

  /// Return the derivative of the activation of a compelx value with 128-bit precision given current
  /// complex activation function option.
  /// 
  /// # Arguments
  /// 
  /// * `val` - value to be given as an argument replaced for the activation.
  pub fn compute_d_val_cf64(&self, val: &Cf64) -> Cf64 {
    let act_func = self.release_dfunc_cf64();
    act_func(*val)
  }

  /// Return the conjugate derivative of the activation of a compelx value with 64-bit precision given current
  /// complex activation function option.
  /// 
  /// # Arguments
  /// 
  /// * `val` - value to be given as an argument replaced for the activation.
  pub fn compute_d_conj_val_cf32(&self, val: &Cf32) -> Cf32 {
    let act_func = self.release_dfunc_conj_cf32();
    act_func(*val)
  }

  /// Return the conjugate derivative of the activation of a compelx value with 128-bit precision given current
  /// complex activation function option.
  /// 
  /// # Arguments
  /// 
  /// * `val` - value to be given as an argument replaced for the activation.
  pub fn compute_d_conj_val_cf64(&self, val: &Cf64) -> Cf64 {
    let act_func = self.release_dfunc_conj_cf64();
    act_func(*val)
  }
}
