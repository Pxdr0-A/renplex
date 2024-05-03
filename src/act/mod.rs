use crate::math::cfloat::{Cf32, Cf64};

/* Activation functions. */

/* Sigmoid */
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

/* Real Imaginary Type Sigmoid */
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
    x: ( d_relu_f32(val.x) + d_relu_f32(val.x) ) * 0.5,
    y: 0.0
  }
}

fn d_ritrelu_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: ( d_relu_f64(val.x) + d_relu_f64(val.x) ) * 0.5,
    y: 0.0
  }
}

fn d_conj_ritrelu_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: ( d_relu_f32(val.x) - d_relu_f32(val.x) ) * 0.5,
    y: 0.0
  }
}

fn d_conj_ritrelu_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: ( d_relu_f64(val.x) - d_relu_f64(val.x) ) * 0.5,
    y: 0.0
  }
}


#[derive(Debug)]
pub enum ActFunc {
  Sigmoid
}

impl ActFunc {
  pub fn compute_f32(&self, vals: &mut [f32]) {
    let act_func = match self {
      ActFunc::Sigmoid => {
        sigmoid_f32
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  pub fn compute_f64(&self, vals: &mut [f64]) {
    let act_func = match self {
      ActFunc::Sigmoid => {
        sigmoid_f64
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  pub fn compute_d_f32(&self, vals: &mut [f32]) {
    let act_func = match self {
      ActFunc::Sigmoid => {
        d_sigmoid_f32
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  pub fn compute_d_f64(&self, vals: &mut [f64]) {
    let act_func = match self {
      ActFunc::Sigmoid => {
        d_sigmoid_f64
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub enum ComplexActFunc {
  RITSigmoid,
  RITReLU
}

impl ComplexActFunc {
  pub fn release_func_cf32(&self) -> fn(Cf32) -> Cf32 {
    match self {
      ComplexActFunc::RITSigmoid => {
        ritsigmoid_cf32
      },
      ComplexActFunc::RITReLU => {
        ritrelu_cf32
      }
    }
  }

  pub fn release_func_cf64(&self) -> fn(Cf64) -> Cf64 {
    match self {
      ComplexActFunc::RITSigmoid => {
        ritsigmoid_cf64
      },
      ComplexActFunc::RITReLU => {
        ritrelu_cf64
      }
    }
  }

  pub fn release_dfunc_cf32(&self) -> fn(Cf32) -> Cf32 {
    match self {
      ComplexActFunc::RITSigmoid => {
        d_ritsigmoid_cf32
      },
      ComplexActFunc::RITReLU => {
        d_ritrelu_cf32
      }
    }
  }

  pub fn release_dfunc_cf64(&self) -> fn(Cf64) -> Cf64 {
    match self {
      ComplexActFunc::RITSigmoid => {
        d_ritsigmoid_cf64
      },
      ComplexActFunc::RITReLU => {
        d_ritrelu_cf64
      }
    }
  }

  pub fn release_dfunc_conj_cf32(&self) -> fn(Cf32) -> Cf32 {
    match self {
      ComplexActFunc::RITSigmoid => {
        d_conj_ritsigmoid_cf32
      },
      ComplexActFunc::RITReLU => {
        d_conj_ritrelu_cf32
      }
    }
  }

  pub fn release_dfunc_conj_cf64(&self) -> fn(Cf64) -> Cf64 {
    match self {
      ComplexActFunc::RITSigmoid => {
        d_conj_ritsigmoid_cf64
      },
      ComplexActFunc::RITReLU => {
        d_conj_ritrelu_cf64
      }
    }
  }

  /* multiple value */

  pub fn compute_cf32(&self, vals: &mut [Cf32]) {
    let act_func = self.release_func_cf32();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  pub fn compute_cf64(&self, vals: &mut [Cf64]) {
    let act_func = self.release_func_cf64();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  pub fn compute_d_cf32(&self, vals: &mut [Cf32]) {
    let act_func = self.release_dfunc_cf32();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  pub fn compute_d_cf64(&self, vals: &mut [Cf64]) {
    let act_func = self.release_dfunc_cf64();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  pub fn compute_d_conj_cf32(&self, vals: &mut [Cf32]) {
    let act_func = self.release_dfunc_conj_cf32();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  pub fn compute_d_conj_cf64(&self, vals: &mut [Cf64]) {
    let act_func = self.release_dfunc_conj_cf64();
    vals.iter_mut().for_each(|val| {*val = act_func(*val);});
  }

  /* single value */

  pub fn compute_val_cf32(&self, val: &Cf32) -> Cf32 {
    let act_func = self.release_func_cf32();
    act_func(*val)
  }

  pub fn compute_val_cf64(&self, val: &Cf64) -> Cf64 {
    let act_func = self.release_func_cf64();
    act_func(*val)
  }

  pub fn compute_d_val_cf32(&self, val: &Cf32) -> Cf32 {
    let act_func = self.release_dfunc_cf32();
    act_func(*val)
  }

  pub fn compute_d_val_cf64(&self, val: &Cf64) -> Cf64 {
    let act_func = self.release_dfunc_cf64();
    act_func(*val)
  }

  pub fn compute_d_conj_val_cf32(&self, val: &Cf32) -> Cf32 {
    let act_func = self.release_dfunc_conj_cf32();
    act_func(*val)
  }

  pub fn compute_d_conj_val_cf64(&self, val: &Cf64) -> Cf64 {
    let act_func = self.release_dfunc_conj_cf64();
    act_func(*val)
  }
}
