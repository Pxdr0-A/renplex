use crate::math::cfloat::{Cf32, Cf64};

/* Activation functions. */
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

fn d_ritsigmoid_cf32(val: Cf32) -> Cf32 {
  Cf32 {
    x: d_sigmoid_f32(val.x), 
    y: d_sigmoid_f32(val.y)
  }
}

fn d_ritsigmoid_cf64(val: Cf64) -> Cf64 {
  Cf64 {
    x: d_sigmoid_f64(val.x), 
    y: d_sigmoid_f64(val.y)
  }
}

#[derive(Debug)]
pub enum ActError {}

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

#[derive(Debug)]
pub enum ComplexActFunc {
  RITSigmoid
}

impl ComplexActFunc {
  pub fn compute_cf32(&self, vals: &mut [Cf32]) {
    let act_func = match self {
      ComplexActFunc::RITSigmoid => {
        ritsigmoid_cf32
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  pub fn compute_cf64(&self, vals: &mut [Cf64]) {
    let act_func = match self {
      ComplexActFunc::RITSigmoid => {
        ritsigmoid_cf64
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  pub fn compute_d_cf32(&self, vals: &mut [Cf32]) {
    let act_func = match self {
      ComplexActFunc::RITSigmoid => {
        d_ritsigmoid_cf32
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }

  pub fn compute_d_cf64(&self, vals: &mut [Cf64]) {
    let act_func = match self {
      ComplexActFunc::RITSigmoid => {
        d_ritsigmoid_cf64
      }
    };

    for val in vals.iter_mut() {
      *val = act_func(*val);
    }
  }
}
