pub enum ActivationFunction {
  SIGMOID,
  TANH,
  RELU
}

pub trait Activation {
  fn activation(&self, func: ActivationFunction) -> Self;
}

macro_rules! activate_me {
  ($value:ident, $func:ident) => {
    match $func {
      ActivationFunction::SIGMOID => {
        1.0 / (1.0 + (-$value).exp())
      },
      ActivationFunction::TANH => {
        $value.tanh()
      },
      ActivationFunction::RELU => {
        if $value.is_sign_positive() { *$value } else { 0.0 }
      }
    } 
  };
}

// To implement Activation to a custom type, those types need to have
// minus trait (negative value)
// exp(), tanh(), is_sign_positive() methods.

// UPDATE THIS CODE TO USE A MACRO!
impl Activation for f32 {
  fn activation(&self, func: ActivationFunction) -> f32 {
    activate_me!(self, func)
  }
}

impl Activation for f64 {
  fn activation(&self, func: ActivationFunction) -> f64 {
    activate_me!(self, func)
  }
}

// Activation implementation for Cfloat (for f32 and f64)