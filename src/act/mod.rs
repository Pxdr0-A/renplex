use crate::math::{Complex, Real};

#[derive(Debug)]
pub enum ActError {
  UnimplementedAct
}

pub enum ActFunc {
  Sigmoid,
  Tanh
}

impl ActFunc {
  pub fn act<T: Real + Copy>(&self, values: &mut [T]) -> Result<(), ActError> {
    match self {
      ActFunc::Sigmoid => { 
        for elm in values.iter_mut() {
          *elm = elm.sigmoid();
        }

        Ok(())
      },
      _ => { Err(ActError::UnimplementedAct) }
    }
  }
}

pub enum ComplexActFunc {
  RITSigmoid,
  RITTanh
}

impl ComplexActFunc {
  pub fn act<T: Complex + Copy>(&self, values: &mut [T]) -> Result<(), ActError> {
    match self {
      ComplexActFunc::RITSigmoid => {
        for elm in values.iter_mut() {
          *elm = elm.rit_sigmoid();
        }

        Ok(())
      },
      _ => { Err(ActError::UnimplementedAct) }
    }
  }
}