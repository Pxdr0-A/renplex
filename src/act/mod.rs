use crate::math::{Complex, Real};

#[derive(Debug)]
pub enum ActError {}

#[derive(Debug)]
pub enum ActFunc {
  Sigmoid
}

impl ActFunc {
  pub fn compute<T: Real + Copy>(&self, values: &mut [T]) -> Result<(), ActError> {
    match self {
      ActFunc::Sigmoid => { 
        for elm in values.iter_mut() {
          *elm = elm.sigmoid();
        }

        Ok(())
      }
    }
  }
}

#[derive(Debug)]
pub enum ComplexActFunc {
  RITSigmoid
}

impl ComplexActFunc {
  pub fn compute<T: Complex + Copy>(&self, values: &mut [T]) -> Result<(), ActError> {
    match self {
      ComplexActFunc::RITSigmoid => {
        for elm in values.iter_mut() {
          *elm = elm.rit_sigmoid();
        }

        Ok(())
      }
    }
  }
}
