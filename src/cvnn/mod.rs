use crate::math::{BasicOperations, Complex, Real};

pub mod layer;
pub mod network;

pub enum Criteria {
  Real,
  Imaginary,
  Norm,
  Phase,
  Log
}

pub enum ComplexCostModel {
  Conventional,
  Group
}

impl ComplexCostModel {
  /// Computes cost when target is real.
  pub fn compute<T: Complex + BasicOperations<T>>(&self, pred: &T, targ: &T::Precision, criteria: &Criteria) -> T::Precision {
    let complex_targ = T::new(*targ, T::Precision::default());
    match self {
      ComplexCostModel::Conventional => {
        match criteria {
          Criteria::Norm => { (pred.norm_sq() - *targ).pow(2) },
          Criteria::Phase => { (pred.phase() - *targ).pow(2) },
          Criteria::Real => { (pred.re() - *targ).pow(2) },
          Criteria::Imaginary => { (pred.im() - *targ).pow(2) },
          Criteria::Log => { 
            (pred.norm_sq() / complex_targ.norm_sq()).log() + 
            (pred.phase().pow(2) - complex_targ.phase().pow(2)) 
          }
        }
      }
      ComplexCostModel::Group => {
        
        match criteria {
          Criteria::Norm => { (*pred - complex_targ).norm_sq().pow(2) },
          Criteria::Phase => { (*pred - complex_targ).phase().pow(2) },
          Criteria::Real => { (*pred - complex_targ).re().pow(2) },
          Criteria::Imaginary => { (*pred - complex_targ).im().pow(2) },
          Criteria::Log => { 
            (pred.norm_sq() / complex_targ.norm_sq()).log() + 
            (pred.phase().pow(2) - complex_targ.phase().pow(2)) 
          }
        }
      }
    }
  }

  /// Computes cost when target is complex.
  pub fn compute_raw<T: Complex + BasicOperations<T>>(&self, pred: &T, targ: &T, criteria: &Criteria) -> T::Precision {
    match self {
      ComplexCostModel::Conventional => {
        match criteria {
          Criteria::Norm => { (pred.norm() - targ.norm()).pow(2) },
          Criteria::Phase => { (pred.phase() - targ.phase()).pow(2) },
          Criteria::Real => { (pred.re() - targ.re()).pow(2) },
          Criteria::Imaginary => { (pred.im() - targ.im()).pow(2) },
          Criteria::Log => { 
            (pred.norm_sq() / targ.norm_sq()).log() + 
            (pred.phase().pow(2) - targ.phase().pow(2)) 
          }
        }
      }
      ComplexCostModel::Group => {
        match criteria {
          Criteria::Norm => { (*pred - *targ).norm_sq() },
          Criteria::Phase => { (*pred - *targ).phase().pow(2) },
          Criteria::Real => { (*pred - *targ).re().pow(2) },
          Criteria::Imaginary => { (*pred - *targ).im().pow(2) },
          Criteria::Log => { 
            (pred.norm_sq() / targ.norm_sq()).log() + 
            (pred.phase().pow(2) - targ.phase().pow(2)) 
          }
        }
      }
    }
  }
}