pub enum LossFunc {
  Conventional
}

pub enum ComplexCriteria {
  Real,
  Imaginary,
  Norm,
  Phase
}

pub enum ComplexLossFunc {
  Conventional,
  Log
}

#[derive(Debug)]
pub enum GradientError {
  InconsistentShape,
  Unimplemented
}
