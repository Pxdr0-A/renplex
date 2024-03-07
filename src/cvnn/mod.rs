pub mod layer;
pub mod network;

pub enum Criteria {
  Real,
  Imaginary,
  Norm,
  Phase
}

pub enum ComplexCostModel {
  Conventional,
  Group
}