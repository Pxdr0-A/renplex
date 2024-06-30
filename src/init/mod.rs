use crate::math::Complex;

/// List of initialization methods that can be applied to the initialization of a layer.
pub enum InitMethod {
  /// Takes a scale param
  Uniform(usize),
  /// Takes the number of input + output units
  XavierGlorotU(usize),
  /// Takes the number of input + output units
  XavierGlorotN(usize),
  /// Takes the number of input units
  Xavier(usize),
  /// Takes the number of input units
  HeInit(usize),
}

impl InitMethod {
  /// Generates a random number according to a distribution defined in the enumeration [`InitMethod`].
  pub fn gen<T: Complex>(&self, seed: &mut u128) -> T {
    match self {
      Self::Uniform(scale) => {
        T::gen(seed, *scale)
      }
      Self::XavierGlorotU(io_units) => {
        T::gen_xagu(seed, *io_units)
      },
      Self::XavierGlorotN(io_units) => {
        T::gen_xagn(seed, *io_units)
      },
      Self::Xavier(i_units) => {
        T::gen_xa(seed, *i_units)
      },
      Self::HeInit(i_units) => {
        T::gen_he(seed, *i_units)
      }
    }
  }
}

/// Ways for generating a prediction for classification models. For now it is just one-hot-encoding.
pub enum PredictModel {
  Sparse
}