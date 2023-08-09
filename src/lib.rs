pub mod prelude;
pub mod math;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn generic_value_activation() {
    // temporary test
    // afterwards change vars to private
    use prelude::neuron::activation::Activation;
    use prelude::neuron::activation::ActivationFunction;

    let a: f32 = 0.34;
    let func1 = ActivationFunction::SIGMOID;

    let result1 = a.activation(func1);
    println!("{}", result1);

    let b: f64 = 0.34;
    let func2 = ActivationFunction::TANH;

    let result2 = b.activation(func2);
    println!("{}", result2);

    let c: f32 = 4.0;
    let func3 = ActivationFunction::RELU;

    let result3 = c.activation(func3);
    println!("{}", result3);

    let d: f32 = -0.23;
    let func4 = ActivationFunction::RELU;

    let result4 = d.activation(func4);
    println!("{}", result4);
  } 

  #[test]
  fn complex_number_mul() {
    use math::Cfloat;

    let a = Cfloat::new(0.34, 0.75);
    let a_rhs = Cfloat::new(0.45, 0.05);
    let b = Cfloat::new(0.81f32, 2.75f32);
    let b_rhs = Cfloat::new(0.12f32, -1.0f32);

    println!("{:?}", a * a_rhs);
    println!("{:?}", b * b_rhs);
  }
}
