pub mod prelude;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn value_activation() {
    // temporary test
    // afterwards change vars to private
    use prelude::neuron::activation::Activation;
    use prelude::neuron::activation::ActivationFunction;

    let a: f32 = 0.34;
    let func = ActivationFunction::SIGMOID;

    let result = a.activation(func);
    println!("{}", result);
  } 
}
