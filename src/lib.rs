pub mod math;
pub mod dataset;
pub mod input;
pub mod act;
pub mod opt;
pub mod init;
pub mod err;
pub mod rvnn;
pub mod cvnn;

#[cfg(test)]
mod basic_tests {

  use super::*;

  #[test]
  fn net_ops() {
    use init::InitMethod;
    use rvnn::layer::LayerLike;
    use rvnn::layer::dense::DenseLayer;
    use act::ActFunc;
    use input::{IOShape, IOType};
    use rvnn::network::Network;

    let ref mut seed = 43827992_u128;
    
    let mut net = Network::<f32>::new();

    println!("{:?}", net);

    net.add_input(
      /* layer to be added (as input) */
      DenseLayer::new(ActFunc::Sigmoid).wrap(),
      /* each neuron has 2 inputs */
      IOShape::Vector(2),
      /* 4 neurons in total */
      4,
      /* random initialization (with scale) */
      InitMethod::Random(4), 
      seed
    ).unwrap();

    net.add(
      /* layer to be added (as hidden) */
      DenseLayer::new(ActFunc::Sigmoid).wrap(), 
      2, 
      InitMethod::Random(4), 
      seed
    ).unwrap();

    println!("{:?}", net);

    let out = net.forward(IOType::Vector(vec![1.0; 2 * 4])).unwrap();

    println!("{:?}", out);

    use math::cfloat::Cf64;
    use cvnn::layer::CLayerLike;
    use cvnn::layer::dense::DenseCLayer;
    use act::ComplexActFunc;
    use cvnn::network::CNetwork;

    let ref mut seed = 43827992_u128;
    
    let mut net = CNetwork::<Cf64>::new();

    println!("{:?}", net);

    net.add_input(
      /* layer to be added (as input) */
      DenseCLayer::new(ComplexActFunc::RITSigmoid).wrap(),
      /* each neuron has 2 inputs */
      IOShape::Vector(2),
      /* 4 neurons in total */
      4,
      /* random initialization (with scale) */
      InitMethod::Random(4), 
      seed
    ).unwrap();

    net.add(
      /* layer to be added (as hidden) */
      DenseCLayer::new(ComplexActFunc::RITSigmoid).wrap(), 
      2, 
      InitMethod::Random(4), 
      seed
    ).unwrap();

    println!("{:?}", net);

    let out = net.forward(IOType::Vector(vec![Cf64 { x: 1.0, y: 1.0 }; 2 * 4])).unwrap();

    println!("{:?}", out);

  }
}