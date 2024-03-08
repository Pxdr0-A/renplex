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
  
  #[test]
  fn cost_test() {
    use dataset::Dataset;
    use init::PredictModel;
    use rvnn::network::Network;
    use rvnn::layer::dense::DenseLayer;
    use rvnn::layer::LayerLike;
    use act::ActFunc;
    use input::IOShape;
    use init::InitMethod;

    let ref mut seed = 182756_u128;

    let n_input_dendrits: usize = 2;
    let n_input_units: usize = 2;
    let input_len = n_input_dendrits * n_input_units;
    let scale: usize = 4;

    let data: Dataset<f32, f32> = Dataset::sample(
      [64, input_len], 
      3, 
      100, 
      10, 
      PredictModel::Sparse, 
      seed
    ).unwrap();
    data.to_csv().unwrap();

    let mut net: Network<f32> = Network::new();
    net.add_input(
      DenseLayer::new(ActFunc::Sigmoid).wrap(), 
      IOShape::Vector(n_input_dendrits), 
      n_input_units,
      InitMethod::Random(scale), 
      seed
    ).unwrap();
    net.add(
      DenseLayer::new(ActFunc::Sigmoid).wrap(), 
      16,
      InitMethod::Random(scale), 
      seed
    ).unwrap();
    net.add(
      DenseLayer::new(ActFunc::Sigmoid).wrap(), 
      64, 
      InitMethod::Random(scale), 
      seed
    ).unwrap();
    net.add(
      DenseLayer::new(ActFunc::Sigmoid).wrap(), 
      4, 
      InitMethod::Random(scale), 
      seed
    ).unwrap();
  }

  #[test]
  fn sigmoid_test() {
    use math::Real;

    let a: f32= 220.33234;

    println!("{}", a.sigmoid());
  }

  #[test]
  fn dataset_tests() {
    use dataset::Dataset;
    use init::PredictModel;
    use math::cfloat::Cf32;

    let ref mut seed = 9891827_u128;

    let data: Dataset<Cf32, f32> = Dataset::sample_complex(
      [64, 2], 
      3, 
      100, 
      10, 
      PredictModel::Sparse, 
      seed
    ).unwrap();

    println!("{:?}", data);
  }

  #[test]
  fn dataset_to_csv_test() {
    use dataset::Dataset;
    use init::PredictModel;

    let ref mut seed = 119824653_u128;

    let data: Dataset<f32, f32> = Dataset::sample(
      [128, 2], 
      4, 
      100, 
      10, 
      PredictModel::Sparse, 
      seed
    ).unwrap();

    data.to_csv().unwrap();
  }
}