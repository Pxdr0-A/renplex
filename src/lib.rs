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

  use std::time::Duration;
  use std::thread;
  use std::io::Write;
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

  #[test]
  fn gradient_descent_test() {
    use dataset::Dataset;
    use init::PredictModel;
    use rvnn::network::Network;
    use rvnn::layer::dense::DenseLayer;
    use rvnn::layer::LayerLike;
    use act::ActFunc;
    use input::IOShape;
    use init::InitMethod;
    use opt::LossFunc;

    let ref mut seed = 91239812_u128;

    let n_input_dendrits: usize = 2;
    let n_input_units: usize = 2;
    let input_len = n_input_dendrits * n_input_units;
    let scale: usize = 4;

    let data: Dataset<f32, f32> = Dataset::sample(
      [128, input_len], 
      3, 
      100, 
      10, 
      PredictModel::Sparse, 
      seed
    ).unwrap();

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
      3,
      InitMethod::Random(scale), 
      seed
    ).unwrap();

    let (loss, _) = net.loss(data.clone(), &LossFunc::Conventional).unwrap();
    println!("{:?}", loss);

    let n_batches: usize = 128;
    for _ in 0..n_batches {
      net.gradient_opt(data.clone(), LossFunc::Conventional, 10e-2).unwrap();

      let (loss, _) = net.loss(data.clone(), &LossFunc::Conventional).unwrap();
      println!("{:?}", loss);
    }

  }

  #[test]
  fn misc() {
    let total_iterations = 100;
    let mut progress = 0;

    // Simulate a time-consuming task
    for _ in 0..total_iterations {
        // Do some work
        thread::sleep(Duration::from_millis(50));

        // Update progress
        progress += 1;

        // Calculate percentage completion
        let percentage = progress as f32 / total_iterations as f32 * 100.0;

        // Print progress bar
        print!("\r[");
        for j in 0..50 {
            if (j as f32) < percentage / 2.0 {
                print!("•");
            } else {
                print!("-");
            }
        }
        print!("] {:.2}% ", percentage);

        // Flush output to ensure immediate display
        std::io::stdout().flush().unwrap();
    }

    // Print new line to separate progress bar from loss information
    println!();

    // Print loss information
    println!("Loss: {:.4}", 0.1234);
  }
}