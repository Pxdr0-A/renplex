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
  use crate::act::ComplexActFunc;
  use crate::cvnn::layer::dense::DenseCLayer;
  use crate::cvnn::network::CNetwork;
  use crate::math::matrix::Matrix;
  use crate::opt::ComplexLossFunc;

use self::math::cfloat::Cf32;
use self::math::Complex;

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

    let ref mut seed = 98897867321_u128;

    let n_input_dendrits: usize = 2;
    let n_input_units: usize = 2;
    let input_len = n_input_dendrits * n_input_units;
    let scale: usize = 1;
    let n_batches: usize = 1024;
    let total_iterations = n_batches;
    let mut progress: usize = 0;
    let degree = 3;

    let data: Dataset<f32, f32> = Dataset::sample(
      [128, input_len], 
      degree, 
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
      degree,
      InitMethod::Random(scale), 
      seed
    ).unwrap();

    let (loss, _) = net.loss(data.clone(), &LossFunc::Conventional).unwrap();
    println!("{:?}", loss);

    let mut mean_loss_vec = Vec::new();
    mean_loss_vec.push(loss);

    for _ in 0..n_batches {
      net.gradient_opt(data.clone(), LossFunc::Conventional, 10e-2).unwrap();

      let (loss, _) = net.loss(data.clone(), &LossFunc::Conventional).unwrap();
      mean_loss_vec.push(loss);

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
              print!(" ");
          }
      }
      print!("] loss: {:.3}", loss);
      // Flush output to ensure immediate display
      std::io::stdout().flush().unwrap();
    }

    println!();

    let rows = mean_loss_vec.len();
    Matrix::from_body(mean_loss_vec, [rows, 1]).to_csv().unwrap();
  }

  #[test]
  fn complex_gradient_descent_test() {
    use dataset::Dataset;
    use init::PredictModel;
    use input::IOShape;
    use init::InitMethod;
    use cvnn::layer::CLayerLike;
    use math::cfloat::Cf32;
    use math::Complex;

    let ref mut seed = 10101010189_u128;

    let n_input_dendrits: usize = 2;
    let n_input_units: usize = 3;
    let degree: usize = 4;
    let input_len = n_input_dendrits * n_input_units;
    let scale: usize = 1;
    let batch_size: usize = 128;
    let n_batches: usize = 2000;
    let total_iterations = n_batches;
    let mut progress: usize = 0;

    let data: Dataset<Cf32, Cf32> = Dataset::sample_complex(
      [batch_size, input_len], 
      degree, 
      100, 
      10, 
      PredictModel::Sparse, 
      seed
    ).unwrap();
    data.to_csv().unwrap();

    let mut net: CNetwork<Cf32> = CNetwork::new();
    net.add_input(
      DenseCLayer::new(ComplexActFunc::RITSigmoid).wrap(), 
      IOShape::Vector(n_input_dendrits), 
      n_input_units,
      InitMethod::Random(scale), 
      seed
    ).unwrap();
    net.add(
      DenseCLayer::new(ComplexActFunc::RITSigmoid).wrap(), 
      8,
      InitMethod::Random(scale), 
      seed
    ).unwrap();
    net.add(
      DenseCLayer::new(ComplexActFunc::RITSigmoid).wrap(), 
      degree,
      InitMethod::Random(scale), 
      seed
    ).unwrap();

    let (loss, _) = net.loss(data.clone(), &ComplexLossFunc::Conventional).unwrap();
    let mut mean_loss_vec = Vec::new();

    mean_loss_vec.push(loss);
    for _ in 0..n_batches {
      net.gradient_opt(data.clone(), ComplexLossFunc::Conventional, Cf32::new(10e-3, 0.0)).unwrap();

      let (loss, _) = net.loss(data.clone(), &ComplexLossFunc::Conventional).unwrap();
      mean_loss_vec.push(loss);

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
              print!(" ");
          }
      }
      print!("] loss: {:.3}", loss);
      // Flush output to ensure immediate display
      std::io::stdout().flush().unwrap();
    }
    println!();

    let rows = mean_loss_vec.len();
    Matrix::from_body(mean_loss_vec, [rows, 1]).to_csv().unwrap();
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

    let num = 3.14159265359; // Your float number
    let precision = 2; // Number of decimal places to round to
    
    let multiplier = 10_f64.powi(precision);
    let rounded = (num * multiplier).round() / multiplier;
    
    println!("Rounded number: {}", rounded)
  }

  #[test]
  fn complex_test() {
    let a = Cf32::new(2.3, 6.1);
    let r = Cf32::new(2.9, 6.2);

    let res1 = a * a.conj() + r * r.conj() - a * r.conj() - r * a.conj();
    let res2 = (a - r).norm_sq();

    println!("{} | {}", res1, res2);
  }
}