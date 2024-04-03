use std::io::Write;

pub mod math;
pub mod dataset;
pub mod input;
pub mod act;
pub mod opt;
pub mod init;
pub mod err;
pub mod rvnn;
pub mod cvnn;

pub fn update_progress(progress: &mut usize, total_iterations: usize, loss: f32) {
  // Update progress
  *progress += 1;
  // Calculate percentage completion
  let percentage = *progress as f32 / total_iterations as f32 * 100.0;
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

#[cfg(test)]
mod basic_tests {

  use std::fs::File;
use std::time::Duration;
  use std::thread;
  use std::io::{Read, Write};
  use crate::act::ComplexActFunc;
  use crate::cvnn::layer::dense::DenseCLayer;
  use crate::cvnn::layer::CLayerLike;
use crate::cvnn::network::CNetwork;
  use crate::math::matrix::Matrix;
  use crate::opt::{ComplexLossFunc, LossFunc};

use self::act::ActFunc;
use self::dataset::Dataset;
use self::init::InitMethod;
use self::input::IOShape;
use self::math::cfloat::Cf32;
use self::math::Complex;
use self::rvnn::layer::dense::DenseLayer;
use self::rvnn::layer::LayerLike;
use self::rvnn::network::Network;

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

    /* test for overfitting */

    let ref mut seed = 28233267845_u128;

    let n_input_dendrits: usize = 2;
    let n_input_units: usize = 2;
    let input_len = n_input_dendrits * n_input_units;
    let scale: usize = 1;
    let batch_size: usize = 128;
    let n_batches: usize = 2000;
    let lr = 10e-2;
    let total_iterations = n_batches;
    let mut progress: usize = 0;
    let degree = 3;

    let data: Dataset<f32, f32> = Dataset::sample(
      [batch_size, input_len], 
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

    println!("{:?}", net.max_pred_test(data.clone()));

    for _ in 0..n_batches {
      net.gradient_opt(data.clone(), LossFunc::Conventional, lr).unwrap();

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
    println!("{:?}", net.max_pred_test(data.clone()));

    let rows = mean_loss_vec.len();
    Matrix::from_body(mean_loss_vec, [rows, 1]).to_csv("out/matrix.csv").unwrap();
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

    /* test for overfitting */

    let ref mut seed = 834893486234672_u128;

    let n_input_dendrits: usize = 2;
    let n_input_units: usize = 2;
    let degree: usize = 3;
    let input_len = n_input_dendrits * n_input_units;
    let scale: usize = 1;
    let batch_size: usize = 128;
    let n_batches: usize = 2000;
    let total_iterations = n_batches;
    let lr = Cf32::new(10e-2, 0.0);
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
      16,
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

    println!("{:?}", net.max_pred_test(data.clone()));

    for _ in 0..n_batches {
      net.gradient_opt(data.clone(), ComplexLossFunc::Conventional, lr).unwrap();

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
    println!("{:?}", net.max_pred_test(data.clone()));

    let rows = mean_loss_vec.len();
    Matrix::from_body(mean_loss_vec, [rows, 1]).to_csv("./out/matrix.csv").unwrap();
  }

  #[test]
  fn minist_train_test() {
    let ref mut seed = 238348892932_u128;

    let n_input_dendrits: usize = 28;
    let n_input_units: usize = 28;
    let scale: usize = 1;
    let batch_size = 100;
    let n_points: usize = 60000;
    let n_test_points: usize = 10000;
    let n_batches: usize = n_points / batch_size;
    let n_test_batches: usize = n_test_points / batch_size;
    let epochs: usize = 100;
    let lr = Cf32::new(10e-2, 0.0);
    let degree = 10;

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
      16,
      InitMethod::Random(scale), 
      seed
    ).unwrap();
    net.add(
      DenseCLayer::new(ComplexActFunc::RITSigmoid).wrap(), 
      16,
      InitMethod::Random(scale), 
      seed
    ).unwrap();
    net.add(
      DenseCLayer::new(ComplexActFunc::RITSigmoid).wrap(), 
      degree,
      InitMethod::Random(scale), 
      seed
    ).unwrap();

    /* CHECK THE NORMALIZATION OF THE DATA OR MAYBE EVEN TRY TO RECONSTRUCT THE IMAGES */

    let mut train_data_file = File::open("./minist/train-images.idx3-ubyte").unwrap();
    let mut train_label_file = File::open("./minist/train-labels.idx1-ubyte").unwrap();
    let mut tracker = 0_usize;

    let mut mean_loss = 0.0;
    let mut mean_acc = 0.0;

    let mut train_loss = Vec::new();
    let mut test_loss = Vec::new();
    let mut train_acc = Vec::new();
    let mut test_acc = Vec::new();

    /* initial test on entire train dataset */
    for _ in 0..n_batches {
      let minist_data_batch: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(&mut train_data_file, &mut train_label_file, batch_size, &mut tracker);
      let ( loss, _) = net.loss(minist_data_batch.clone(), &ComplexLossFunc::Conventional).unwrap();
      let acc = net.max_pred_test(minist_data_batch);

      mean_loss += loss;
      mean_acc += acc;
    }
  
    mean_loss /= n_batches as f32;
    mean_acc /= n_batches as f32;
    train_loss.push(mean_loss);
    train_acc.push(mean_acc);
    println!("Initial Train Values -> Loss: {:.3}, Acc: {:.3}", mean_loss, mean_acc);

    /* initial test on entire test dataset */
    let mut test_data_file = File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let  mut test_label_file = File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    tracker = 0_usize;

    mean_loss = 0.0;
    mean_acc = 0.0;
    for _ in 0..n_test_batches {
      let minist_data_batch = Dataset::minist_as_complex_batch(&mut test_data_file, &mut test_label_file, batch_size, &mut tracker);
      let (loss_per_epoch, _) = net.loss(minist_data_batch.clone(), &ComplexLossFunc::Conventional).unwrap();
      let acc = net.max_pred_test(minist_data_batch);

      mean_loss += loss_per_epoch;
      mean_acc += acc;
    }

    mean_loss /= n_test_batches as f32;
    mean_acc /= n_test_batches as f32;
    test_loss.push(mean_loss);
    test_acc.push(mean_acc);
    println!("Initial Test Values -> Loss: {:.3}, Acc: {:.3}", mean_loss, mean_acc);

    /* begin training process */
    for e in 0..epochs {
      train_data_file = File::open("./minist/train-images.idx3-ubyte").unwrap();
      train_label_file = File::open("./minist/train-labels.idx1-ubyte").unwrap();
      tracker = 0_usize;

      mean_loss = 0.0;
      mean_acc = 0.0;

      /* train */
      for _ in 0..n_batches {
        let minist_data_batch = Dataset::minist_as_complex_batch(&mut train_data_file, &mut train_label_file, batch_size, &mut tracker);

        net.gradient_opt(minist_data_batch.clone(), ComplexLossFunc::Conventional, lr).unwrap();
        
        let (loss_per_epoch, _) = net.loss(minist_data_batch.clone(), &ComplexLossFunc::Conventional).unwrap();
        let acc = net.max_pred_test(minist_data_batch);
        mean_loss += loss_per_epoch;
        mean_acc += acc;
      }

      mean_loss /= n_batches as f32;
      mean_acc /= n_batches as f32;
      train_loss.push(mean_loss);
      train_acc.push(mean_acc);
      println!("Train Epoch {} -> Loss: {:.3}, Acc: {:.3}", e+1, mean_loss, mean_acc);

      /* test */
      test_data_file = File::open("./minist/t10k-images.idx3-ubyte").unwrap();
      test_label_file = File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
      tracker = 0_usize;

      mean_loss = 0.0;
      mean_acc = 0.0;
      /* test the results of the epoch */
      for _ in 0..n_test_batches {
        let minist_data_batch = Dataset::minist_as_complex_batch(&mut test_data_file, &mut test_label_file, batch_size, &mut tracker);      
        let (loss_per_epoch, _) = net.loss(minist_data_batch.clone(), &ComplexLossFunc::Conventional).unwrap();
        let acc = net.max_pred_test(minist_data_batch);

        mean_loss += loss_per_epoch;
        mean_acc += acc;
      }

      mean_loss /= n_test_batches as f32;
      mean_acc /= n_test_batches as f32;
      test_loss.push(mean_loss);
      test_acc.push(mean_acc);
      println!("Test Epoch {} -> Loss: {:.3}, Acc: {:.3}", e+1, mean_loss, mean_acc);
    }

    let train_epochs = train_loss.len();
    let test_epochs = test_loss.len();
    Matrix::from_body(train_loss, [train_epochs, 1]).to_csv("./out/clc_train_loss.csv").unwrap();
    Matrix::from_body(train_acc, [train_epochs, 1]).to_csv("./out/clc_train_acc.csv").unwrap();
    Matrix::from_body(test_loss, [test_epochs, 1]).to_csv("./out/clc_test_loss.csv").unwrap();
    Matrix::from_body(test_acc, [test_epochs, 1]).to_csv("./out/clc_test_acc.csv").unwrap();
  }

  #[test]
  fn complex_minist_train_test() {
    let ref mut seed = 238348892932_u128;

    let n_input_dendrits: usize = 28;
    let n_input_units: usize = 28;
    let scale: usize = 1;
    let batch_size = 100;
    let n_points: usize = 60000;
    let n_test_points: usize = 10000;
    let n_batches: usize = n_points / batch_size;
    let n_test_batches: usize = n_test_points / batch_size;
    let epochs: usize = 100;
    let lr = 10e-2;
    let degree = 10;

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

    /* CHECK THE NORMALIZATION OF THE DATA OR MAYBE EVEN TRY TO RECONSTRUCT THE IMAGES */

    let mut train_data_file = File::open("./minist/train-images.idx3-ubyte").unwrap();
    let mut train_label_file = File::open("./minist/train-labels.idx1-ubyte").unwrap();
    let mut tracker = 0_usize;

    let mut mean_loss = 0.0;
    let mut mean_acc = 0.0;

    let mut train_loss = Vec::new();
    let mut test_loss = Vec::new();
    let mut train_acc = Vec::new();
    let mut test_acc = Vec::new();

    /* initial test on entire train dataset */
    for _ in 0..n_batches {
      let minist_data_batch: Dataset<f32, f32> = Dataset::minist_as_batch(&mut train_data_file, &mut train_label_file, batch_size, &mut tracker);
      let ( loss, _) = net.loss(minist_data_batch.clone(), &LossFunc::Conventional).unwrap();
      let acc = net.max_pred_test(minist_data_batch);

      mean_loss += loss;
      mean_acc += acc;
    }
  
    mean_loss /= n_batches as f32;
    mean_acc /= n_batches as f32;
    train_loss.push(mean_loss);
    train_acc.push(mean_acc);
    println!("Initial Train Values -> Loss: {:.3}, Acc: {:.3}", mean_loss, mean_acc);

    /* initial test on entire test dataset */
    let mut test_data_file = File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let  mut test_label_file = File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    tracker = 0_usize;

    mean_loss = 0.0;
    mean_acc = 0.0;
    for _ in 0..n_test_batches {
      let minist_data_batch = Dataset::minist_as_batch(&mut test_data_file, &mut test_label_file, batch_size, &mut tracker);
      let (loss_per_epoch, _) = net.loss(minist_data_batch.clone(), &LossFunc::Conventional).unwrap();
      let acc = net.max_pred_test(minist_data_batch);

      mean_loss += loss_per_epoch;
      mean_acc += acc;
    }

    mean_loss /= n_test_batches as f32;
    mean_acc /= n_test_batches as f32;
    test_loss.push(mean_loss);
    test_acc.push(mean_acc);
    println!("Initial Test Values -> Loss: {:.3}, Acc: {:.3}", mean_loss, mean_acc);

    /* begin training process */
    for e in 0..epochs {
      train_data_file = File::open("./minist/train-images.idx3-ubyte").unwrap();
      train_label_file = File::open("./minist/train-labels.idx1-ubyte").unwrap();
      tracker = 0_usize;

      mean_loss = 0.0;
      mean_acc = 0.0;

      /* train */
      for _ in 0..n_batches {
        let minist_data_batch = Dataset::minist_as_batch(&mut train_data_file, &mut train_label_file, batch_size, &mut tracker);

        net.gradient_opt(minist_data_batch.clone(), LossFunc::Conventional, lr).unwrap();
        
        let (loss_per_epoch, _) = net.loss(minist_data_batch.clone(), &LossFunc::Conventional).unwrap();
        let acc = net.max_pred_test(minist_data_batch);
        mean_loss += loss_per_epoch;
        mean_acc += acc;
      }

      mean_loss /= n_batches as f32;
      mean_acc /= n_batches as f32;
      train_loss.push(mean_loss);
      train_acc.push(mean_acc);
      println!("Train Epoch {} -> Loss: {:.3}, Acc: {:.3}", e+1, mean_loss, mean_acc);

      /* test */
      test_data_file = File::open("./minist/t10k-images.idx3-ubyte").unwrap();
      test_label_file = File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
      tracker = 0_usize;

      mean_loss = 0.0;
      mean_acc = 0.0;
      /* test the results of the epoch */
      for _ in 0..n_test_batches {
        let minist_data_batch = Dataset::minist_as_batch(&mut test_data_file, &mut test_label_file, batch_size, &mut tracker);      
        let (loss_per_epoch, _) = net.loss(minist_data_batch.clone(), &LossFunc::Conventional).unwrap();
        let acc = net.max_pred_test(minist_data_batch);

        mean_loss += loss_per_epoch;
        mean_acc += acc;
      }

      mean_loss /= n_test_batches as f32;
      mean_acc /= n_test_batches as f32;
      test_loss.push(mean_loss);
      test_acc.push(mean_acc);
      println!("Test Epoch {} -> Loss: {:.3}, Acc: {:.3}", e+1, mean_loss, mean_acc);
    }

    let train_epochs = train_loss.len();
    let test_epochs = test_loss.len();
    Matrix::from_body(train_loss, [train_epochs, 1]).to_csv("./out/lc_train_loss.csv").unwrap();
    Matrix::from_body(train_acc, [train_epochs, 1]).to_csv("./out/lc_train_acc.csv").unwrap();
    Matrix::from_body(test_loss, [test_epochs, 1]).to_csv("./out/lc_test_loss.csv").unwrap();
    Matrix::from_body(test_acc, [test_epochs, 1]).to_csv("./out/lc_test_acc.csv").unwrap();
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

  #[test]
  fn read_data() {
    let mut train_data_file = File::open("./minist/train-images.idx3-ubyte").unwrap();
    let buf = &mut [0u8; 16];

    train_data_file.read(buf).unwrap();
    println!("{:?}", buf);
  }
}