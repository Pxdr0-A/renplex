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
  use std::fs::File;

  use crate::act::ComplexActFunc;
  use crate::cvnn::layer::conv::ConvCLayer;
  use crate::cvnn::layer::dense::DenseCLayer;
use crate::cvnn::layer::flatten::Flatten;
  use crate::cvnn::layer::CLayer;
  use crate::cvnn::network::CNetwork;
  use crate::dataset::Dataset;
  use crate::init::{InitMethod, PredictModel};
  use crate::input::IOShape;
  use crate::math::cfloat::Cf32;
  use crate::math::matrix::Matrix;
use crate::opt::ComplexLossFunc;

  #[test]
  fn dataset_to_csv_test() {
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
  fn conv_test() {
    let matrix = Matrix::from_body(
      vec![
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0
      ], [4, 4]);
    
    let a = matrix.conv(&Matrix::from_body(
      vec![
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0
      ], [3, 3])).unwrap();
    
    println!("{}", a);
  }

  #[test]
  fn image_conv_test() {
    let train_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    let batch_size = 2;
    let ref mut tracker = 0;

    let data: Dataset<f32, f32> = Dataset::minist_as_batch(train_data_file, train_label_file, batch_size, tracker);

    let (image_point, _) = data.get_point(1);

    let image = Matrix::from_body(image_point.to_vec(), [28, 28]);
    image.to_csv("./out/original.csv").unwrap();

    let kernel1: Matrix<f32> = Matrix::from_body(
      vec![
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
      ], [3, 3]);

    let kernel2: Matrix<f32> = Matrix::from_body(
      vec![
        -1.0, -2.0, -1.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 1.0,
      ], [3, 3]);

    let mut image_conv1 = image.conv(&kernel1).unwrap();
    let mut image_conv2 = image.conv(&kernel2).unwrap();

    for row in image_conv1.rows_as_iter_mut() {
      for elm in row {
        if elm.is_sign_negative() {
          *elm = 0.0;
        }
      }
    }

    for row in image_conv2.rows_as_iter_mut() {
      for elm in row {
        if elm.is_sign_negative() {
          *elm = 0.0;
        }
      }
    }

    image_conv1.to_csv("./out/conv_image.csv").unwrap();
    image_conv2.to_csv("./out/conv_image1.csv").unwrap();
  }

  #[test]
  fn conv_network_test() {
    let ref mut seed = 827278272298732;
    
    let conv_scale: usize = 10;
    let dense_scale: usize = 1;

    let input_layer: CLayer<Cf32> = ConvCLayer::init(
      IOShape::Matrix([28, 28]), 
      vec![[7, 7], [5, 5], [3, 3]], 
      ComplexActFunc::RITReLU, 
      InitMethod::Random(conv_scale),
      seed
    ).unwrap().wrap();
    let second_layer: CLayer<Cf32> = Flatten::init([28, 28]).wrap();
    let third_layer: CLayer<Cf32> = DenseCLayer::init(
      IOShape::Vector(28*28), 
      16, 
      ComplexActFunc::RITSigmoid, 
      InitMethod::Random(dense_scale),
      seed
    ).unwrap().wrap();
    let output_layer: CLayer<Cf32> = DenseCLayer::init(
      IOShape::Vector(16), 
      10, 
      ComplexActFunc::RITSigmoid, 
      InitMethod::Random(dense_scale), 
      seed
    ).unwrap().wrap();

    let mut network: CNetwork<Cf32> = CNetwork::new();
    network.add_input(input_layer).unwrap();
    network.add(second_layer).unwrap();
    network.add(third_layer).unwrap();
    network.add(output_layer).unwrap();

    let mut train_loss_vec = Vec::new();
    let mut test_loss_vec = Vec::new();
    let mut train_acc_vec = Vec::new();
    let mut test_acc_vec = Vec::new();

    let total_train_data = 60000;
    let total_test_data = 10000;
    let batch_size = 100;
    let train_batches = total_train_data / batch_size;
    let test_batches = total_test_data / batch_size;  

    let ref mut train_tracker = 0;
    let train_data_file = &mut File::open("./minist/train-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/train-labels.idx1-ubyte").unwrap();
    let mut train_loss = 0.0;
    let mut train_acc = 0.0;
    for _ in 0..train_batches {
      let initial_train_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(train_data_file, train_label_file, batch_size, train_tracker);
      let (initial_train_loss, _) = network.loss(initial_train_data.clone(), &ComplexLossFunc::Conventional).unwrap();
      let initial_train_acc = network.max_pred_test(initial_train_data.clone());
      train_loss += initial_train_loss;
      train_acc += initial_train_acc;
    }

    train_loss /= train_batches as f32;
    train_acc /= train_batches as f32;
    println!("Initial loss: {}", train_loss);
    println!("Initial loss: {}", train_acc);
    train_loss_vec.push(train_loss);
    train_acc_vec.push(train_acc);

    let ref mut test_tracker = 0;
    let test_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let test_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    let mut test_loss = 0.0;
    let mut test_acc = 0.0;
    for _ in 0..test_batches {
      let initial_test_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(test_data_file, test_label_file, batch_size, test_tracker);
      let (initial_test_loss, _) = network.loss(initial_test_data.clone(), &ComplexLossFunc::Conventional).unwrap();
      let initial_test_acc = network.max_pred_test(initial_test_data);
      test_loss += initial_test_loss;
      test_acc += initial_test_acc;
    }

    test_loss /= test_batches as f32;
    test_acc /= test_batches as f32;
    println!("Initial loss: {}", test_loss);
    println!("Initial loss: {}", test_acc);
    test_loss_vec.push(test_loss);
    test_acc_vec.push(test_acc);
  }
}