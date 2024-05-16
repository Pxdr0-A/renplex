use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

use renplex::act::ComplexActFunc;
use renplex::cvnn::layer::conv::ConvCLayer;
use renplex::cvnn::layer::dense::DenseCLayer;
use renplex::cvnn::layer::flatten::Flatten;
use renplex::cvnn::layer::reduce::Reduce;
use renplex::cvnn::layer::CLayer;
use renplex::cvnn::network::CNetwork;
use renplex::dataset::Dataset;
use renplex::init::InitMethod;
use renplex::input::IOShape;
use renplex::math::cfloat::Cf32;
use renplex::math::matrix::Matrix;
use renplex::math::Complex;
use renplex::opt::ComplexLossFunc;

fn _get_1conv_layer_cvcnn(seed: &mut u128) -> CNetwork<Cf32> {
  let conv_scale: usize = 1;
  let dense_scale: usize = 1;

  let input_layer_kernels = vec![[3,3]; 8];

  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::FeatureMaps(1),
    input_layer_kernels,
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let flatten_layer: CLayer<Cf32> = Flatten::init(vec![[28, 28]; 8]).wrap();
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(28*28*8), 
    64,
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let output_layer: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(64), 
    10, 
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale), 
    seed
  ).unwrap().wrap();

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(input_layer).unwrap();
  network.add(flatten_layer).unwrap();
  network.add(first_dense).unwrap();
  network.add(output_layer).unwrap();

  network
}

fn _get_2conv_layer_cvcnn(seed: &mut u128) -> CNetwork<Cf32> {
  let conv_scale: usize = 1;
  let dense_scale: usize = 1;

  let input_layer_kernels = vec![[3,3]; 8];
  let second_layer_kernels = vec![[3,3]; 4];

  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::FeatureMaps(1),
    input_layer_kernels,
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let avg_pooling_layer: CLayer<Cf32> = Reduce::init(
    8, 
    [2, 2],
    Box::new(|block: &[Cf32]| { 
      let block_len = block.len(); 
      block
        .into_iter()
        .fold(Cf32::default(), |acc, elm| { acc + *elm }) / Cf32::new(block_len as f32, 0.0)
    }),
    Matrix::from_body(
      vec![
        Cf32::new(0.25, 0.0), Cf32::new(0.50, 0.0), Cf32::new(0.25, 0.0),
        Cf32::new(0.50, 0.0), Cf32::new(1.00, 0.0), Cf32::new(0.50, 0.0),
        Cf32::new(0.25, 0.0), Cf32::new(0.50, 0.0), Cf32::new(0.25, 0.0)
      ], [3, 3])
  ).wrap();
  let another_conv: CLayer<Cf32> = ConvCLayer::init(
    IOShape::FeatureMaps(8),
    second_layer_kernels,
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let flatten_layer: CLayer<Cf32> = Flatten::init(vec![[14, 14]; 8*4]).wrap();
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(14*14*8*4), 
    64,
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let output_layer: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(64), 
    10, 
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale), 
    seed
  ).unwrap().wrap();

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(input_layer).unwrap();
  network.add(avg_pooling_layer).unwrap();
  network.add(another_conv).unwrap();
  network.add(flatten_layer).unwrap();
  network.add(first_dense).unwrap();
  network.add(output_layer).unwrap();

  network
}

fn _get_fully_connected_cvnn(seed: &mut u128) -> CNetwork<Cf32> {
  let dense_scale: usize = 1;

  let flatten_layer: CLayer<Cf32> = Flatten::init(vec![[28, 28]; 1]).wrap();
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(28*28), 
    28,
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let second_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(28), 
    16, 
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale), 
    seed
  ).unwrap().wrap();
  let third_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(16), 
    16, 
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale), 
    seed
  ).unwrap().wrap();
  let forth_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(16), 
    10, 
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale), 
    seed
  ).unwrap().wrap();

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(flatten_layer).unwrap();
  network.add(first_dense).unwrap();
  network.add(second_dense).unwrap();
  network.add(third_dense).unwrap();
  network.add(forth_dense).unwrap();

  network
}

fn test_pipeline(
  network: &CNetwork<Cf32>,
  loss_func: &ComplexLossFunc,
  test_batches: usize,
  batch_size: usize,
  epoch: usize,
  test_loss_vec: &mut Vec<f32>, 
  test_acc_vec: &mut Vec<f32>,
) {
  /* test pipeline */
  let ref mut test_tracker = 0;
  let test_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
  let test_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
  let mut mean_test_loss = 0.0;
  let mut mean_test_acc = 0.0;
  for t in 0..test_batches {
    let t_test: Instant = Instant::now();
    let test_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(test_data_file, test_label_file, batch_size, test_tracker);
    let inst_test_loss = network.loss(test_data.clone(), loss_func).unwrap();
    let inst_test_acc = network.max_pred_test(test_data);
    mean_test_loss += inst_test_loss;
    mean_test_acc += inst_test_acc;

    //let accu = (t+1) as f32;
    print!(
      "\rTest Values | Epoch {}, Batch {} -> Loss: {:.3}, Accuracy: {:.3} (time: {:.3?})", 
      epoch+1, t+1, inst_test_loss, inst_test_acc, t_test.elapsed()
    );
    io::stdout().flush().unwrap();
  }

  mean_test_loss /= test_batches as f32;
  mean_test_acc /= test_batches as f32;
  println!();
  println!("Test Values | Epoch {} -> Mean Loss: {:.3}, Mean Accuracy: {:.3}", epoch+1, mean_test_loss, mean_test_acc);
  test_loss_vec.push(mean_test_loss);
  test_acc_vec.push(mean_test_acc);
}

fn train_pipeline(
  network: &mut CNetwork<Cf32>,
  loss_func: &ComplexLossFunc,
  lr: Cf32,
  train_batches: usize,
  batch_size: usize,
  epoch: usize,
  train_loss_vec: &mut Vec<f32>, 
  train_acc_vec: &mut Vec<f32>,
) {
  /* training pipeline */
  let ref mut train_tracker = 0;
  let train_data_file = &mut File::open("./minist/train-images.idx3-ubyte").unwrap();
  let train_label_file = &mut File::open("./minist/train-labels.idx1-ubyte").unwrap();
  let mut mean_train_loss = 0.0;
  let mut mean_train_acc = 0.0;
  for b in 0..train_batches {
    let t: Instant = Instant::now();
    let train_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(train_data_file, train_label_file, batch_size, train_tracker);

    network.gradient_opt(train_data.clone(), &loss_func, lr).unwrap();

    let inst_train_loss = network.loss(train_data.clone(), &loss_func).unwrap();
    let inst_train_acc = network.max_pred_test(train_data);
    mean_train_loss += inst_train_loss;
    mean_train_acc += inst_train_acc;
    
    //let accu = (b+1) as f32;
    print!(
      "\rEpoch {}, Batch {} -> Loss: {:.3}, Accuracy: {:.3} (time: {:.3?})", 
      epoch+1, b+1, inst_train_loss, inst_train_acc, t.elapsed()
    );
    io::stdout().flush().unwrap();
  }

  mean_train_loss /= train_batches as f32;
  mean_train_acc /= train_batches as f32;
  println!();
  println!("Epoch {} -> Mean Loss: {:.3}, Mean Accuracy: {:.3}", epoch+1, mean_train_loss, mean_train_acc);
  train_loss_vec.push(mean_train_loss);
  train_acc_vec.push(mean_train_acc);
}

fn main() {
  let ref mut seed = 34987346939856829;
  
  let mut network = _get_fully_connected_cvnn(seed);
  println!("Created the Network.");

  let mut train_loss_vec = Vec::new();
  let mut test_loss_vec: Vec<f32> = Vec::new();
  let mut train_acc_vec = Vec::new();
  let mut test_acc_vec: Vec<f32> = Vec::new();

  let total_train_data = 60000;
  let total_test_data = 10000;
  let batch_size = 100;
  let train_batches = total_train_data / batch_size;
  let test_batches = total_test_data / batch_size;
  let epochs: usize = 50;

  let lr = Cf32::new(0.1, 0.0);
  let loss_func = ComplexLossFunc::Conventional;
  println!("Begining training and testing pipeline.");
  for e in 0..epochs {
    test_pipeline(
      &network, 
      &loss_func, 
      test_batches, 
      batch_size, 
      e,
      &mut test_loss_vec, 
      &mut test_acc_vec
    );

    train_pipeline(
      &mut network, 
      &loss_func, 
      lr, 
      train_batches, 
      batch_size, 
      e, 
      &mut train_loss_vec, 
      &mut train_acc_vec
    )
  }

  Matrix::from_body(train_loss_vec, [epochs, 1]).to_csv("./out/loss_0_1.csv").unwrap();
  Matrix::from_body(train_acc_vec, [epochs, 1]).to_csv("./out/acc_0_1.csv").unwrap();
  Matrix::from_body(test_loss_vec, [epochs, 1]).to_csv("./out/test_loss_0_1.csv").unwrap();
  Matrix::from_body(test_acc_vec, [epochs, 1]).to_csv("./out/test_acc_0_1.csv").unwrap();
}