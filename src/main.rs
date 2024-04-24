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

fn main() {
  let ref mut seed = 9868943421677;
    
  let conv_scale: usize = 8;
  let dense_scale: usize = 1;

  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::FeatureMaps(1),
    vec![[7, 7], [5, 5], [3, 3]],
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let avg_pooling_layer: CLayer<Cf32> = Reduce::init(
    3, 
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
    IOShape::FeatureMaps(3),
    vec![[7, 7], [5, 5], [3, 3]],
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let flatten_layer: CLayer<Cf32> = Flatten::init(vec![[14, 14]; 3*3]).wrap();
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(14*14*3*3), 
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

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(input_layer).unwrap();
  network.add(avg_pooling_layer).unwrap();
  network.add(another_conv).unwrap();
  network.add(flatten_layer).unwrap();
  network.add(first_dense).unwrap();
  network.add(output_layer).unwrap();

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
  let epochs: usize = 10;

  let lr = Cf32::new(1.0, 0.0);
  println!("Begining training and testing pipeline.");
  for e in 0..epochs {
    /* training pipeline */
    let ref mut train_tracker = 0;
    let train_data_file = &mut File::open("./minist/train-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/train-labels.idx1-ubyte").unwrap();
    let mut train_loss = 0.0;
    let mut train_acc = 0.0;
    for b in 0..train_batches {
      let t: Instant = Instant::now();
      let train_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(train_data_file, train_label_file, batch_size, train_tracker);
      
      network.gradient_opt(train_data.clone(), ComplexLossFunc::Conventional, lr).unwrap();

      let current_train_loss = network.loss(train_data.clone(), &ComplexLossFunc::Conventional).unwrap();
      let current_train_acc = network.max_pred_test(train_data);
      train_loss += current_train_loss;
      train_acc += current_train_acc;
      
      
      print!("\rEpoch {}, Batch {} -> Loss: {:.3}, Accuracy: {:.3} (time: {:.3?})", e+1, b+1, current_train_loss, current_train_acc, t.elapsed());
      io::stdout().flush().unwrap();
    }

    train_loss /= train_batches as f32;
    train_acc /= train_batches as f32;
    println!();
    println!("Epoch {} -> Mean Loss: {:.3}, Mean Accuracy: {:.3}", e+1, train_loss, train_acc);
    train_loss_vec.push(train_loss);
    train_acc_vec.push(train_acc);

    /* test pipeline */
    let ref mut test_tracker = 0;
    let test_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let test_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    let mut test_loss = 0.0;
    let mut test_acc = 0.0;
    for t in 0..test_batches {
      let initial_test_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(test_data_file, test_label_file, batch_size, test_tracker);
      let initial_test_loss = network.loss(initial_test_data.clone(), &ComplexLossFunc::Conventional).unwrap();
      let initial_test_acc = network.max_pred_test(initial_test_data);
      test_loss += initial_test_loss;
      test_acc += initial_test_acc;

      print!("\rTest Values | Epoch {}, Batch {} -> Loss: {:.3}, Accuracy: {:.3}", e+1, t+1, initial_test_loss, initial_test_acc);
      io::stdout().flush().unwrap();
    }

    test_loss /= test_batches as f32;
    test_acc /= test_batches as f32;
    println!();
    println!("Test Values | Epoch {} -> Mean Loss: {:.3}, Mean Accuracy: {:.3}", e+1, test_loss, test_acc);
    test_loss_vec.push(test_loss);
    test_acc_vec.push(test_acc);
  }

  Matrix::from_body(train_loss_vec, [epochs, 1]).to_csv("./out/conv_network_loss.csv").unwrap();
  Matrix::from_body(train_acc_vec, [epochs, 1]).to_csv("./out/conv_network_acc.csv").unwrap();
  Matrix::from_body(test_loss_vec, [epochs, 1]).to_csv("./out/conv_network_test_loss.csv").unwrap();
  Matrix::from_body(test_acc_vec, [epochs, 1]).to_csv("./out/conv_network_test_acc.csv").unwrap();
}