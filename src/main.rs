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
use renplex::input::{IOShape, IOType};
use renplex::math::cfloat::Cf32;
use renplex::math::matrix::Matrix;
use renplex::math::Complex;
use renplex::opt::ComplexLossFunc;

fn _get_1conv_layer_cvcnn(seed: &mut u128) -> (usize, CNetwork<Cf32>) {

  let input_units = 8;
  let k_size = [3, 3];
  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::Matrix(1),
    input_units,
    k_size,
    ComplexActFunc::RITReLU, 
    InitMethod::HeInit(input_units * k_size[0] * k_size[1]),
    seed
  ).unwrap().wrap();

  let flatten_layer: CLayer<Cf32> = Flatten::init([26, 26], 8).wrap();
  
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(26*26*8), 
    16,
    ComplexActFunc::RITSigmoid, 
    InitMethod::XavierGlorotU(26*26*8 + 16),
    seed
  ).unwrap().wrap();
  
  let output_layer: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(16), 
    10, 
    ComplexActFunc::RITSigmoid, 
    InitMethod::XavierGlorotU(16 + 10), 
    seed
  ).unwrap().wrap();

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(input_layer).unwrap();
  network.add(flatten_layer).unwrap();
  network.add(first_dense).unwrap();
  network.add(output_layer).unwrap();

  println!("Created 1 layer, CV-CNN.");

  (1, network)
}

fn _get_2conv_layer_cvcnn(seed: &mut u128) -> (usize, CNetwork<Cf32>) {
  let input_units = 8;
  let k_size = [3, 3];
  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::Matrix(1),
    input_units,
    k_size,
    ComplexActFunc::RITReLU, 
    InitMethod::HeInit(input_units * k_size[0] * k_size[1]),
    seed
  ).unwrap().wrap();

  let avg_pooling_layer: CLayer<Cf32> = Reduce::init(
    8, 
    [2,2],
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

  let inter_units = 8;
  let inter_k_size = [3, 3];
  let inter_conv: CLayer<Cf32> = ConvCLayer::init(
    IOShape::Matrix(8),
    inter_units,
    inter_k_size,
    ComplexActFunc::RITReLU, 
    InitMethod::HeInit(inter_units * inter_k_size[0] * inter_k_size[1]),
    seed
  ).unwrap().wrap();

  let flatten_layer: CLayer<Cf32> = Flatten::init([11, 11], 8).wrap();
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(11*11*8), 
    16,
    ComplexActFunc::RITSigmoid, 
    InitMethod::XavierGlorotU(11*11*8 + 16),
    seed
  ).unwrap().wrap();
  let output_layer: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(16), 
    10, 
    ComplexActFunc::RITSigmoid, 
    InitMethod::XavierGlorotU(16 + 10), 
    seed
  ).unwrap().wrap();

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(input_layer).unwrap();
  network.add(avg_pooling_layer).unwrap();
  network.add(inter_conv).unwrap();
  network.add(flatten_layer).unwrap();
  network.add(first_dense).unwrap();
  network.add(output_layer).unwrap();

  println!("Created 2 layer, CV-CNN.");

  (2, network)
}

fn _get_fully_connected_cvnn(seed: &mut u128) -> (usize, CNetwork<Cf32>) {
  let dense_act = ComplexActFunc::RITSigmoid;

  let flatten_layer: CLayer<Cf32> = Flatten::init([28, 28], 1).wrap();

  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(28*28), 
    28,
    dense_act, 
    InitMethod::XavierGlorotU(28*28 + 28),
    seed
  ).unwrap().wrap();

  let second_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(28), 
    16, 
    dense_act, 
    InitMethod::XavierGlorotU(28 + 16), 
    seed
  ).unwrap().wrap();

  let third_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(16), 
    16, 
    dense_act, 
    InitMethod::XavierGlorotU(16 + 16), 
    seed
  ).unwrap().wrap();

  let forth_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(16), 
    10, 
    dense_act, 
    InitMethod::XavierGlorotU(16 + 10), 
    seed
  ).unwrap().wrap();

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(flatten_layer).unwrap();
  network.add(first_dense).unwrap();
  network.add(second_dense).unwrap();
  network.add(third_dense).unwrap();
  network.add(forth_dense).unwrap();

  println!("Created fully connected CVNN");

  (0, network)
}

fn extract_features(
  network: &CNetwork<Cf32>,
  conv_network_id: usize, 
  lr_re: f32, 
  lr_im: f32,
  epochs: usize
) {
  /* intercept to inspect feature maps or intermediate activations. */
  let test_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
  let test_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
  let batch_size = 100;
  let ref mut tracker = 0;
  let data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(test_data_file, test_label_file, batch_size, tracker);
  let (image_point, _) = data.get_point(50);

  let image;
  match image_point {
    IOType::Matrix(map) => {image = map[0].clone();},
    _ => {panic!("ups...")}
  }
  image.to_csv(format!(
    "./out/complex_features/conv{}/lr_{:.2}_{:.2}_{}e_original.csv", 
    conv_network_id, lr_re, lr_im, epochs
  )).unwrap();

  if conv_network_id == 1 {
    let (feature_maps, _) = network.intercept(image_point.clone(), 1).unwrap();
    match feature_maps {
      IOType::Matrix(maps) => {
        maps.into_iter().enumerate().for_each(|(id, feature)| {
          feature.to_csv(format!(
            "out/complex_features/conv{}/lr_{:.2}_{:.2}_{}e_feature_{}.csv", 
            conv_network_id, lr_re, lr_im, epochs, id
          )).unwrap();
        })
      },
      _ => { panic!("nope, not this feature") }
    }
  } else if conv_network_id == 2 {
    let order = 1;
    let (feature_maps, _) = network.intercept(image_point.clone(), order).unwrap();
    match feature_maps {
      IOType::Matrix(maps) => {
        maps.into_iter().enumerate().for_each(|(id, feature)| {
          feature.to_csv(format!(
            "out/complex_features/conv{}/lr_{:.2}_{:.2}_{}e_feature_{}_{}.csv", 
            conv_network_id, lr_re, lr_im, epochs, id, order
          )).unwrap();
        })
      },
      _ => { panic!("nope, not this feature") }
    }

    let order = 3;
    let (feature_maps, _) = network.intercept(image_point.clone(), 3).unwrap();
    match feature_maps {
      IOType::Matrix(maps) => {
        maps.into_iter().enumerate().for_each(|(id, feature)| {
          feature.to_csv(format!(
            "out/complex_features/conv{}/lr_{:.2}_{:.2}_{}e_feature_{}_{}.csv", 
            conv_network_id, lr_re, lr_im, epochs, id, order
          )).unwrap();
        })
      },
      _ => { panic!("nope, not this feature") }
    }
  }
}

fn test_pipeline(
  network: &CNetwork<Cf32>,
  loss_func: &ComplexLossFunc,
  test_batches: usize,
  batch_size: usize,
  test_loss_vec: &mut Vec<f32>, 
  test_acc_vec: &mut Vec<f32>
) {
  /* test pipeline */
  let ref mut test_tracker = 0;
  let test_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
  let test_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
  let mut mean_test_loss = 0.0;
  let mut mean_test_acc = 0.0;
  for b in 0..test_batches {
    let t_test: Instant = Instant::now();
    let test_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(test_data_file, test_label_file, batch_size, test_tracker);
    let inst_test_loss = network.loss(test_data.clone(), loss_func).unwrap();
    let inst_test_acc = network.max_pred_test(test_data);
    mean_test_loss += inst_test_loss;
    mean_test_acc += inst_test_acc;

    let accu = (b+1) as f32;
    print!(
      "\r| Batch {} -> Loss: ({:.3}, {:.3}), Accuracy: ({:.3}, {:.3}) (time: {:.3?})", 
      b+1, inst_test_loss, mean_test_loss/accu, inst_test_acc, mean_test_acc/accu, t_test.elapsed()
    );
    io::stdout().flush().unwrap();
  }

  println!();
  mean_test_loss /= test_batches as f32;
  mean_test_acc /= test_batches as f32;
  test_loss_vec.push(mean_test_loss);
  test_acc_vec.push(mean_test_acc);
}

fn train_pipeline(
  network: &mut CNetwork<Cf32>,
  loss_func: &ComplexLossFunc,
  lr: Cf32,
  train_batches: usize,
  batch_size: usize,
  train_loss_vec: &mut Vec<f32>, 
  train_acc_vec: &mut Vec<f32>
) {
  /* training pipeline */
  let ref mut train_tracker = 0;
  let train_data_file = &mut File::open("./minist/train-images.idx3-ubyte").unwrap();
  let train_label_file = &mut File::open("./minist/train-labels.idx1-ubyte").unwrap();
  let mut mean_train_loss = 0.0;
  let mut mean_train_acc = 0.0;

  for b in 0..train_batches {
    let t_train: Instant = Instant::now();
    let train_data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(train_data_file, train_label_file, batch_size, train_tracker);

    network.gradient_opt(train_data.clone(), &loss_func, lr).unwrap();

    let inst_train_loss = network.loss(train_data.clone(), &loss_func).unwrap();
    let inst_train_acc = network.max_pred_test(train_data);

    mean_train_loss += inst_train_loss;
    mean_train_acc += inst_train_acc;
    
    let accu = (b+1) as f32;
    print!(
      "\r| Batch {} -> Loss: ({:.3}, {:.3}), Accuracy: ({:.3}, {:.3}) (time: {:.3?})", 
      b+1, inst_train_loss, mean_train_loss/accu, inst_train_acc, mean_train_acc/accu, t_train.elapsed()
    );
    io::stdout().flush().unwrap();
  }

  println!();
  mean_train_loss /= train_batches as f32;
  mean_train_acc /= train_batches as f32;
  train_loss_vec.push(mean_train_loss);
  train_acc_vec.push(mean_train_acc);
}

fn main() {
  // ANSI escape codes
  let reset = "\x1b[0m";
  let bold = "\x1b[1m";
  let _red = "\x1b[31m";
  let green = "\x1b[32m";
  let yellow = "\x1b[33m";
  let underline = "\x1b[4m";

  let ref mut seed = 437628367189104305197;
  println!("Using seed: {}", seed);

  let (network_id, mut network) = _get_1conv_layer_cvcnn(seed);
  println!("Created the Network.");

  let mut train_loss_vec: Vec<f32> = Vec::new();
  let mut test_loss_vec: Vec<f32> = Vec::new();
  let mut train_acc_vec: Vec<f32> = Vec::new();
  let mut test_acc_vec: Vec<f32> = Vec::new();

  let total_train_data = 60000;
  let total_test_data = 10000;
  let batch_size = 100;
  let train_batches = total_train_data / batch_size;
  let test_batches = total_test_data / batch_size;
  let epochs: usize = 20;

  /* lr
  Dense : 3 or 4
  */
  let r = 2.5_f32;
  let phase = std::f32::consts::PI * 0.0;
  let lr_re = r * phase.cos();
  let lr_im = r * phase.sin();
  let lr = Cf32::new(lr_re, lr_im);
  let loss_func = ComplexLossFunc::Conventional;

  println!("Begining training and testing pipeline.");
  println!("Using constant learning rate: [{:.1}, {:.1}]", lr_re, lr_im);
  println!("Total Epochs: {}", epochs);
  println!("Batch size: {} | Number of Batches (train, test): {:?}", batch_size, (train_batches, test_batches));
  for e in 0..epochs {
    println!();
    println!("{}Epoch {}/{}{}",bold, e+1, epochs, reset);
    println!("| {}{}{}Train Pipeline{}", bold, underline, yellow, reset);
    train_pipeline(
      &mut network, 
      &loss_func, 
      lr, 
      train_batches, 
      batch_size, 
      &mut train_loss_vec, 
      &mut train_acc_vec
    );

    println!("| {}{}{}Test Pipeline{}", bold, underline, green, reset);
    test_pipeline(
      &network, 
      &loss_func, 
      test_batches, 
      batch_size, 
      &mut test_loss_vec, 
      &mut test_acc_vec
    );
  }
  
  println!();
  println!("Ended training.");

  /* Network exploration */
  if network_id > 0 {
    /* it is a convolutional network */
    extract_features(&network, network_id, lr_re, lr_im, epochs);
  }
  
  /* Save Results (learning curves) to local memory as .csv file */
  Matrix::from_body(train_loss_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/loss_{:.2}_{:.2}_{}e.csv", 
    network_id, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(train_acc_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/acc_{:.2}_{:.2}_{}e.csv", 
    network_id, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(test_loss_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/test_loss_{:.2}_{:.2}_{}e.csv", 
    network_id, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(test_acc_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/test_acc_{:.2}_{:.2}_{}e.csv", 
    network_id, lr_re, lr_im, epochs
  )).unwrap();

  println!("Saved relevant data.");
  println!("Ended Pipeline.")
}
