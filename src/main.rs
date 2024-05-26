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

fn _get_1conv_layer_cvcnn(seed: &mut u128) -> CNetwork<Cf32> {
  let conv_scale: usize = 1;
  let dense_scale: usize = 1;

  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::FeatureMaps(1),
    8,
    [3,3],
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let flatten_layer: CLayer<Cf32> = Flatten::init(vec![[26, 26]; 8]).wrap();
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(26*26*8), 
    8,
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let output_layer: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(8), 
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

  println!("Created 1 layer, CV-CNN.");

  network
}

fn _get_2conv_layer_cvcnn(seed: &mut u128) -> CNetwork<Cf32> {
  let conv_scale: usize = 1;
  let dense_scale: usize = 1;

  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::FeatureMaps(1),
    8,
    [3,3],
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
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
  let another_conv: CLayer<Cf32> = ConvCLayer::init(
    IOShape::FeatureMaps(8),
    4,
    [3,3],
    ComplexActFunc::RITReLU, 
    InitMethod::Random(conv_scale),
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let flatten_layer: CLayer<Cf32> = Flatten::init(vec![[11, 11]; 4]).wrap();
  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(11*11*4), 
    8*4,
    ComplexActFunc::RITSigmoid, 
    InitMethod::Random(dense_scale),
    seed
  ).unwrap().wrap();
  let output_layer: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Vector(8*4), 
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

  println!("Created 2 layer, CV-CNN.");

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

  println!("Created fully connected CVNN");

  network
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
    IOType::FeatureMaps(map) => {image = map[0].clone();},
    _ => {panic!("ups...")}
  }
  image.to_csv(format!(
    "./out/complex_features/conv{}/lr_{}_{}_{}e_original.csv", 
    conv_network_id, lr_re, lr_im, epochs
  )).unwrap();

  let (feature_maps, _) = network.intercept(image_point.clone(), 1).unwrap();
  match feature_maps {
    IOType::FeatureMaps(maps) => {
      maps.into_iter().enumerate().for_each(|(id, feature)| {
        feature.to_csv(format!(
          "out/complex_features/conv{}/lr_{}_{}_{}e_feature_{}.csv", 
          conv_network_id, lr_re, lr_im, epochs, id
        )).unwrap();
      })
    },
    _ => { panic!("nope, not this feature") }
  }
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
      "\rEpoch {} | Batch {} -> Loss: {:.3}, Accuracy: {:.3} (time: {:.3?})", 
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
  let ref mut seed = 437628367189104305197;
  println!("Using seed: {}", seed);

  let mut network = _get_2conv_layer_cvcnn(seed);
  let conv_network_id = 2;
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

  /*
    Dense lr: 4 -> 7 (peak at 6.0)
    Conv1 lr: 1 -> 3 (peak at 2.5)
    Conv2 lr: so far 1.0 is better (80%)
  */
  let lr_re = 1.0;
  let lr_im = 0.0;
  let lr = Cf32::new(lr_re, lr_im);
  let loss_func = ComplexLossFunc::Conventional;

  println!("Begining training and testing pipeline.");
  println!("Using constant learning rate: {}", lr);
  println!("Total Epochs: {}", epochs);
  println!("Batch size: {} | Number of Batches (train, test): {:?}", batch_size, (train_batches, test_batches));
  for e in 0..epochs {
    train_pipeline(
      &mut network, 
      &loss_func, 
      lr, 
      train_batches, 
      batch_size, 
      e, 
      &mut train_loss_vec, 
      &mut train_acc_vec
    );

    test_pipeline(
      &network, 
      &loss_func, 
      test_batches, 
      batch_size, 
      e,
      &mut test_loss_vec, 
      &mut test_acc_vec
    );
  }

  /* Network exploration */
  println!("Ended training. Exploring the trained network.");
  extract_features(&network, conv_network_id, lr_re, lr_im, epochs);
  
  /* save results to local memory as .csv file */
  Matrix::from_body(train_loss_vec, [epochs, 1]).to_csv(format!(
    "./out/lr_conv{}/loss_{}_{}_{}e.csv", 
    conv_network_id, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(train_acc_vec, [epochs, 1]).to_csv(format!(
    "./out/lr_conv{}/acc_{}_{}_{}e.csv", 
    conv_network_id, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(test_loss_vec, [epochs, 1]).to_csv(format!(
    "./out/lr_conv{}/test_loss_{}_{}_{}e.csv", 
    conv_network_id, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(test_acc_vec, [epochs, 1]).to_csv(format!(
    "./out/lr_conv{}/test_acc_{}_{}_{}e.csv", 
    conv_network_id, lr_re, lr_im, epochs
  )).unwrap();

  println!("Saved relevant data.");
  println!("Ended Pipeline.")
}
