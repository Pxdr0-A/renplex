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
use renplex::math::matrix::{Matrix, SliceToMatrix};
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
  let input_features = 1;
  let input_units = 8;
  let k_size = [3, 3];
  let input_layer: CLayer<Cf32> = ConvCLayer::init(
    IOShape::Matrix(input_features),
    input_units,
    k_size,
    ComplexActFunc::RITReLU, 
    InitMethod::HeInit(input_units * k_size[0] * k_size[1]),
    seed
  ).unwrap().wrap();

  let avg_pooling_layer: CLayer<Cf32> = Reduce::init(
    input_units, 
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

  let inter_units = 16;
  let inter_k_size = [3, 3];
  let inter_conv: CLayer<Cf32> = ConvCLayer::init(
    IOShape::Matrix(input_units),
    inter_units,
    inter_k_size,
    ComplexActFunc::RITReLU, 
    InitMethod::HeInit(inter_units * inter_k_size[0] * inter_k_size[1]),
    seed
  ).unwrap().wrap();

  let flatten_layer: CLayer<Cf32> = Flatten::init([11, 11], inter_units).wrap();

  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(11*11*inter_units), 
    16,
    ComplexActFunc::RITSigmoid, 
    InitMethod::XavierGlorotU(11*11*inter_units + 16),
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

fn _get_auto_encoder(seed: &mut u128, samples: usize) -> (usize, CNetwork<Cf32>) {
  let input_units = 128; // originally equal to samples
  let inter_encoding = 32;
  let min_encoding = 16;

  let inter_act = ComplexActFunc::RITTanh;
  let out_act = ComplexActFunc::None;

  let first_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(samples), 
    input_units,
    inter_act, 
    InitMethod::XavierGlorotU(samples + input_units),
    seed
  ).unwrap().wrap();

  let second_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(input_units), 
    inter_encoding, 
    inter_act, 
    InitMethod::XavierGlorotU(input_units + inter_encoding), 
    seed
  ).unwrap().wrap();

  let third_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(inter_encoding), 
    min_encoding, 
    inter_act, 
    InitMethod::XavierGlorotU(inter_encoding + min_encoding), 
    seed
  ).unwrap().wrap();

  let forth_dense: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(min_encoding), 
    inter_encoding, 
    inter_act, 
    InitMethod::XavierGlorotU(min_encoding + inter_encoding), 
    seed
  ).unwrap().wrap();

  let output_layer: CLayer<Cf32> = DenseCLayer::init(
    IOShape::Scalar(inter_encoding), 
    samples, 
    out_act, 
    InitMethod::XavierGlorotU(inter_encoding + samples), 
    seed
  ).unwrap().wrap();

  println!("Initiated Layers.");

  let mut network: CNetwork<Cf32> = CNetwork::new();
  network.add_input(first_dense).unwrap();
  network.add(second_dense).unwrap();
  network.add(third_dense).unwrap();
  network.add(forth_dense).unwrap();
  network.add(output_layer).unwrap();

  println!("Created CVNN Auto-encoder.");

  (3, network)
}

fn extract_features(
  network: &CNetwork<Cf32>,
  conv_network_id: usize, 
  init_seed_val: u128,
  epochs: usize
) {
  /* intercept to inspect feature maps or intermediate activations. */
  let train_data_file = &mut File::open("./minist/train-images.idx3-ubyte").unwrap();
  let train_label_file = &mut File::open("./minist/train-labels.idx1-ubyte").unwrap();
  let batch_size = 60;
  let ref mut tracker = 0;
  let data: Dataset<Cf32, Cf32> = Dataset::minist_as_complex_batch(train_data_file, train_label_file, batch_size, tracker);
  
  // it is a 6
  let (image_point1, _) = data.get_point(18);
  // it is a 4
  let (image_point2, _) = data.get_point(58);

  match image_point1 {
    IOType::Matrix(map) => { 
      map[0].to_csv(format!(
        "./out/complex_features/conv{}/{}_2_{}e_original.csv", 
        conv_network_id, init_seed_val, epochs
      )).unwrap();
    },
    _ => {panic!("ups...")}
  }

  match image_point2 {
    IOType::Matrix(map) => { 
      map[0].to_csv(format!(
        "./out/complex_features/conv{}/{}_4_{}e_original.csv", 
        conv_network_id, init_seed_val, epochs
      )).unwrap();
    },
    _ => {panic!("ups...")}
  }

  if conv_network_id == 1 {
    unimplemented!("You really want results for 1 layer?")
  } else if conv_network_id == 2 {
    // point 1
    let order = 1;
    let (feature_maps, _) = network.intercept(image_point1.clone(), order).unwrap();
    match feature_maps {
      IOType::Matrix(maps) => {
        maps.into_iter().enumerate().for_each(|(id, feature)| {
          feature.to_csv(format!(
            "out/complex_features/conv{}/{}_2_{}e_feature_{}_{}.csv", 
            conv_network_id, init_seed_val, epochs, id, order
          )).unwrap();
        })
      },
      _ => { panic!("nope, not this feature") }
    }

    let order = 3;
    let (feature_maps, _) = network.intercept(image_point1.clone(), order).unwrap();
    match feature_maps {
      IOType::Matrix(maps) => {
        maps.into_iter().enumerate().for_each(|(id, feature)| {
          feature.to_csv(format!(
            "out/complex_features/conv{}/{}_2_{}e_feature_{}_{}.csv", 
            conv_network_id, init_seed_val, epochs, id, order
          )).unwrap();
        })
      },
      _ => { panic!("nope, not this feature") }
    }

    // point 2
    let order = 1;
    let (feature_maps, _) = network.intercept(image_point2.clone(), order).unwrap();
    match feature_maps {
      IOType::Matrix(maps) => {
        maps.into_iter().enumerate().for_each(|(id, feature)| {
          feature.to_csv(format!(
            "out/complex_features/conv{}/{}_4_{}e_feature_{}_{}.csv", 
            conv_network_id, init_seed_val, epochs, id, order
          )).unwrap();
        })
      },
      _ => { panic!("nope, not this feature") }
    }

    let order = 3;
    let (feature_maps, _) = network.intercept(image_point2.clone(), order).unwrap();
    match feature_maps {
      IOType::Matrix(maps) => {
        maps.into_iter().enumerate().for_each(|(id, feature)| {
          feature.to_csv(format!(
            "out/complex_features/conv{}/{}_4_{}e_feature_{}_{}.csv", 
            conv_network_id, init_seed_val, epochs, id, order
          )).unwrap();
        })
      },
      _ => { panic!("nope, not this feature") }
    }
  }
}

fn _test_pipeline(
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
    
    // Performance Metrics
    let inst_test_loss = network.loss(&test_data, loss_func).unwrap();
    let inst_test_acc = network.max_pred_test(&test_data);
    
    mean_test_loss += inst_test_loss;
    mean_test_acc += inst_test_acc;

    let accu = (b+1) as f32;
    print!(
      "\r| Batch {} -> Loss: ({:.4}, {:.4}), Accuracy: ({:.4}, {:.4}) (time: {:.3?})", 
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

fn _train_pipeline(
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

    // Performance Metrics First from previous batch (memory efficient)
    let inst_train_loss = network.loss(&train_data, &loss_func).unwrap();
    let inst_train_acc = network.max_pred_test(&train_data);

    mean_train_loss += inst_train_loss;
    mean_train_acc += inst_train_acc;

    let accu = (b+1) as f32;
    print!(
      "\r| Batch {} -> Loss: ({:.4}, {:.4}), Accuracy: ({:.4}, {:.4})", 
      b+1, inst_train_loss, mean_train_loss/accu, inst_train_acc, mean_train_acc/accu
    );

    // Train
    network.gradient_opt(train_data, &loss_func, lr).unwrap();

    print!(" (time: {:.3?})", t_train.elapsed());
    io::stdout().flush().unwrap();
  }

  println!();
  mean_train_loss /= train_batches as f32;
  mean_train_acc /= train_batches as f32;
  train_loss_vec.push(mean_train_loss);
  train_acc_vec.push(mean_train_acc);
}

fn _signal_pipeline(
  network: &mut CNetwork<Cf32>,
  loss_func: &ComplexLossFunc,
  lr: Cf32,
  samples: usize,
  noise_thr: f32,
  batches: usize,
  batch_size: usize,
  mut train_gen_seed: u128,
  mut test_gen_seed: u128,
  train_loss_vec: &mut Vec<f32>,
  test_loss_vec: &mut Vec<f32>,
) {
  // this pipeline does 1 epoch

  let train_seed = &mut train_gen_seed; // 223895746827_u128;
  let test_seed = &mut  test_gen_seed; // 4346877248692_u128;

  let mut mean_train_loss = 0.0;
  let mut mean_test_loss = 0.0;
  for b in 0..batches {
    let batch_time: Instant = Instant::now();

    let (train_batch, test_batch) = Dataset::signal_reconstruction(
      samples,
      batch_size, 
      noise_thr, 
      train_seed, 
      test_seed
    );
  
    let accu = (b+1) as f32;

    let inst_loss = network.loss(&train_batch, loss_func).unwrap();
    mean_train_loss = ( mean_train_loss + inst_loss ) / accu;

    //let mean_loss = mean_train_loss/accu;
    // work from the initial value lr
    //let mut lr= lr;
    // automatic decay
    //if mean_loss < 0.0500 {
    //  lr = lr / Cf32::new((9.9*mean_loss).powi(4), 0.0);
    //}

    network.gradient_opt(train_batch, loss_func, lr).unwrap();

    let inst_tloss = network.loss(&test_batch, loss_func).unwrap();

    mean_test_loss = ( mean_test_loss + inst_tloss ) / accu;

    print!(
      "\r| Batch {} -> Loss: ({:.4}, {:.4}), Val Loss: ({:.4}, {:.4}) (time: {:.3?})", 
      b+1, 
      inst_loss, mean_train_loss, 
      inst_tloss, mean_test_loss, 
      batch_time.elapsed()
    );
    io::stdout().flush().unwrap();
  }

  println!();

  train_loss_vec.push(mean_train_loss);
  test_loss_vec.push(mean_test_loss);
}

fn main() {
  // ANSI escape codes
  let reset = "\x1b[0m";
  let bold = "\x1b[1m";
  let _red = "\x1b[31m";
  let _green = "\x1b[32m";
  let yellow = "\x1b[33m";
  let underline = "\x1b[4m";
    
  let mut seeds = [
    891298565,
    918232853,
    328557473,
    348769349,
    224783561,
    981347827
  ];

  let ref mut seed = seeds[4];
  let init_seed_val = *seed;
  println!("Using seed: {}", init_seed_val);

  // for generating datasets
  let train_gen_seed = 223895746827_u128;
  let test_gen_seed = 4346877248692_u128;

  // for signal applications
  let samples = 512;
  let noise_thr = 25e-3;
  let _batches = 200;

  let (network_id, mut network) = _get_2conv_layer_cvcnn(seed);
  println!("Created the Network.");

  let mut train_loss_vec: Vec<f32> = Vec::new();
  let mut test_loss_vec: Vec<f32> = Vec::new();

  // classification vectors
  let mut _train_acc_vec: Vec<f32> = Vec::new();
  let mut _test_acc_vec: Vec<f32> = Vec::new();

  let batch_size = 100;
  let epochs: usize = 16; // 32 for dense

  // MNIST Data
  let _total_train_data = 60000;
  let _total_test_data = 10000;
  let _train_batches = _total_train_data / batch_size;
  let _test_batches = _total_test_data / batch_size;

  /* lr
  Dense : 3 or 4
  Signal: 0.0050 (could progress) 0.01 (slightly better)
  */
  let r = 7500e-4_f32;
  let phase = 0.0 * std::f32::consts::PI / 100.0;
  let lr_re = r * phase.cos();
  let lr_im = r * phase.sin();
  let lr = Cf32::new(lr_re, lr_im);
  let loss_func = ComplexLossFunc::MeanSquare;

  println!("Begining training and testing pipeline.");
  println!("Using constant learning rate: [{:.4}, {:.4}]", lr_re, lr_im);
  println!("Loss Function: {:?}", loss_func);
  println!("Total Epochs: {}", epochs);
  //println!("Batch size: {} | Number of Batches (train, test): {:?}", batch_size, (train_batches, test_batches));
  for e in 0..epochs {
    println!();
    println!("{}Epoch {}/{}{}",bold, e+1, epochs, reset);
    // Pipeline for signal reconstruction
    /*
    println!("| {}{}{}Signal Pipeline{}", bold, underline, yellow, reset);
    signal_pipeline(
      &mut network, 
      &loss_func, 
      lr, 
      samples,
      noise_thr,
      batches, 
      batch_size,
      train_gen_seed,
      test_gen_seed,
      &mut train_loss_vec, 
      &mut test_loss_vec
    );
    */

    // Pipeline for classification
    println!("| {}{}{}Train Pipeline{}", bold, underline, yellow, reset);
    _train_pipeline(
      &mut network, 
      &loss_func, 
      lr, 
      _train_batches, 
      batch_size, 
      &mut train_loss_vec, 
      &mut _train_acc_vec
    );

    println!("| {}{}{}Test Pipeline{}", bold, underline, _green, reset);
    _test_pipeline(
      &network, 
      &loss_func, 
      _test_batches, 
      batch_size, 
      &mut test_loss_vec, 
      &mut _test_acc_vec
    );  
  }
  
  println!();
  println!("Ended training.");

  /* Network exploration */
  if network_id == 1 || network_id == 2 {
    /* it is a convolutional network */
    extract_features(&network, network_id, init_seed_val, epochs);
  } else if network_id == 3 {
    let mini_batch_size = 10;
    let mut train_seed = train_gen_seed;
    let mut test_seed = test_gen_seed;
    let (train_batch, _) = Dataset::signal_reconstruction(
      samples,
      mini_batch_size, 
      noise_thr, 
      &mut train_seed, 
      &mut test_seed
    );

    for b in 0..mini_batch_size {
      let (corrupted_signal, clean_signal) = train_batch.get_point(b);
      let clean_prediction = network.forward(corrupted_signal).unwrap();

      // save results
      corrupted_signal.as_slice().to_matrix([samples, 1]).unwrap().to_csv(format!("out/signal_rec/{}_x{}_lr_{:.4}_{:.4}_signal_{}e.csv", init_seed_val, b, lr_re, lr_im, epochs)).unwrap();
      clean_signal.as_slice().to_matrix([samples, 1]).unwrap().to_csv(format!("out/signal_rec/{}_y{}_lr_{:.4}_{:.4}_signal_{}e.csv", init_seed_val, b, lr_re, lr_im, epochs)).unwrap();
      clean_prediction.as_slice().to_matrix([samples, 1]).unwrap().to_csv(format!("out/signal_rec/{}_yp{}_lr_{:.4}_{:.4}_signal_{}e.csv", init_seed_val, b, lr_re, lr_im, epochs)).unwrap();
    }
  }

  /* Save Results (learning curves) to local memory as .csv file */
  Matrix::from_body(train_loss_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/{}_loss_{:.4}_{:.4}_{}e.csv", 
    network_id, init_seed_val, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(_train_acc_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/{}_accu_{:.4}_{:.4}_{}e.csv", 
    network_id, init_seed_val, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(test_loss_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/{}_tloss_{:.4}_{:.4}_{}e.csv", 
    network_id, init_seed_val, lr_re, lr_im, epochs
  )).unwrap();
  Matrix::from_body(_test_acc_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/{}_taccu_{:.4}_{:.4}_{}e.csv", 
    network_id, init_seed_val, lr_re, lr_im, epochs
  )).unwrap();

  println!("Saved relevant data.");
  println!("Ended Pipeline.")
}
