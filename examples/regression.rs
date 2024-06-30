use std::io::{self, Write};
use std::time::Instant;

use renplex::act::ComplexActFunc;
use renplex::cvnn::layer::dense::DenseCLayer;
use renplex::cvnn::layer::CLayer;
use renplex::cvnn::network::CNetwork;
use renplex::dataset::Dataset;
use renplex::init::InitMethod;
use renplex::input::IOShape;
use renplex::math::cfloat::Cf32;
use renplex::math::matrix::{Matrix, SliceToMatrix};
use renplex::math::Complex;
use renplex::opt::ComplexLossFunc;

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

  println!("Created CVNN Auto-encoder. Params: {}.", network.params_len());

  (3, network)
}

fn signal_rec_train_test(
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

fn reg_pipeline(init_seed_val: u128, seed: &mut u128) {
  // ANSI escape codes
  let reset = "\x1b[0m";
  let bold = "\x1b[1m";
  let _red = "\x1b[31m";
  let _green = "\x1b[32m";
  let yellow = "\x1b[33m";
  let underline = "\x1b[4m";

  // for generating datasets
  let train_gen_seed = 223895746827_u128;
  let test_gen_seed = 4346877248692_u128;

  // for signal applications
  let samples = 512;
  let noise_thr = 25e-3;
  let batches = 200;

  let (network_id, mut network) = _get_auto_encoder(seed, samples);
  println!("Created the Network.");

  let mut train_loss_vec: Vec<f32> = Vec::new();
  let mut test_loss_vec: Vec<f32> = Vec::new();

  let batch_size = 100;
  let epochs: usize = 16; // 32 for dense

  /* lr
  Signal: 0.0050 (could progress) 0.01 (slightly better)
  */
  let r = 50e-4_f32;
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
    println!("| {}{}{}Signal Pipeline{}", bold, underline, yellow, reset);
    signal_rec_train_test(
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
  }
  
  println!();
  println!("Ended training.");

  /* Network exploration */
  if network_id == 3 {
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
  Matrix::from_body(test_loss_vec, [epochs, 1]).to_csv(format!(
    "./out/{}/{}_tloss_{:.4}_{:.4}_{}e.csv", 
    network_id, init_seed_val, lr_re, lr_im, epochs
  )).unwrap();

  println!("Saved relevant data.");
  println!("Ended Pipeline.")
}



fn main() {
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

  reg_pipeline(init_seed_val, seed)
}
