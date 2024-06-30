pub mod math;
pub mod dataset;
pub mod input;
pub mod act;
pub mod opt;
pub mod init;
pub mod err;
pub mod cvnn;

#[cfg(test)]
mod basic_tests {
  use std::f32::consts::PI;
  use std::fs::File;
  use std::time::Instant;
  use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
  use rayon::ThreadPoolBuilder;
  use crate::dataset::Dataset;
  use crate::input::IOType;
  use crate::math::cfloat::Cf32;
  use crate::math::matrix::Matrix;
  use crate::math::matrix::SliceOps;
  use crate::math::Complex;

  #[test]
  fn matrix_add() {
    let mut matrix1 = Matrix::from_body(
      vec![
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0
      ], [4, 4]);
    
    let matrix2 = Matrix::from_body(
      vec![
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0
      ], [4, 4]);
    
    matrix1.add_mut(&matrix2).unwrap();
    println!("{}", matrix1);
    matrix1.add_mut_scalar(2.0).unwrap();
    println!("{}", matrix1);
  }

  #[test]
  fn matrix_mul() {
    let mut matrix1 = Matrix::from_body(
      vec![
        2.0, 1.0, 1.2,
        1.5, 5.0, 1.0,
        9.3, 1.0, 0.0
      ], [3, 3]);

    matrix1.mul_mut_scalar(2.0).unwrap();
    println!("{}", matrix1);
    matrix1.div_mut_scalar(2.0).unwrap();
    println!("{}", matrix1);
    
    let res = matrix1.mul_vec(vec![1.0, 2.0, 0.0]).unwrap();
    let res1 = matrix1.mul_slice(&vec![1.0, 2.0, 0.0]).unwrap();

    println!("{:?}", res);
    println!("{:?}", res1);
  }

  #[test]
  fn vec_add() {
    let mut vec1 = vec![2.0, 1.0, 2.0];

    vec1.add_slice_mut(&[1.0, 2.0, 3.0]).unwrap();
    println!("{:?}", vec1);
    let mut a = vec1.add_slice(&[1.0, 2.0, 3.0]).unwrap();
    println!("{:?}", a);
    a.mul_mut_scalar(2.0).unwrap();
    println!("{:?}", a);
    a.mul_slice_mut(&[2.0, 0.5, 3.0]).unwrap();
    println!("{:?}", a);
    let b = a.mul_slice(&[0.5, 2.0, 0.333]).unwrap();
    println!("{:?}", b);

    let c = &[2.0, 3.0, 1.0].scalar_prod(&[1.0, 2.0, 2.0]);
    println!("{:?}", c);
  }

  #[test]
  fn image_conv_test() {
    let train_data_file = &mut File::open("./minist/train-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/train-labels.idx1-ubyte").unwrap();
    let batch_size = 100;
    let ref mut tracker = 0;

    let data: Dataset<f32, f32> = Dataset::minist_as_batch(train_data_file, train_label_file, batch_size, tracker);

    let (image_point, _) = data.get_point(58);

    let image;
    match image_point {
      IOType::Matrix(map) => {
        image = map[0].clone();
      },
      _ => {panic!("ups...")}
    }

    image.to_csv("./out/conv_tests/original.csv".to_string()).unwrap();

    let image_convx = image.convolution(
      &Matrix::from_body(
        vec![
          1.0, 0.0, -1.0,
          2.0, 0.0, -2.0,
          1.0, 0.0, -1.0
        ], 
        [3,3])
    ).unwrap();

    let image_convy = image.convolution(
      &Matrix::from_body(
        vec![
          1.0, 2.0, 1.0,
          0.0, 0.0, 0.0,
          -1.0, -2.0, -1.0
        ], 
        [3,3])
    ).unwrap();

    let image_conv = image_convx
      .get_body()
      .iter()
      .zip(image_convy.get_body().iter())
      .map(|(elmx, elmy)| { ( elmx.powi(2) + elmy.powi(2) ).sqrt() })
      .collect::<Vec<_>>();

    let complex_img = image_convx
      .get_body()
      .iter()
      .zip(image_convy.get_body().iter())
      .map(|(elmx, elmy)| { 
        let norm = ( elmx.powi(2) + elmy.powi(2) ).sqrt();
        let phase;
        if *elmx == 0.0 { phase = PI; }
        else { phase = elmy.atan2(*elmx); }

        Cf32::new(norm, phase)
      }).collect::<Vec<_>>();

    let image_conv = Matrix::from_body(image_conv, [26, 26]);
    let image_cconv = Matrix::from_body(complex_img, [26, 26]);

    image_conv.to_csv("./out/conv_tests/conv_sobel.csv".to_string()).unwrap();
    image_cconv.to_csv("./out/conv_tests/conv_csobel.csv".to_string()).unwrap();
  }

  #[test]
  fn image_pool_test() {
    let train_data_file = &mut File::open("./minist/train-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/train-labels.idx1-ubyte").unwrap();
    let batch_size = 100;
    let ref mut tracker = 0;

    let data: Dataset<f32, f32> = Dataset::minist_as_batch(train_data_file, train_label_file, batch_size, tracker);

    let (image_point, _) = data.get_point(58);

    let image;
    match image_point {
      IOType::Matrix(map) => {
        image = map[0].clone();
      },
      _ => {panic!("ups...")}
    }

    image.to_csv("./out/conv_tests/original.csv".to_string()).unwrap();

    let image_avg_pooled = image.block_reduce(
      &[2, 2],
      |block: &[f32]| { 
        let block_len = block.len(); 
        block
          .into_iter()
          .fold(f32::default(), |acc, elm| { acc + *elm }) / (block_len as f32)
      }
    ).unwrap();

    image_avg_pooled.to_csv("./out/conv_tests/avg_pool.csv".to_string()).unwrap();

    let image_upsampled = image_avg_pooled.fractional_upsampling(
      &[2,2], 
      &Matrix::from_body(
        vec![
          0.25, 0.5, 0.25,
          0.5, 1.0, 0.5,
          0.25, 0.5, 0.25
        ], 
        [3,3])
    ).unwrap();

    image_upsampled.to_csv("./out/conv_tests/avg_pool_upsampled.csv".to_string()).unwrap();
  }

  #[test]
  fn paralelize() {
    let pool = ThreadPoolBuilder::new()
      .num_threads(8)
      .build()
      .unwrap();

    let _n = pool.install(|| {true});
  }

  #[test]
  fn par_test() {
    const LEN: usize = 2_usize.pow(15);
    let lhs = &mut [1; LEN];
    let rhs = &mut [1; LEN];
    
    let now = Instant::now();
    lhs
      .into_iter()
      .zip(rhs.iter())
      .for_each(|(elm, other)| { *elm *= *other; });
    println!("Without {:?}", now.elapsed());

    let now = Instant::now();
    lhs
      .into_par_iter()
      .zip(rhs.par_iter())
      .for_each(|(elm, other)| { *elm *= *other; });
    println!("With {:?}", now.elapsed());
  }

  #[test]
  fn unsigned() {
    let u1: usize = 10;
    let u2: usize = 9;

    let u = u1 - u2;

    println!("{}", u);
  }

  #[test]
  fn string_test() {
    let _a = format!("./out/loss_{}_{}.csv", 0.3, 0.1);
  }

  #[test]
  fn atan_test() {
    let x = ((-1.0_f32) / (1.0)).atan();
    let y = (-1.0_f32).atan2(-1.0);
    println!("{}", x);
    println!("{}", y);
  }

  #[test]
  fn fancy_prints() {
    // ANSI escape codes
    let reset = "\x1b[0m";
    let bold = "\x1b[1m";
    let red = "\x1b[31m";
    let green = "\x1b[32m";
    let yellow = "\x1b[33m";
    let blue = "\x1b[34m";

    // Example usage
    println!("{}This is bold text{}", bold, reset);
    println!("{}This is red text{}", red, reset);
    println!("{}This is green text{}", green, reset);
    println!("{}This is yellow text{}", yellow, reset);
    println!("{}This is blue text{}", blue, reset);

    // Combining styles
    println!("{}{}This is bold and red text{}",
            bold, red, reset);

    // More ANSI escape codes
    let underline = "\x1b[4m";
    let bright_red = "\x1b[91m";
    let bg_yellow = "\x1b[43m";

    // Example usage
    println!("{}{}This is underlined text{}", underline, bright_red, reset);
    println!("{}{}This has a yellow background{}", bg_yellow, bright_red, reset);
    println!("{}{}{}This is bold, underlined, and bright red with yellow background{}", bold, underline, bright_red, reset);
  }

  #[test]
  fn signal_data() {
    let train_seed = &mut 223895746827_u128;
    let test_seed = &mut 4346877248692_u128;
    
    let (train_batch, test_batch) = Dataset::signal_reconstruction(
      512,
      200 * 100,
      25e-3, 
      train_seed,
      test_seed
    );

    train_batch.to_csv("train.csv".to_string()).unwrap();
    test_batch.to_csv("test.csv".to_string()).unwrap();
  }

  #[test]
  fn hyper_tan() {
    let a = 1000000000.0_f32.tanh();

    use crate::math::Complex;
    use crate::math::cfloat::Cf32;

    let z1 = Cf32::newe(100.0, PI / 3.0);

    println!("{}, {}", a, z1);
  }
}