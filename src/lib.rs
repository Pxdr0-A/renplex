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
  use std::fs::File;
  use std::time::Instant;
  use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
  use rayon::ThreadPoolBuilder;
  use crate::dataset::Dataset;
  use crate::input::IOType;
  use crate::math::matrix::Matrix;
  use crate::math::matrix::SliceOps;

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

    let c = &[2.0, 3.0, 1.0].scalar_prod(&[1.0, 2.0, 2.0]).unwrap();
    println!("{:?}", c);
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
  fn pooling() {
    let train_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    let batch_size = 10;
    let ref mut tracker = 0;

    let data: Dataset<f32, f32> = Dataset::minist_as_batch(train_data_file, train_label_file, batch_size, tracker);

    let (image_point, _) = data.get_point(4);

    let image;
    match image_point {
      IOType::FeatureMaps(map) => {
        image = map[0].clone();
      },
      _ => {panic!("ups...")}
    }
    image.to_csv("./out/original_pool.csv".to_string()).unwrap();

    let image_max_pooled = image.block_reduce(
      &[2, 2],
      |slice| {
        slice
          .iter()
          .fold(-f32::INFINITY,|acc, elm| {
            if acc < *elm { 
              *elm
            } else {
              acc
            }
          })
      }
    ).unwrap();

    image_max_pooled.to_csv("./out/max_pool.csv".to_string()).unwrap();

    let image_upsampled = image_max_pooled.fractional_upsampling(
      &[2,2], 
      &Matrix::from_body(
        vec![
          0.25, 0.5, 0.25,
          0.5, 1.0, 0.5,
          0.25, 0.5, 0.25
        ], 
        [3,3])
    ).unwrap();

    image_upsampled.to_csv("./out/max_pool_upsampled.csv".to_string()).unwrap();
  }

  #[test]
  fn image_conv_test() {
    let train_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    let batch_size = 100;
    let ref mut tracker = 0;

    let data: Dataset<f32, f32> = Dataset::minist_as_batch(train_data_file, train_label_file, batch_size, tracker);

    let (image_point, _) = data.get_point(50);

    let image;
    match image_point {
      IOType::FeatureMaps(map) => {
        image = map[0].clone();
      },
      _ => {panic!("ups...")}
    }
    image.to_csv("./out/original.csv".to_string()).unwrap();

    let kernel1: Matrix<f32> = Matrix::from_body(
      vec![
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
      ], [3, 3]);
    let inv_kernel1: Matrix<f32> = Matrix::from_body(
      vec![
        -1.0, -2.0, -1.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 1.0,
      ], [3, 3]);

    let kernel2: Matrix<f32> = Matrix::from_body(
      vec![
        -1.0, -2.0, -1.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 1.0,
      ], [3, 3]);

    let mut image_conv1 = image.conv(&kernel1).unwrap();
    let mut image_rev = image_conv1.deconv(&kernel1).unwrap();
    let mut image_trans_rev = image_conv1.conv(&inv_kernel1).unwrap();
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

    for row in image_rev.rows_as_iter_mut() {
      for elm in row {
        if elm.is_sign_negative() {
          *elm = 0.0;
        }
      }
    }

    for row in image_trans_rev.rows_as_iter_mut() {
      for elm in row {
        if elm.is_sign_negative() {
          *elm = 0.0;
        }
      }
    }

    image_conv1.to_csv("./out/conv_image.csv".to_string()).unwrap();
    image_conv2.to_csv("./out/conv_image1.csv".to_string()).unwrap();
    image_rev.to_csv("./out/conv_image_rev.csv".to_string()).unwrap();
    image_trans_rev.to_csv("out/conv_image_rev1.csv".to_string()).unwrap();
  }

  #[test]
  fn image_dconv_test() {
    let train_data_file = &mut File::open("./minist/t10k-images.idx3-ubyte").unwrap();
    let train_label_file = &mut File::open("./minist/t10k-labels.idx1-ubyte").unwrap();
    let batch_size = 100;
    let ref mut tracker = 0;

    let data: Dataset<f32, f32> = Dataset::minist_as_batch(train_data_file, train_label_file, batch_size, tracker);

    let (image_point, _) = data.get_point(50);

    let image;
    match image_point {
      IOType::FeatureMaps(map) => {
        image = map[0].clone();
      },
      _ => {panic!("ups...")}
    }
    image.to_csv("./out/original_d.csv".to_string()).unwrap();

    let dimage = image.dconv((0,2), &[3,3]).unwrap();

    dimage.to_csv("./out/dconv_image.csv".to_string()).unwrap();

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
}