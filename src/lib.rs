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

use crate::dataset::Dataset;
  use crate::init::PredictModel;
use crate::math::matrix::Matrix;


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

    let (image_point, _) = data.get_point(0);

    let image = Matrix::from_body(image_point.to_vec(), [28, 28]);
    image.to_csv("./out/original.csv").unwrap();

    let kernel: Matrix<f32> = Matrix::from_body(
      vec![
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
      ], [3, 3]);

    let image_conv = image.conv(&kernel).unwrap();
    image_conv.to_csv("./out/conv_image.csv").unwrap();
  }
}