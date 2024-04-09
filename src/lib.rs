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
  use std::time::Duration;
  use std::thread;
  use std::io::{Read, Write};
  use crate::act::ComplexActFunc;
  use crate::cvnn::layer::dense::DenseCLayer;
  use crate::cvnn::network::CNetwork;
  use crate::dataset::Dataset;
  use crate::init::PredictModel;
  use crate::math::matrix::Matrix;
  use crate::opt::{ComplexLossFunc, LossFunc};

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
}