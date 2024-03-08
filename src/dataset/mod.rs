mod err;

use std::{any::TypeId, fmt::Debug, fs::File, io::Write, slice::Chunks};

use crate::{init::PredictModel, math::{matrix::Matrix, random::lcgi, BasicOperations, Complex, Real}};
use err::DatasetSampleError;
//use crate::math::random::{lcgf32, lcgf64};

/// A very low-level and simple dataset representation.
/// Body contains the input that will be feed directly onto a Network.
/// Target contains the expected output to be directly used in calculating the cost function.
#[derive(Debug)]
pub struct Dataset<B, T> {
    pub body: Matrix<B>,
    /// Represents the intended output of the network, i.e. the output layer desired results.
    pub target: Matrix<T>
}

impl<B, T> Dataset<B, T> {
  pub fn rows_as_iter(&self) -> (Chunks<'_, B>, Chunks<'_, T>) {
    (self.body.rows_as_iter(), self.target.rows_as_iter())
  }
}

impl<B: Debug, T: Debug + PartialOrd + Copy> Dataset<B, T> {
  pub fn to_csv(&self) -> std::io::Result<()> {
    let mut file = File::create("data.csv")?;
    
    let data_row_len = self.body.get_shape()[1];
    let target_row_len = self.target.get_shape()[1];
    let data_chunks = self.body.get_body().chunks(data_row_len);
    let target_chunks = self.target.get_body().chunks(target_row_len);

    let mut header = vec![];
    header.append(
      &mut (0..data_row_len)
        .map(|elm| { format!("feature{}", elm) })
        .collect::<Vec<String>>()
    );
    
    header.push("class".to_string());
    writeln!(file, "{}", header.join(","))?;

    let mut string_val;
    let mut class;
    for (body, target) in data_chunks.zip(target_chunks) {
      string_val = format!("{:?}", body)
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "");
      (class, _) = target.iter().enumerate().max_by(|(_, &a), (_, &b)| { a.partial_cmp(&b).unwrap() }).unwrap();

      write!(file, "{}", string_val)?;
      writeln!(file, ",{}", class)?;
    }

    Ok(())
  }
}

impl<B: Complex + BasicOperations<B>, T: Real + BasicOperations<T>> Dataset<B, T>
  where <B as Complex>::Precision: 'static, T: 'static {
  
  /// Samples a [`Dataset`] with complex values.
  /// This artificial data has rectangular-like symmetry.
  pub fn sample_complex(
    dims: [usize; 2],
    degree: usize,
    macro_scale: usize,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<B, T>, DatasetSampleError> {

    if dims[0] < degree { return Err(DatasetSampleError::BellowMinimumSamples) }
    if TypeId::of::<B::Precision>() != TypeId::of::<T>() { return Err(DatasetSampleError::IncompatibleTargetType) }

    let mut sample_matrix: Matrix<B> = Matrix::with_capacity(dims);
      
    // spray focal points
    let mut centers: Matrix<B> = Matrix::with_capacity([degree, dims[1]]);
    let mut center: Vec<B> = Vec::with_capacity(dims[1]);
    for _ in 0..degree {
      for _ in 0..dims[1] {
        center.push(
          // random point relative to origin
          B::gen(seed, macro_scale)
        );
      }

      // add_row will clean the center vector
      centers.add_mut_row(&mut center).unwrap();
    }
    drop(center);

    let mut class_center: &[B];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut labels: Matrix<T> = Matrix::with_capacity([dims[0], degree]);
    let mut added_row: Vec<B> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();
      labels.add_mut_row(&mut one_hot_vec).unwrap();
      
      class_center = centers.row(selected_class).unwrap();

      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + B::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_matrix.add_mut_row(&mut added_row).unwrap();
    }

    Ok(
      Dataset {
        body: sample_matrix,
        target: labels
      }
    )
  }
}

impl<T: Real + BasicOperations<T>> Dataset<T, T> {
  /// Samples a [`Dataset`] with real values.
  /// This artificial data has rectangular-like symmetry.
  pub fn sample(
      dims: [usize; 2],
      degree: usize,
      macro_scale: usize,
      micro_scale: usize,
      pred_method: PredictModel,
      seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {

    if dims[0] < degree { return Err(DatasetSampleError::BellowMinimumSamples) }

    let mut sample_matrix: Matrix<T> = Matrix::with_capacity(dims);
      
    // spray focal points
    let mut centers: Matrix<T> = Matrix::with_capacity([degree, dims[1]]);
    let mut center: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..degree {
      for _ in 0..dims[1] {
        center.push(
          // random point relative to origin
          T::gen(seed, macro_scale)
        );
      }

      // add_row will clean the center vector
      centers.add_mut_row(&mut center).unwrap();
    }
    drop(center);

    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut labels: Matrix<T> = Matrix::with_capacity([dims[0], degree]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);

      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();
      labels.add_mut_row(&mut one_hot_vec).unwrap();
      
      class_center = centers.row(selected_class).unwrap();

      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + T::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_matrix.add_mut_row(&mut added_row).unwrap();
    }

    Ok(
      Dataset {
        body: sample_matrix,
        target: labels
      }
    )
  }
}
