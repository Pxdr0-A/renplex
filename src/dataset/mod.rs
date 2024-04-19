mod err;

use std::fmt::Display;
use std::{fmt::Debug, fs::File, slice::Iter, vec::IntoIter};
use std::io::{Read, Write};

use crate::{init::PredictModel, input::IOType, math::{matrix::Matrix, random::lcgi, BasicOperations, Complex, Real}};
use err::DatasetSampleError;
//use crate::math::random::{lcgf32, lcgf64};

/// A very low-level and simple dataset representation.
/// Body contains the input that will be feed directly onto a Network.
/// Target contains the expected output to be directly used in calculating the cost function.
#[derive(Debug, Clone)]
pub struct Dataset<B, T> {
    inputs: Vec<IOType<B>>,
    /// Represents the intended output of the network, i.e. the output layer desired results.
    target: Vec<IOType<T>>
}

impl<T: Display + Copy> Dataset<T, T> {
  pub fn to_csv(&self) -> std::io::Result<()> {
    let mut file = File::create("dataset.csv")?;

    let (inputs_iter, targets_iter) = self.points_as_iter();
    let mut string_row_input;
    let mut string_row_target;
    for (input, target) in inputs_iter.zip(targets_iter) {
      let mut input_iter = input.to_vec().into_iter();
      write!(file, "{}", input_iter.next().unwrap())?;
      for elm_input in input_iter {
        string_row_input = format!(",{}", elm_input);
        write!(file, "{}", string_row_input)?;
      }
      for elm_input in target.to_vec().into_iter() {
        string_row_target = format!(",{}", elm_input);
        write!(file, "{}", string_row_target)?;
      }

      writeln!(file, "")?;
    }

    Ok(())
  }
}

impl<B, T> Dataset<B, T> {
  pub fn new() -> Dataset<B, T> {
    Dataset { inputs: Vec::new(), target: Vec::new() }
  }

  pub fn get_n_points(&self) -> usize {
    self.inputs.len()
  }

  pub fn get_point(&self, i: usize) -> (&IOType<B>, &IOType<T>) {
    (&self.inputs[i], &self.target[i])
  }

  pub fn points_as_iter(&self) -> (Iter<'_, IOType<B>>, Iter<'_, IOType<T>>) {
    (self.inputs.iter(), self.target.iter())
  }

  pub fn points_into_iter(self) -> (IntoIter<IOType<B>>, IntoIter<IOType<T>>) {
    (self.inputs.into_iter(), self.target.into_iter())
  }

  pub fn add_point(&mut self, point: (IOType<B>, IOType<T>)) {
    self.inputs.push(point.0);
    self.target.push(point.1);
  }
}

const MINIST_DEGREE: usize = 10;
const MINIST_TRAIN_POINTS: usize = 60000;
const _MINIST_TEST_POINTS: usize = 10000;
const MINIST_TRAIN_LABEL_SIZE: usize = MINIST_TRAIN_POINTS + 8;
const _MINIST_TEST_LABEL_SIZE: usize = 4542;
const MINIST_IMAGE_DIMS: (usize, usize) = (28, 28);

impl<T: Real + BasicOperations<T>> Dataset<T, T> {
  /// To extract a batch from MINIST tarining dataset.
  pub fn minist_as_batch(data_file: &mut File, label_file: &mut File, batch_size: usize, tracker: &mut usize) -> Dataset<T, T> {
    /* batch_size needs to be a multiple of the data or not? */
    /* have not read a single byte */
    if *tracker == 0 {
      /* skip first 16 bytes of training image file */
      data_file.read(&mut [0u8; 16]).unwrap();
      /* skip first 8 bytes of training label file */
      *tracker += label_file.read(&mut [0u8; 8]).unwrap();
    }
    
    let ref mut image_buffer = [0u8; MINIST_IMAGE_DIMS.0 * MINIST_IMAGE_DIMS.1];
    let ref mut label_buffer = [0u8; 1];

    let mut data_batch = Dataset::new();


    for _ in 0..batch_size {
      data_file.read(image_buffer).unwrap();
      *tracker += label_file.read(label_buffer).unwrap();
      let image = image_buffer.iter().map(|elm| { T::usize_to_real(*elm as usize) / T::usize_to_real(255) }).collect::<Vec<T>>();
      data_batch.add_point((
        IOType::FeatureMaps(vec![Matrix::from_body(image, [28, 28]); 1]),
        IOType::Vector(T::gen_pred(MINIST_DEGREE, label_buffer[0] as usize, &PredictModel::Sparse).unwrap())
      ));

      if *tracker == MINIST_TRAIN_LABEL_SIZE {
        break;
      }
    }

    data_batch
  }
}

impl<T: Complex + BasicOperations<T>> Dataset<T, T> {
  /// To extract a batch from MINIST tarining dataset.
  pub fn minist_as_complex_batch(data_file: &mut File, label_file: &mut File, batch_size: usize, tracker: &mut usize) -> Dataset<T, T> {
    /* batch_size needs to be a multiple of the data or not? */
    /* have not read a single byte */
    if *tracker == 0 {
      /* skip first 16 bytes of training image file */
      data_file.read(&mut [0u8; 16]).unwrap();
      /* skip first 8 bytes of training label file */
      *tracker += label_file.read(&mut [0u8; 8]).unwrap();
    }
    
    let ref mut image_buffer = [0u8; MINIST_IMAGE_DIMS.0 * MINIST_IMAGE_DIMS.1];
    let ref mut label_buffer = [0u8; 1];

    let mut data_batch = Dataset::new();

    for _ in 0..batch_size {
      data_file.read(image_buffer).unwrap();
      *tracker += label_file.read(label_buffer).unwrap();
      let image = image_buffer.iter().map(|elm| { T::usize_to_complex(*elm as usize) / T::usize_to_complex(255) }).collect::<Vec<T>>();
      data_batch.add_point((
        IOType::FeatureMaps(vec![Matrix::from_body(image, [28, 28]); 1]),
        IOType::Vector(T::gen_pred(MINIST_DEGREE, label_buffer[0] as usize, &PredictModel::Sparse).unwrap())
      ));

      if *tracker == MINIST_TRAIN_LABEL_SIZE {
        break;
      }
    }

    data_batch
  }
}

/* Synthetic Dataset utilities */
impl<T: Real + BasicOperations<T>> Dataset<T, T> {
  pub fn gen_centers(features: usize, degree: usize, scale: usize, seed: &mut u128) -> Matrix<T> {
    // spray focal points
    let mut centers: Matrix<T> = Matrix::with_capacity([degree, features]);
    let mut center: Vec<T> = Vec::with_capacity(features);
    for _ in 0..degree {
      for _ in 0..features {
        center.push(
          // random point relative to origin
          T::gen(seed, scale)
        );
      }

      // add_row will clean the center vector
      centers.add_mut_row(&mut center).unwrap();
    }

    centers
  }

  /// Samples a [`Dataset`] with real values with vector inputs and vector outputs.
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
      
    // spray focal points
    let centers = Dataset::<T, T>::gen_centers(dims[1], degree, macro_scale, seed);

    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();

      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          ( class_center[col] + T::gen(seed, micro_scale) ) / T::usize_to_real(macro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_body.push(IOType::Vector(added_row.clone()));
      labels.push(IOType::Vector(one_hot_vec.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }

  pub fn gen_batch_sample(
    dims: [usize; 2],
    centers: Matrix<T>,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {
    
    let centers_shape = centers.get_shape();
    let degree = centers_shape[0];
    let _features = centers_shape[1];
    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();

      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + T::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_body.push(IOType::Vector(added_row.clone()));
      labels.push(IOType::Vector(one_hot_vec.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }
}

impl<T: Complex + BasicOperations<T>> Dataset<T, T> {
  pub fn gen_complex_centers(features: usize, degree: usize, scale: usize, seed: &mut u128) -> Matrix<T> {
    // spray focal points
    let mut centers: Matrix<T> = Matrix::with_capacity([degree, features]);
    let mut center: Vec<T> = Vec::with_capacity(features);
    for _ in 0..degree {
      for _ in 0..features {
        center.push(
          // random point relative to origin
          T::gen(seed, scale)
        );
      }

      // add_row will clean the center vector
      centers.add_mut_row(&mut center).unwrap();
    }

    centers
  }

  /// Samples a [`Dataset`] with complex values with vector inputs and vector outputs.
  /// This artificial data has rectangular-like symmetry.
  pub fn sample_complex(
    dims: [usize; 2],
    degree: usize,
    macro_scale: usize,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {

    if dims[0] < degree { return Err(DatasetSampleError::BellowMinimumSamples) }
      
    let centers = Dataset::<T, T>::gen_complex_centers(dims[1], degree, macro_scale, seed);

    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();
      
      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          ( class_center[col] + T::gen(seed, micro_scale) ) / T::usize_to_complex(macro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      labels.push(IOType::Vector(one_hot_vec.clone()));
      sample_body.push(IOType::Vector(added_row.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }

  pub fn gen_complex_batch_sample(
    dims: [usize; 2],
    centers: Matrix<T>,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {
    let centers_shape = centers.get_shape();
    let degree = centers_shape[0];
    let _features = centers_shape[1];
    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();

      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + T::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_body.push(IOType::Vector(added_row.clone()));
      labels.push(IOType::Vector(one_hot_vec.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }
}
