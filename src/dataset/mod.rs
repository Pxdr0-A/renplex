/// Module containing simple dataset utilities.


use std::fmt::Display;
use std::{fmt::Debug, fs::File, slice::Iter, vec::IntoIter};
use std::io::{Read, Write};

use crate::math::cfloat::Cf32;
use crate::math::random::{lcgf32, lcgi};
use crate::{init::PredictModel, input::IOType, math::{matrix::Matrix, BasicOperations, Complex, Real}};
//use crate::math::random::{lcgf32, lcgf64};

/// A simple representation for a batch of data with independent and dependent variable.
/// Body contains the independent variable, i.e. input that will be feed directly onto a Network.
/// Target contains the independent variable i.e. expected output to be directly used in calculating the cost function.
#[derive(Debug, Clone)]
pub struct Dataset<B, T> {
    inputs: Vec<IOType<B>>,
    /// Represents the intended output of the network, i.e. the output layer desired results.
    target: Vec<IOType<T>>
}

impl<T: Display + Copy> Dataset<T, T> {
  /// Exports a `Dataset` struct to a csv file.
  pub fn to_csv(&self, name: String) -> std::io::Result<()> {
    let mut file = File::create(name)?;

    let (inputs_iter, targets_iter) = self.points_as_iter();
    let mut string_row_input;
    let mut string_row_target;

    for (input, target) in inputs_iter.zip(targets_iter) {
      let mut input_iter = input.as_slice().into_iter();
      write!(file, "{}", input_iter.next().unwrap())?;
      for elm_input in input_iter {
        string_row_input = format!(",{}", elm_input);
        write!(file, "{}", string_row_input)?;
      }
      for elm_input in target.as_slice().into_iter() {
        string_row_target = format!(",{}", elm_input);
        write!(file, "{}", string_row_target)?;
      }

      writeln!(file, "")?;
    }

    Ok(())
  }
}

impl<B, T> Dataset<B, T> {
  /// Creates a new empty [`Dataset`] struct.
  pub fn new() -> Dataset<B, T> {
    Dataset { inputs: Vec::new(), target: Vec::new() }
  }

  /// Returns the number of points in the data batch.
  pub fn get_n_points(&self) -> usize {
    self.inputs.len()
  }

  /// Returns a reference to the i-th point inside the data batch.
  /// 
  /// # Arguments
  /// 
  /// * `i` - index of the i-th point in the data batch.
  pub fn get_point(&self, i: usize) -> (&IOType<B>, &IOType<T>) {
    (&self.inputs[i], &self.target[i])
  }

  /// Returns an iterator over all body and target points on the data batch.
  pub fn points_as_iter(&self) -> (Iter<'_, IOType<B>>, Iter<'_, IOType<T>>) {
    (self.inputs.iter(), self.target.iter())
  }

  /// Consumes the dataset and returns an iterator over all body and target points on the data batch.
  pub fn points_into_iter(self) -> (IntoIter<IOType<B>>, IntoIter<IOType<T>>) {
    (self.inputs.into_iter(), self.target.into_iter())
  }

  /// Adds a point to the data batch.
  /// 
  /// # Argument
  /// 
  /// * `point` - point containing independent and dependent variable sample to be added on the data batch.
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
  /// To extract a batch of real pixels from MINIST training or test dataset.
  pub fn minist_as_batch(
    data_file: &mut File, 
    label_file: &mut File, 
    batch_size: usize, 
    tracker: &mut usize
  ) -> Dataset<T, T> {
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
        IOType::Matrix(vec![Matrix::from_body(image, [28, 28]); 1]),
        IOType::Scalar(T::gen_pred(MINIST_DEGREE, label_buffer[0] as usize, &PredictModel::Sparse).unwrap())
      ));

      if *tracker == MINIST_TRAIN_LABEL_SIZE {
        break;
      }
    }

    data_batch
  }
}

impl<T: Complex + BasicOperations<T>> Dataset<T, T> {
  /// To extract a batch of imaginary (with only real part) pixels from MINIST training or test dataset.
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
        IOType::Matrix(vec![Matrix::from_body(image, [28, 28]); 1]),
        IOType::Scalar(T::gen_pred(MINIST_DEGREE, label_buffer[0] as usize, &PredictModel::Sparse).unwrap())
      ));

      if *tracker == MINIST_TRAIN_LABEL_SIZE {
        break;
      }
    }

    data_batch
  }
}


use std::f32::consts::PI as PI32;

impl Dataset<Cf32, Cf32> {
  /// Contructs and returns a bacth of data for a signal reconstruction application.
  pub fn signal_reconstruction(
    samples: usize,
    batch_size: usize,
    noise_threshold: f32,
    train_seed: &mut u128,
    test_seed: &mut u128
  ) -> (Dataset<Cf32, Cf32>, Dataset<Cf32, Cf32>) {
    let mut train_batch = Dataset::new();
    let mut test_batch = Dataset::new();

    // frequencies will be between 0.1 and 1
    // perform at least 1 revolution
    let t_max = 4.0 * PI32;
    let sampling_period = t_max / ( samples as f32 );
    let n_waves = 8;

    let t = (0..samples).map(|elm| {
      sampling_period * (elm as f32)
    }).collect::<Vec<_>>();

    for _b in 0..batch_size {
      // training samples logic
      let train_waves_len = lcgi(train_seed, n_waves) + 1;
      let train_waves = (0..train_waves_len).map(|_id| {
        // generate amplitude
        // one order of magnitude of amplitudes
        let a = lcgf32(train_seed) * 0.9 + 0.1;
        // generate frequency
        // one order of magnitude of frequencies
        let w = lcgf32(train_seed) * 0.9 + 0.1;
        // generate phase
        let phi = 2.0 * PI32 * lcgf32(train_seed);

        (a, w, phi)
      }).collect::<Vec<_>>();

      let train_func = |time: &f32| { 
        let sample = train_waves.iter().fold(Cf32::new(0.0, 0.0), 
        |acc, (a, w, phi)| {
          let phase = *w * *time + *phi;
          acc + Cf32::newe(*a, phase)
        });
        
        sample
      };

      // testing samples logic
      let test_waves_len = lcgi(test_seed, n_waves) + 1;
      let test_waves = (0..test_waves_len).map(|_id| {
        // generate amplitude (around 1)
        let a = lcgf32(test_seed) * 0.9 + 0.1;
        // generate frequency
        let w = lcgf32(test_seed) * 0.9 + 0.1;
        // generate phase
        let phi = 2.0 * PI32 * lcgf32(test_seed);

        (a, w, phi)
      }).collect::<Vec<_>>();

      let test_func = |time: &f32| { 
        let sample = test_waves.iter().fold(Cf32::new(0.0, 0.0), 
        |acc, (a, w, phi)| {
            let phase = *w * *time + *phi;
            acc + Cf32::newe(*a, phase)
        });

        sample
      };

      // determine original signal
      let mut clean_signal = t.iter()
        // compute signal
        .map(train_func)
        .collect::<Vec<_>>();
      // normalization
      let train_max = clean_signal.iter().fold(0.0, |acc, elm| { 
        let norm = elm.norm();
        if norm > acc { norm } else { acc }
      });
      if train_max != 0.0 { 
        clean_signal.iter_mut().for_each(|elm| { *elm /= Cf32::new(train_max, 0.0); }); 
      }
      
      // determine original signal (test)
      let mut clean_signal_t = t.iter()
        // compute signal
        .map(test_func)
        .collect::<Vec<_>>();
      // normalization
      let test_max = clean_signal_t.iter().fold(0.0, |acc, elm| { 
        let norm = elm.norm();
        if norm > acc { norm } else { acc }
      });
      if test_max != 0.0 { 
        clean_signal_t.iter_mut().for_each(|elm| { *elm /= Cf32::new(test_max, 0.0); });
      }

      // exagerated power line noise on amplitude
      let int_percentage = noise_threshold / 20.0;
      // corrupt the signal with gausian noise
      let corrupted_signal = clean_signal.chunks(2).flat_map(|elm| {
        let elm1 = elm[0];
        let elm2 = elm[1];
        // check point amplitude
        let a1 = elm1.norm();
        let noise1 = noise_threshold + int_percentage * a1;
        let a2 = elm2.norm();
        let noise2 = noise_threshold + int_percentage * a2;

        // generates random uniform values between 0 and 1
        let u1 = lcgf32(test_seed);
        let u2 = lcgf32(test_seed);
        // Box-Muller transform (normal distribution with noise as std)
        let z1 = ( (-2.0 * u1.ln()).sqrt() * (2.0 * PI32 * u2).cos() ) * noise1;
        let z2 = ( (-2.0 * u1.ln()).sqrt() * (2.0 * PI32 * u2).sin() ) * noise2;

        let p1 = 2.0 * PI32 * lcgf32(train_seed);
        let p2 = 2.0 * PI32 * lcgf32(train_seed);
        
        // corrupt point
        [elm1 + Cf32::newe(z1, p1), elm2 + Cf32::newe(z2, p2)]
      }).collect::<Vec<_>>();
      let corrupted_signal_t = clean_signal_t.chunks(2).flat_map(|elm| {
        let elm1 = elm[0];
        let elm2 = elm[1];
        // check point amplitude
        let a1 = elm1.norm();
        let noise1 = noise_threshold + int_percentage * a1;
        let a2 = elm2.norm();
        let noise2 = noise_threshold + int_percentage * a2;

        // generates random uniform values between 0 and 1
        let u1 = lcgf32(test_seed);
        let u2 = lcgf32(test_seed);
        // Box-Muller transform (normal distribution with noise as std)
        let z1 = ( (-2.0 * u1.ln()).sqrt() * (2.0 * PI32 * u2).cos() ) * noise1;
        let z2 = ( (-2.0 * u1.ln()).sqrt() * (2.0 * PI32 * u2).sin() ) * noise2;

        let p1 = 2.0 * PI32 * lcgf32(test_seed);
        let p2 = 2.0 * PI32 * lcgf32(test_seed);

        // corrupt point
        [elm1 + Cf32::newe(z1, p1), elm2 + Cf32::newe(z2, p2)]
      }).collect::<Vec<_>>();

      // add to the dataset
      train_batch.add_point((IOType::Scalar(corrupted_signal), IOType::Scalar(clean_signal)));
      test_batch.add_point((IOType::Scalar(corrupted_signal_t), IOType::Scalar(clean_signal_t)));
    }
    
    (train_batch, test_batch)
  }
}
