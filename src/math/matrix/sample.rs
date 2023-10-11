use super::Matrix;
use crate::math::random::{lcgf32, lcgf64};

impl Matrix<f32> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        seed: &mut u128

    ) -> (Matrix<f32>, Vec<f32>) {

        assert!(
            dims[0] > degree,
            "Number of instances needs to be larger than the number of classes."
        );

        let mut sample_matrix: Matrix<f32> = Matrix::new(dims);

        let macro_scale: f32 = 100.0;
        let micro_scale: f32 = macro_scale / 50.0;

        let mut rand_val: f32 = lcgf32(seed);

        // spray focal points
        let mut centers: Matrix<f32> = Matrix::new([degree, dims[1]]);
        let mut center: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..degree {
            for _ in 0..dims[1] {
                center.push(
                    // random point relative to origin
                    rand_val * macro_scale - (macro_scale / 2.0)
                );

                rand_val = lcgf32(seed);
            }

            // add_row will clean the center vector
            centers.add_row(&mut center);
        }

        let mut class_center: &[f32];
        let mut selected_class: f32;
        let mut labels: Vec<f32> = Vec::with_capacity(dims[0]);
        let mut added_row: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (rand_val * (degree as f32) - 1.0).round();
            labels.push(selected_class);
            
            class_center = centers.row(
                &(
                    selected_class as usize
                )
            );

            for col in 0..dims[1] {
                rand_val = lcgf32(seed);

                added_row.push(
                    class_center[col] + rand_val * micro_scale - (micro_scale / 2.0)
                );
            }
            
            // add_row will clean the added_row vec
            sample_matrix.add_row(&mut added_row);

            rand_val = lcgf32(seed);
        }

        (sample_matrix, labels)
    }
}

// pass this to a macro for f64
impl Matrix<f64> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        seed: &mut u128

    ) -> (Matrix<f64>, Vec<f64>) {

        assert!(
            dims[0] > degree,
            "Number of instances needs to be larger than the number of classes."
        );

        let mut sample_matrix: Matrix<f64> = Matrix::new(dims);

        let macro_scale: f64 = 100.0;
        let micro_scale: f64 = macro_scale / 50.0;

        let mut rand_val: f64 = lcgf64(seed);

        // spray focal points
        let mut centers: Matrix<f64> = Matrix::new([degree, dims[1]]);
        let mut center: Vec<f64> = Vec::with_capacity(dims[1]);
        for _ in 0..degree {
            for _ in 0..dims[1] {
                center.push(
                    // random point relative to origin
                    rand_val * macro_scale - (macro_scale / 2.0)
                );

                rand_val = lcgf64(seed);
            }

            // add_row will clean the center vector
            centers.add_row(&mut center);
        }

        let mut class_center: &[f64];
        let mut selected_class: f64;
        let mut labels: Vec<f64> = Vec::with_capacity(dims[0]);
        let mut added_row: Vec<f64> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (rand_val * (degree as f64) - 1.0).round();
            labels.push(selected_class);
            
            class_center = centers.row(
                &(
                    selected_class as usize
                )
            );

            for col in 0..dims[1] {
                rand_val = lcgf64(seed);

                added_row.push(
                    class_center[col] + rand_val * micro_scale - (micro_scale / 2.0)
                );
            }
            
            // add_row will clean the added_row vec
            sample_matrix.add_row(&mut added_row);

            rand_val = lcgf64(seed);
        }

        (sample_matrix, labels)
    }
}