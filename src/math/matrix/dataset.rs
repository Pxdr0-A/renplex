use super::Matrix;
use crate::math::{random::{lcgf32, lcgf64}, complex::{Cfloat, casts::ComplexCast}};


// CREATE A DATASET STRUCT!
#[derive(Debug)]
pub struct Dataset<B, T> {
    pub body: Matrix<B>,
    pub target: Vec<T>
}


impl Dataset<f32, f32> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        seed: &mut u128

    ) -> Dataset<f32, f32> {

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
        let mut selected_class: usize;
        let mut labels: Vec<f32> = Vec::with_capacity(dims[0]);
        let mut added_row: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (rand_val * (degree as f32)) as usize;
            labels.push(selected_class as f32);
            
            class_center = centers.row(&selected_class);

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

        Dataset {
            body: sample_matrix,
            target: labels
        }
    }

    pub fn to_complex(&mut self) -> Dataset<Cfloat<f32>, f32> {
        let mut complex_matrix: Matrix<Cfloat<f32>> = Matrix::new(self.body.shape);
        
        let mut dataset_row;
        let mut added_row: Vec<Cfloat<f32>>;
        for _ in 0..self.body.shape[0] {
            // deletes, dealocates and returns the row.
            dataset_row = self.body.del_row(&0);

            added_row = dataset_row    
                .into_iter()
                .map(|x| x.to_complex())
                .collect();

            complex_matrix.add_row(&mut added_row);
        }

        let mut labels: Vec<f32> = Vec::with_capacity(self.body.shape[0]);
        labels.append(&mut self.target);

        self.target.shrink_to_fit();

        Dataset { 
            body: complex_matrix, 
            target: labels
        }
    }
}

// pass this to a macro for f64
impl Dataset<f64, f64> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        seed: &mut u128

    ) -> Dataset<f64, f64> {

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
        let mut selected_class: usize;
        let mut labels: Vec<f64> = Vec::with_capacity(dims[0]);
        let mut added_row: Vec<f64> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (rand_val * (degree as f64)) as usize;
            labels.push(selected_class as f64);
            
            class_center = centers.row(&selected_class);

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

        Dataset { 
            body: sample_matrix, 
            target: labels 
        }
    }
}