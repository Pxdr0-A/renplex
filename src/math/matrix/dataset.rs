use super::Matrix;

//use crate::math::random::{lcgf32, lcgf64};

#[derive(Debug)]
pub struct Dataset<B, T> {
    pub body: Matrix<B>,
    pub target: Matrix<T>,
    pub degree: usize
}


/*
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
        
        // spray focal points
        let mut centers: Matrix<f32> = Matrix::new([degree, dims[1]]);
        let mut center: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..degree {
            for _ in 0..dims[1] {
                center.push(
                    // random point relative to origin
                    lcgf32(seed) * macro_scale - (macro_scale / 2.0)
                );
            }

            // add_row will clean the center vector
            centers.add_mut_row(&mut center);
        }
        drop(center);

        let mut class_center: &[f32];
        let mut selected_class: usize;
        let mut one_hot_vec: Vec<f32>;
        let mut labels: Matrix<f32> = Matrix::new(dims);
        let mut added_row: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (lcgf32(seed) * (degree as f32)) as usize;

            one_hot_vec = vec![0.0; degree];
            one_hot_vec[selected_class] += 1.0; 
            labels.add_mut_row(&mut one_hot_vec);
            
            class_center = centers.row(selected_class);

            for col in 0..dims[1] {
                added_row.push(
                    class_center[col] + lcgf32(seed) * micro_scale - (micro_scale / 2.0)
                );
            }
            
            // add_row will clean the added_row vec
            sample_matrix.add_mut_row(&mut added_row);
        }

        Dataset {
            body: sample_matrix,
            target: labels,
            degree
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

        // spray focal points
        let mut centers: Matrix<f64> = Matrix::new([degree, dims[1]]);
        let mut center: Vec<f64> = Vec::with_capacity(dims[1]);
        for _ in 0..degree {
            for _ in 0..dims[1] {
                center.push(
                    // random point relative to origin
                    lcgf64(seed) * macro_scale - (macro_scale / 2.0)
                );
            }

            // add_row will clean the center vector
            centers.add_mut_row(&mut center);
        }
        drop(center);

        let mut class_center: &[f64];
        let mut selected_class: usize;
        let mut one_hot_vec: Vec<f64>;
        let mut labels: Matrix<f64> = Matrix::new(dims);
        let mut added_row: Vec<f64> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (lcgf64(seed) * (degree as f64)) as usize;

            one_hot_vec = vec![0.0; degree];
            one_hot_vec[selected_class] += 1.0; 
            labels.add_mut_row(&mut one_hot_vec);
            
            class_center = centers.row(selected_class);

            for col in 0..dims[1] {
                added_row.push(
                    class_center[col] + lcgf64(seed) * micro_scale - (micro_scale / 2.0)
                );
            }
            
            // add_row will clean the added_row vec
            sample_matrix.add_mut_row(&mut added_row);
        }

        Dataset { 
            body: sample_matrix, 
            target: labels,
            degree
        }
    }
}
*/