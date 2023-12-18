use std::ops::{AddAssign, Add, Mul, Div, Sub, Rem};

use super::Matrix;

use crate::math::ops::base::Number;
use crate::math::random::Lcg;


// CREATE A DATASET STRUCT!
#[derive(Debug)]
pub struct Dataset<B, T> {
    pub body: Matrix<B>,
    pub target: Matrix<T>,
    pub degree: usize
}

impl<B, T> Dataset<B, T> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        scales: [B; 2],
        seed: &mut B,
        loc_seed: &mut usize
    ) -> Dataset<B, T>
        where 
            B: Number + Copy + AddAssign + Add<Output = B> + Sub<Output = B> + Mul<Output = B> + Div<Output = B> + Rem<Output = B>,
            T: Number + Clone + AddAssign + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> {
        
        assert!(
            dims[0] > degree,
            "Number of instances needs to be larger than the number of classes."
        );

        println!("Setting up generator 1...");
        let rnd_gen1 = Lcg::<B>::setup();
        println!("Setting up generator 2...");
        // discrete randomizer
        let mut rnd_loc1 = Lcg::<usize>::setup();
        rnd_loc1.set_range(rnd_loc1.range / degree);
        println!("Done.");

        let bin = B::unit() + B::unit();

        let mut body = Matrix::new(dims);
        let mut target = Matrix::new(dims);

        let mut centers= Matrix::new([degree, dims[1]]);
        let mut center = Vec::with_capacity(dims[1]);

        for _ in 0..degree {
            for _ in 0..dims[1] {
                center.push(
                    // random point relative to origin
                    rnd_gen1.gen(seed) * scales[0] - (scales[0] / bin)
                );
            }

            // add_row will clean the center vector
            centers.add_row(&mut center);
        }

        let mut class_center;
        let mut rnd_class;
        let mut added_row = Vec::with_capacity(dims[1]);
        // for this sample it is going to be a sparse configuration.
        let mut target_vec;
        for _ in 0..dims[0] {
            rnd_class = rnd_loc1.gen(loc_seed);
            println!("{}", rnd_class);
            class_center = centers.row(rnd_class);

            target_vec = vec![T::null(); degree];
            target_vec[rnd_class] += T::unit();
            target.add_row(&mut target_vec);

            for col in 0..dims[1] {
                added_row.push(
                    class_center[col] + rnd_gen1.gen(seed) * scales[1] - (scales[1] / bin)
                );
            }
            
            // add_row will clean the added_row vec
            body.add_row(&mut added_row);

        }

        Dataset { body, target, degree }

    }

}

/*

impl Dataset<f32, u8> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        seed: &mut u128

    ) -> Dataset<f32, u8> {

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
        let mut labels: Vec<u8> = Vec::with_capacity(dims[0]);
        let mut added_row: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (rand_val * (degree as f32)) as usize;
            labels.push(selected_class as u8);
            
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
}

// pass this to a macro for f64
impl Dataset<f64, u8> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        seed: &mut u128

    ) -> Dataset<f64, u8> {

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
        let mut labels: Vec<u8> = Vec::with_capacity(dims[0]);
        let mut added_row: Vec<f64> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (rand_val * (degree as f64)) as usize;
            labels.push(selected_class as u8);
            
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


impl<B> Dataset<B, u8> {
    pub fn to_complex(&mut self) -> Dataset<Cfloat<B>, u8> 
        where 
            B: ComplexCast<B> {

        let mut complex_matrix: Matrix<Cfloat<B>> = Matrix::new(self.body.shape);
        
        let mut dataset_row;
        let mut added_row: Vec<Cfloat<B>>;
        for _ in 0..self.body.shape[0] {
            // deletes, dealocates and returns the row.
            dataset_row = self.body.del_row(&0);

            added_row = dataset_row    
                .into_iter()
                .map(|x| x.to_complex())
                .collect();

            complex_matrix.add_row(&mut added_row);
        }

        let mut labels: Vec<u8> = Vec::with_capacity(self.body.shape[0]);
        labels.append(&mut self.target);

        self.target.shrink_to_fit();

        Dataset { 
            body: complex_matrix, 
            target: labels
        }
    }
}

 */
