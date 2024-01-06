use crate::math::cfloat::{Cf32, Cf64};
use crate::math::matrix::Matrix;
use crate::math::matrix::dataset::Dataset;

use super::layer::{ComplexInputLayer, ComplexLayer};
use super::ComplexParam;


pub enum ComplexCriteria {
    REAL,
    IMAGINARY,
    NORM,
    PHASE
}

impl ComplexCriteria {
    fn apply_cf32(&self, output: Vec<Cf32>) -> Vec<f32> {
        match self {
            ComplexCriteria::REAL => { output.into_iter().map(|e| {e.re()}).collect() },
            ComplexCriteria::IMAGINARY => { output.into_iter().map(|e| {e.im()}).collect() },
            ComplexCriteria::NORM => { output.into_iter().map(|e| {e.norm()}).collect() },
            ComplexCriteria::PHASE => { output.into_iter().map(|e| {e.phase()}).collect() }
        }
    }

    fn apply_cf64(&self, output: Vec<Cf64>) -> Vec<f64> {
        match self {
            ComplexCriteria::REAL => { output.into_iter().map(|e| {e.re()}).collect() },
            ComplexCriteria::IMAGINARY => { output.into_iter().map(|e| {e.im()}).collect() },
            ComplexCriteria::NORM => { output.into_iter().map(|e| {e.norm()}).collect() },
            ComplexCriteria::PHASE => { output.into_iter().map(|e| {e.phase()}).collect() }
        }
    }
}


#[derive(Debug)]
pub struct FeedFoward<CP: ComplexParam> {
    input: ComplexInputLayer<CP>,
    hidden: Vec<ComplexLayer<CP>>
}

impl<CP: ComplexParam + Copy> FeedFoward<CP> {
    pub fn new(input: ComplexInputLayer<CP>) -> FeedFoward<CP> {
        FeedFoward { 
            input, 
            hidden: Vec::new()
        }
    }

    pub fn add_layer(&mut self, layer: ComplexLayer<CP>) {
        self.hidden.push(layer);
    }

    pub fn foward(&self, input: &[CP]) -> Vec<CP> {
        let mut out = self.input.signal(input);

        for layer in &self.hidden {
            out = layer.signal(&out);
        }

        out
    }
}

// Complex f32 implementations
impl FeedFoward<Cf32> {
    pub fn cost(&self, data: Dataset<Cf32, f32>, criterion: &ComplexCriteria) -> Matrix<f32> {
        let mut predictions = Matrix::new(data.body.shape);
        for r in 0..data.body.shape[0] {
            predictions.add_row(
                criterion.apply_cf32(
                    self.foward(data.body.row(r))
                )
            );
        }
    
        predictions
            .sub(data.target)
            .powi(2)
    }

    pub fn cp_cost(&self, data: Dataset<Cf32, Cf32>) -> Matrix<Cf32> {
        let mut predictions = Matrix::new(data.body.shape);
        for r in 0..data.body.shape[0] {
            predictions.add_row(
                self.foward(data.body.row(r))
            );
        }
    
        predictions
            .sub_cp(data.target)
            .abs_sq()
    } 
}

// Complex f64 implementations
impl FeedFoward<Cf64> {
    pub fn cost(&self, data: Dataset<Cf64, f64>, criterion: &ComplexCriteria) -> Matrix<f64> {
        let mut predictions = Matrix::new(data.body.shape);
        for r in 0..data.body.shape[0] {
            predictions.add_row(
                criterion.apply_cf64(
                    self.foward(data.body.row(r))
                )
            );
        }
    
        predictions
            .sub(data.target)
            .powi(2)
    }

    pub fn cp_cost(&self, data: Dataset<Cf64, Cf64>) -> Matrix<Cf64> {
        let mut predictions = Matrix::new(data.body.shape);
        for r in 0..data.body.shape[0] {
            predictions.add_row(
                self.foward(data.body.row(r))
            );
        }
    
        predictions
            .sub_cp(data.target)
            .abs_sq()
    }
}
