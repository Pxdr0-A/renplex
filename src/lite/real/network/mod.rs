use crate::math::matrix::Matrix;
use crate::math::matrix::dataset::Dataset;

use super::layer::{InputLayer, Layer};
use super::Param;

#[derive(Debug)]
pub struct FeedFoward<P: Param> {
    input: InputLayer<P>,
    hidden: Vec<Layer<P>>
}

impl<P: Param + Copy> FeedFoward<P> {
    pub fn new(input: InputLayer<P>) -> FeedFoward<P> {
        FeedFoward { 
            input, 
            hidden: Vec::new()
        }
    }

    pub fn add_layer(&mut self, layer: Layer<P>) {
        self.hidden.push(layer);
    }

    pub fn foward(&self, input: &[P]) -> Vec<P> {
        let mut out = self.input.signal(input);

        for layer in &self.hidden {
            out = layer.signal(&out);
        }

        out
    }

    pub fn cost(&self, data: Dataset<P, P>) -> Matrix<P> {
        let mut predictions = Matrix::new(data.body.shape);
        for r in 0..data.body.shape[0] {
            predictions.add_row(self.foward(data.body.row(r)));
        }

        predictions
            .sub(data.target)
            .powi(2)
    }
}