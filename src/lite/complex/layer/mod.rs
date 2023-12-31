pub mod dense;

use dense::{DenseInputLayer, DenseLayer};
use crate::lite::complex::ComplexParam;

#[derive(Debug)]
pub enum InputLayer<CP: ComplexParam> {
    DenseInputLayer(DenseInputLayer<CP>)
}

impl<CP: ComplexParam + Copy> InputLayer<CP> {
    pub fn signal(&self, input: &[CP]) -> Vec<CP> {
        match self {
            InputLayer::DenseInputLayer(layer) => {
                layer
                    .signal(input)
            }
        }
    }
}

#[derive(Debug)]
pub enum Layer<P: ComplexParam> {
    DenseLayer(DenseLayer<P>)
}

impl<CP: ComplexParam + Copy> Layer<CP> {
    pub fn signal(&self, input: &[CP]) -> Vec<CP> {
        match self {
            Layer::DenseLayer(layer) => {
                layer
                    .signal(input)
            }
        }
    }
}