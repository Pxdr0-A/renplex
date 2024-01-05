pub mod dense;

use dense::{DenseCInputLayer, DenseCLayer};
use crate::lite::complex::ComplexParam;

#[derive(Debug)]
pub enum ComplexInputLayer<CP: ComplexParam> {
    DenseCInputLayer(DenseCInputLayer<CP>)
}

impl<CP: ComplexParam + Copy> ComplexInputLayer<CP> {
    pub fn signal(&self, input: &[CP]) -> Vec<CP> {
        match self {
            ComplexInputLayer::DenseCInputLayer(layer) => {
                layer
                    .signal(input)
            }
        }
    }
}

#[derive(Debug)]
pub enum ComplexLayer<P: ComplexParam> {
    DenseCLayer(DenseCLayer<P>)
}

impl<CP: ComplexParam + Copy> ComplexLayer<CP> {
    pub fn signal(&self, input: &[CP]) -> Vec<CP> {
        match self {
            ComplexLayer::DenseCLayer(layer) => {
                layer
                    .signal(input)
            }
        }
    }
}