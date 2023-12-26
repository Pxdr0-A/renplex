pub mod dense;


use self::dense::{DenseInputLayer, DenseLayer};
use crate::lite::real::Param;


pub enum InputLayer<P: Param> {
    DenseInputLayer(DenseInputLayer<P>)
}

impl<P: Param + Copy> InputLayer<P> {
    pub fn signal(&self, input: &[P]) -> Vec<P> {
        match self {
            InputLayer::DenseInputLayer(layer) => {
                layer
                    .signal(input)
                    .unwrap()
            }
        }
    }
}

pub enum Layer<P: Param> {
    DenseLayer(DenseLayer<P>)
}

impl<P: Param + Copy> Layer<P> {
    pub fn signal(&self, input: &[P]) -> Vec<P> {
        match self {
            Layer::DenseLayer(layer) => {
                layer
                    .signal(input)
            }
        }
    }
}
