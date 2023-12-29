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
}