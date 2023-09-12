use std::ops::{AddAssign, Mul};

use super::neuron::Neuron;
use super::neuron::activation::Activation;


pub struct Layer<W> {
    pub id: usize,
    pub units: Vec<Neuron<W>>
}

impl<W> Layer<W> {
    pub fn new(id: usize, units: Vec<Neuron<W>>) -> Layer<W> {
        Layer {
            id,
            units
        }
    }

    pub fn signal(&self, input: &[W]) -> Vec<W> 
        where 
            W: AddAssign + Mul<Output = W> + Activation, 
            W: Copy {
        // For hidden layer only

        // try to implement concurrency if possible
        // learn more about it
        let mut output = Vec::with_capacity(self.units.len());
        for n in &self.units {
            output.push(n.signal(input));
        }

        output
    }
}