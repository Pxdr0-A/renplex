pub mod activation;


//std
use std::ops::{AddAssign, Mul};

// local
use self::activation::{ActivationFunction, Activation};


pub struct Neuron<W> {
    pub id: usize,
    pub weights: Vec<W>,
    pub bias: W,
    pub activation: ActivationFunction,
}

impl<W> Neuron<W> {
    pub fn new(
        id: usize, 
        weights: Vec<W>,
        bias: W, 
        activation: ActivationFunction
    ) -> Neuron<W> {
        
        Neuron { 
            id, 
            weights, 
            bias, 
            activation 
        }
    }

    pub fn signal(&self, input: &[W]) -> W 
        where 
            W: AddAssign + Mul<Output = W> + Activation, 
            W: Copy {
        
        assert_eq!(self.weights.len(), input.len(),
                   "Input length must match the number of neuron inputs."
        );

        // init cycle (bias needs to come with the proper sign)
        let mut out = self.bias;
        for i in 0..input.len() {
            out += self.weights[i] * input[i];
        }

        out.activation(&self.activation)
    }
}
