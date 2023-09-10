pub mod activation;

use activation::ActivationFunction;


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
}