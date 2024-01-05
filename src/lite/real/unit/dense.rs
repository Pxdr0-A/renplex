use crate::lite::real::ActFunction;

use super::Param;

#[derive(Debug)]
pub struct DenseNeuron<P: Param> {
    weights: Vec<P>,
    bias: P,
    acti: ActFunction
}


impl<P: Param + Copy> DenseNeuron<P> {
    /// Returns a `Neuron<W>` with the specified weights, bias and activation function.
    /// 
    /// # Arguments
    /// 
    /// * `weights` - Vector with the weights.
    /// * `bias` - Bias of the neuron (associated with unit input).
    /// * `activation` - Activation function to be associated with the neuron.
    ///    Check `renplex::prelude::neuron::activation::ActivationFunction` enum for the available options.
    pub fn new(weights: Vec<P>, bias: P, acti: ActFunction) -> DenseNeuron<P> {
        DenseNeuron { 
            weights, 
            bias, 
            acti 
        }
    }

    pub fn get_weights(&self) -> &[P] {
        &self.weights
    }

    pub fn get_input_len(&self) -> usize {
        self.weights.len()
    }

    /// Returns the result from the neuron's activation against an input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the neuron. 
    ///             Needs to be in agreement with the number of weights.
    pub fn signal(&self, input: &[P]) -> P {
        if self.weights.len() != input.len() { panic!("Input size does not match the number of weights.") }

        let mut out = self.bias.neg();
        // do with iterators maybe?
        for i in 0..input.len() {
            out.add_mut(
                self.weights[i].mul(input[i])
            );
        }

        out.act(&self.acti)
    }
    
}

#[derive(Debug)]
pub struct NeuronSignalError(usize, usize);
