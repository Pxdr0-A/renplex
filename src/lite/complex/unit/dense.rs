use crate::lite::complex::{ComplexParam, ComplexActFunction};

#[derive(Debug)]
pub struct DenseNeuron<CP: ComplexParam> {
    weights: Vec<CP>,
    bias: CP,
    acti: ComplexActFunction
}


impl<CP: ComplexParam + Copy> DenseNeuron<CP> {
    /// Returns a `Neuron<W>` with the specified weights, bias and activation function.
    /// 
    /// # Arguments
    /// 
    /// * `weights` - Vector with the weights.
    /// * `bias` - Bias of the neuron (associated with unit input).
    /// * `activation` - Activation function to be associated with the neuron.
    ///    Check `renplex::prelude::neuron::activation::ActivationFunction` enum for the available options.
    pub fn new(weights: Vec<CP>, bias: CP, acti: ComplexActFunction) -> DenseNeuron<CP> {
        DenseNeuron { 
            weights, 
            bias: acti.derive_bias(bias), 
            acti 
        }
    }

    pub fn get_weights(&self) -> &[CP] {
        &self.weights
    }

    /// Returns the result from the neuron's activation against an input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the neuron. 
    ///             Needs to be in agreement with the number of weights.
    pub fn signal(&self, input: &[CP]) -> CP {
        if self.weights.len() != input.len() { panic!("Input size does not match the number of weights.") }

        let mut out = self.bias;
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