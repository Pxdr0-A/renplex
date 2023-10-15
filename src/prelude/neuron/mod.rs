pub mod activation;


//std
use std::ops::{AddAssign, Mul};

// local
use self::activation::{ActivationFunction, Activatable};


pub struct Neuron<W> {
    pub weights: Vec<W>,
    pub bias: W,
    pub activation: ActivationFunction,
}

impl<W> Neuron<W> {
    /// Returns a `Neuron<W>` with the specified weights, bias and activation function.
    /// 
    /// # Arguments
    /// 
    /// * `weights` - Vector with the weights.
    /// * `bias` - Bias of the neuron (associated with unit input).
    /// * `activation` - Activation function to be associated with the neuron.
    ///    Check `renplex::prelude::neuron::activation::ActivationFunction` for the available options.
    pub fn new(
        weights: Vec<W>,
        bias: W, 
        activation: ActivationFunction
    ) -> Neuron<W> {
        
        Neuron { 
            weights, 
            bias, 
            activation 
        }
    }

    /// Returns the result from the neuron's activation against an input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the neuron. 
    ///             Needs to be in agreement with the number of weights.
    pub fn signal(&self, input: &[W]) -> W 
        where 
            W: AddAssign + Mul<Output = W> + Activatable, 
            W: Copy {
        
        assert_eq!(
            self.weights.len(), input.len(),
            "Input length must match the number of neuron inputs."
        );

        // init cycle (bias needs to come with the proper sign)
        // or just correct it and make W accept the Neg trait
        let mut out = self.bias;
        for i in 0..input.len() {
            out += self.weights[i] * input[i];
        }

        out.activation(&self.activation)
    }
}
