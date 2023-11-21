pub mod activation;


//std
use std::{ops::{AddAssign, Mul, Neg}, fmt::Debug};

use crate::math::ops::base::Number;

// local
use self::activation::{ActivationFunction, Activatable};

#[derive(Debug)]
pub struct Neuron<W> {
    pub weights: Vec<W>,
    pub bias: W,
    pub activation: ActivationFunction,
}

impl<W> Neuron<W> 
    where 
        W: AddAssign + Neg<Output = W> + Mul<Output = W> + PartialEq, 
        W: Activatable + Number, 
        W: Copy + Debug {
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
    pub fn signal(&self, input: &[W]) -> W {
        
        assert_eq!(
            self.weights.len(), input.len(),
            "Input length must match the number of neuron inputs."
        );

        // init cycle (bias needs to come with the proper sign)
        // or just correct it and make W accept the Neg trait
        let mut out = -self.bias;
        for i in 0..input.len() {
            out += self.weights[i] * input[i];
        }

        let out_result = out.activation(&self.activation);
        
        //print!("({:?}, {:?})", out, out_result);

        out_result
    }
}
