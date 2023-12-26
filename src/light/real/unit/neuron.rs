use crate::light::real::ActivationFunction;

use super::Param;

pub struct Neuron<P: Param> {
    weights: Vec<P>,
    bias: P,
    acti: ActivationFunction
}


impl<P: Param + Copy> Neuron<P> {
    /// Returns a `Neuron<W>` with the specified weights, bias and activation function.
    /// 
    /// # Arguments
    /// 
    /// * `weights` - Vector with the weights.
    /// * `bias` - Bias of the neuron (associated with unit input).
    /// * `activation` - Activation function to be associated with the neuron.
    ///    Check `renplex::prelude::neuron::activation::ActivationFunction` enum for the available options.
    fn new(weights: Vec<P>, bias: P, acti: ActivationFunction) -> Neuron<P> {
        
        Neuron { 
            weights, 
            bias, 
            acti 
        }
    }

    /// Returns the result from the neuron's activation against an input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the neuron. 
    ///             Needs to be in agreement with the number of weights.
    fn signal(&self, input: &[P]) -> P {
        
        assert_eq!(
            self.weights.len(), input.len(),
            "Input length must match the number of neuron inputs."
        );

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
