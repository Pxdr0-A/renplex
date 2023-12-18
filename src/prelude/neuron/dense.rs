
use super::ActivationFunction;
use super::UnitLike;
use super::param::Param;


#[derive(Debug)]
pub struct DenseNeuron<P: Param> {
    weights: Vec<P>,
    bias: P,
    activation: ActivationFunction
}


impl<P> UnitLike<P> for DenseNeuron<P>
    where
        P: Param + Copy {

    /// Returns a `Neuron<W>` with the specified weights, bias and activation function.
    /// 
    /// # Arguments
    /// 
    /// * `weights` - Vector with the weights.
    /// * `bias` - Bias of the neuron (associated with unit input).
    /// * `activation` - Activation function to be associated with the neuron.
    ///    Check `renplex::prelude::neuron::activation::ActivationFunction` enum for the available options.
    fn new(weights: Vec<P>, bias: P, activation: ActivationFunction) -> DenseNeuron<P> {
        
        DenseNeuron { 
            weights, 
            bias, 
            activation 
        }

    }

    fn get_weights(&self) -> &[P] {
        
        &self.weights

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

        out.act(&self.activation)

    }
    
}
