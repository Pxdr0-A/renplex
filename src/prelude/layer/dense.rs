use crate::prelude::neuron::UnitLike;
use crate::prelude::neuron::dense::DenseNeuron;
use crate::prelude::neuron::param::Param;

use super::LayerLike;


/// Conventional fully connected layer for input purposes.
#[derive(Debug)]
pub struct DenseInputLayer<P: Param> {
    units: Vec<DenseNeuron<P>>
}

/// Conventional fully connected layer.
#[derive(Debug)]
pub struct DenseLayer<P: Param> {
    units: Vec<DenseNeuron<P>>
}


impl<P> LayerLike<P, DenseNeuron<P>> for DenseInputLayer<P> 
    where 
        P: Param + Copy {

    /// Returns an empty `InputLayer<W>`. Enough memory is allocated in the process.
    /// 
    /// # Arguments
    /// 
    /// * `n_units` - Number of units that the layer will contain. 
    ///               A Vec will be allocated with enough memory for the n_units.
    fn new(n_units: usize) -> DenseInputLayer<P> {

        DenseInputLayer {
            units: Vec::<DenseNeuron<P>>::with_capacity(n_units)
        }

    }

    /// Updates the `InputLayer<W>` object with a `Neuron<W>`
    /// 
    /// # Arguments
    /// 
    /// * `neuron` - Neuron to add to the respective layer.
    fn add(&mut self, neuron: DenseNeuron<P>) {

        self.units.push(neuron);

    }

    /// Returns a new Vec resultant from fowarding a signal through the input layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    fn signal(&self, input: &[P]) -> Vec<P> {

        // Try to implement concurrency if possible
        let mut output = Vec::with_capacity(self.units.len());

        let mut demand: usize;
        let mut position: usize = 0;
        for neuron in &self.units {
            // The -1 is to work around index counts
            demand = position + ( neuron.get_weights().len() - 1 );

            assert!(
                demand < input.len(), 
                "Input with inconsistent shape given to InputLayer."
            );

            output.push(
                neuron.signal(&input[position..=demand])
            );

            // The +1 is to move one place. 
            // We do not want to repeat the last point
            position = demand + 1;
        }

        output
    }

}

impl<P> LayerLike<P, DenseNeuron<P>> for DenseLayer<P> 
    where 
        P: Param + Copy {

    /// Returns an empty `HiddenLayer<W>`. Enough memory is allocated in the process.
    /// 
    /// # Arguments
    /// 
    /// * `n_units` - Number of units that the layer will contain. 
    ///               A Vec will be allocated with enough memory for the n_units.
    fn new(n_units: usize) -> DenseLayer<P> {

        DenseLayer {
            units: Vec::<DenseNeuron<P>>::with_capacity(n_units)
        }

    }

    /// Updates the `HiddenLayer<W>` object with a `Neuron<W>`
    /// 
    /// # Arguments
    /// 
    /// * `neuron` - Neuron to add to the respective layer.
    fn add(&mut self, neuron: DenseNeuron<P>) {

        self.units.push(neuron);

    }

    /// Returns a new Vec resultant from fowarding an input through a hidden layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    fn signal(&self, input: &[P]) -> Vec<P> {

        // try to implement concurrency if possible
        let mut output = Vec::with_capacity(self.units.len());
        for n in &self.units {
            output.push(n.signal(input));
        }

        output
    }

}
