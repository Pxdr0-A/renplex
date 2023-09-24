use std::ops::{AddAssign, Mul};

use super::neuron::Neuron;
use super::neuron::activation::Activation;

pub struct InputLayer<W> {
    pub units: Vec<Neuron<W>>
}

pub struct HiddenLayer<W> {
    pub units: Vec<Neuron<W>>
}

pub trait Layer<W> {
    fn new(n_units: usize) -> Self;

    fn add(&mut self, neuron: Neuron<W>);

    fn signal(&self, input: &[W]) -> Vec<W>
        where 
            W: AddAssign + Mul<Output = W> + Activation, 
            W: Copy;
}

impl<W> Layer<W> for InputLayer<W> {
    /// Returns an empty `InputLayer<W>`. Enough memory is allocated in the process.
    /// 
    /// # Arguments
    /// 
    /// * `n_units` - Number of units that the layer will contain. 
    ///               A Vec will be allocated with enough memory for the n_units.
    fn new(n_units: usize) -> InputLayer<W> {
        InputLayer {
            units: Vec::<Neuron<W>>::with_capacity(n_units)
        }
    }

    /// Updates the `InputLayer<W>` object with a `Neuron<W>`
    /// 
    /// # Arguments
    /// 
    /// * `neuron` - Neuron to add to the respective layer.
    fn add(&mut self, neuron: Neuron<W>) {
        self.units.push(neuron);
    }

    /// Returns a new Vec resultant from fowarding a signal through the input layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    fn signal(&self, input: &[W]) -> Vec<W> 
        where 
            W: AddAssign + Mul<Output = W> + Activation, 
            W: Copy {
        

        // Try to implement concurrency if possible
        let mut output = Vec::with_capacity(self.units.len());

        let mut demand: usize;
        let mut position: usize = 0;
        for neuron in &self.units {
            // The -1 is to work around index counts
            demand = position + ( neuron.weights.len() - 1 );
            assert!(
                demand <= input.len(), 
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

impl<W> Layer<W> for HiddenLayer<W> {
    /// Returns an empty `HiddenLayer<W>`. Enough memory is allocated in the process.
    /// 
    /// # Arguments
    /// 
    /// * `n_units` - Number of units that the layer will contain. 
    ///               A Vec will be allocated with enough memory for the n_units.
    fn new(n_units: usize) -> HiddenLayer<W> {
        HiddenLayer {
            units: Vec::<Neuron<W>>::with_capacity(n_units)
        }
    }

    /// Updates the `HiddenLayer<W>` object with a `Neuron<W>`
    /// 
    /// # Arguments
    /// 
    /// * `neuron` - Neuron to add to the respective layer.
    fn add(&mut self, neuron: Neuron<W>) {
        self.units.push(neuron);
    }

    /// Returns a new Vec resultant from fowarding an input through a hidden layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    fn signal(&self, input: &[W]) -> Vec<W> 
        where 
            W: AddAssign + Mul<Output = W> + Activation, 
            W: Copy {

        // try to implement concurrency if possible
        let mut output = Vec::with_capacity(self.units.len());
        for n in &self.units {
            output.push(n.signal(input));
        }

        output
    }
}