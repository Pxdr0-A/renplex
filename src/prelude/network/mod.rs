use std::ops::{AddAssign, Mul};

use crate::math::matrix::Matrix;

use super::neuron::Neuron;
use super::layer::{InputLayer, HiddenLayer, Layer};
use super::neuron::activation::Activation;

pub struct Network<W> {
    pub input_layer: InputLayer<W>,
    pub hidden_layers: Vec<HiddenLayer<W>>
}

impl<W> Network<W> {
    /// Returns a `Network<W>` with just input. Reallocation will happen everytime a layer is added.
    /// 
    /// # Arguments
    /// 
    /// * `n_units` - Number of input neurons of the network.
    pub fn new(n_units: usize) -> Network<W> {
        Network {
            input_layer: InputLayer::new(n_units),
            hidden_layers: Vec::<HiddenLayer<W>>::new()
        }
    }

    /// Updates a `Netowork<W>` with an `HiddenLayer<W>`
    /// 
    /// # Arguments
    /// 
    /// * `layer` - Hidden layer of neurons to add to the network.
    pub fn add(&mut self, layer: HiddenLayer<W>) {
        self.hidden_layers.push(layer);
    }

    /// Updates some layer of a `Netowork<W>` with a `Neuron<W>`
    /// 
    /// # Arguments
    /// 
    /// * `order` - Index of the layer to add the neuron. 
    /// * `neuron` - Neuron to add to the respective layer.
    pub fn add_unit(&mut self, order: usize, neuron: Neuron<W>) {
        if order == 0 {
            self.input_layer.add(neuron);
        } else {
            assert!(
                order <= self.hidden_layers.len(), 
                "Attempted to add a unit to an unexistent layer."
            );
            self.hidden_layers[order-1].add(neuron);
        }
    }

    /// Propagates a signal through a `Network<W>` given an input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice with the input data. Must be in agreement with the input length.
    pub fn foward(&self, input: &[W]) -> Vec<W>
        where 
            W: AddAssign + Mul<Output = W> + Activation, 
            W: Copy {
        
        let mut out = self.input_layer.signal(input);
        for layer in &self.hidden_layers {
            out = layer.signal(&out);
        }

        out
    }

    pub fn fit<T>(&self, _data: Matrix<T>) {
        todo!()
    }
}