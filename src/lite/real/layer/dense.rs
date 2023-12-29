use crate::lite::real::{Param, ActivationFunction};
use crate::lite::real::unit::dense::DenseNeuron;
use crate::math::random::lcgf32;
use super::{InputLayer, Layer};

#[derive(Debug)]
pub struct DenseInputLayer<P: Param> {
    input_size: usize,
    units: Vec<DenseNeuron<P>>
}

impl<P: Param + Copy> DenseInputLayer<P> {
    pub fn new(capacity: usize, input_size: usize) -> DenseInputLayer<P> {
        DenseInputLayer {
            input_size,
            units: Vec::with_capacity(capacity)
        }
    }

    pub fn add(&mut self, neuron: DenseNeuron<P>) {
        self.units.push(neuron);
    }

    pub fn set_input_size(&mut self, size: usize) -> Result<(), LayerInputError> {
        match self.input_size == size {
            true => {  },
            false => { return Err(LayerInputError::ClashingInput(self.input_size, size)); }
        }

        let entries: usize = self.units
            .iter()
            .map(|elm| {elm.get_weights().len()})
            .sum();

        match size % entries == 0 && size / entries >= 2 {
            true => { self.input_size = size },
            false => { return Err(LayerInputError::InconsistentInput(size, entries)); }
        }

        Ok(())
    }

    /// Returns a new Vec resultant from fowarding a signal through the input layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    pub fn signal(&self, input: &[P]) -> Vec<P> {
        match self.input_size == input.len() {
            true => {},
            false => { panic!("Input size not matching input length.") }
        }

        let mut output = Vec::with_capacity(self.units.len());

        let mut demand: usize;
        let mut position: usize = 0;
        for neuron in &self.units {
            // The -1 is to work around index counts
            demand = position + ( neuron.get_weights().len() - 1 );

            output.push(
                neuron
                    .signal(&input[position..=demand])
            );

            // The +1 is to move one place. 
            // We do not want to repeat the last point
            position = demand + 1;

        }

        output
    }

    pub fn wrap(self) -> InputLayer<P> {
        InputLayer::DenseInputLayer(self)
    }
}

impl DenseInputLayer<f32> {
    pub fn init(
        capacity: usize,
        input_size: usize,
        acti: ActivationFunction,
        scale: f32,
        seed: &mut u128) -> DenseInputLayer<f32> {
        
        let inputs = match input_size % capacity == 0 && input_size / capacity >= 2 {
            true => { input_size / capacity },
            false => { panic!("Input size needs to be a multiple of the number of units.") }
        };

        let mut layer: DenseInputLayer<f32> = DenseInputLayer::new(capacity, input_size);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {elm * lcgf32(seed) - (scale / 2.0)})
                        .collect(), 
                    scale * lcgf32(seed) - (scale / 2.0), 
                    acti.clone()
                )
            );
        }

        layer
    }
}

#[derive(Debug)]
pub struct DenseLayer<P: Param> {
    units: Vec<DenseNeuron<P>>
}

impl<P: Param + Copy> DenseLayer<P> {
    pub fn new(capacity: usize) -> DenseLayer<P> {
        DenseLayer {
            units: Vec::with_capacity(capacity)
        }
    }

    pub fn add(&mut self, neuron: DenseNeuron<P>) {
        self.units.push(neuron);
    }

    /// Returns a new Vec resultant from fowarding an input through a hidden layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    pub fn signal(&self, input: &[P]) -> Vec<P> {
        let mut output = Vec::with_capacity(self.units.len());
        for neuron in &self.units {
            output.push(
                neuron
                    .signal(input)
            );
        }

        output
    }

    pub fn wrap(self) -> Layer<P> {
        Layer::DenseLayer(self)
    }
}

impl DenseLayer<f32> {
    pub fn init(
        capacity: usize,
        inputs: usize,
        acti: ActivationFunction,
        scale: f32,
        seed: &mut u128) -> DenseLayer<f32> {
        
        let mut layer: DenseLayer<f32> = DenseLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {elm * lcgf32(seed) - (scale / 2.0)})
                        .collect(), 
                    scale * lcgf32(seed) - (scale / 2.0), 
                    acti.clone()
                )
            );
        }

        layer
    }

}


#[derive(Debug)]
pub enum UnsetInputError {
    ClashingInput(usize, usize)
}

#[derive(Debug)]
pub enum LayerInputError {
    ClashingInput(usize, usize),
    InconsistentInput(usize, usize)
}