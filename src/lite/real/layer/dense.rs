use crate::lite::real::{Param, ActFunction};
use crate::lite::real::unit::dense::DenseNeuron;
use crate::math::random::{lcgf32, lcgf64};
use super::{InputLayer, Layer};

#[derive(Debug)]
pub struct DenseInputLayer<P: Param> {
    units: Vec<DenseNeuron<P>>
}

impl<P: Param + Copy> DenseInputLayer<P> {
    pub fn new(capacity: usize) -> DenseInputLayer<P> {
        DenseInputLayer {
            units: Vec::with_capacity(capacity)
        }
    }

    pub fn add(&mut self, neuron: DenseNeuron<P>) {
        self.units.push(neuron);
    }

    pub fn get_input_size(&self) -> usize {
        self.units
            .iter()
            .map(|elm| { elm.get_input_len() })
            .reduce(|acc, elm| { acc + elm })
            .unwrap()
    }

    /// Returns a new Vec resultant from fowarding a signal through the input layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    pub fn signal(&self, input: &[P]) -> Vec<P> {
        if self.get_input_size() == input.len() { panic!("Input size not matching input length.") }

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
        acti: ActFunction,
        scale: f32,
        seed: &mut u128) -> DenseInputLayer<f32> {
        
        let inputs = if input_size % capacity == 0 && input_size / capacity >= 2 { 
            input_size / capacity 
        } else {
            panic!("Input size needs to be a multiple of the number of units.") 
        };

        let mut layer: DenseInputLayer<f32> = DenseInputLayer::new(capacity);
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

impl DenseInputLayer<f64> {
    pub fn init(
        capacity: usize,
        input_size: usize,
        acti: ActFunction,
        scale: f64,
        seed: &mut u128) -> DenseInputLayer<f64> {
        
        let inputs = if input_size % capacity == 0 && input_size / capacity >= 2 { 
            input_size / capacity 
        } else { 
            panic!("Input size needs to be a multiple of the number of units.") 
        };

        let mut layer: DenseInputLayer<f64> = DenseInputLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {elm * lcgf64(seed) - (scale / 2.0)})
                        .collect(), 
                    scale * lcgf64(seed) - (scale / 2.0), 
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
        acti: ActFunction,
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

impl DenseLayer<f64> {
    pub fn init(
        capacity: usize,
        inputs: usize,
        acti: ActFunction,
        scale: f64,
        seed: &mut u128) -> DenseLayer<f64> {
        
        let mut layer: DenseLayer<f64> = DenseLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {elm * lcgf64(seed) - (scale / 2.0)})
                        .collect(), 
                    scale * lcgf64(seed) - (scale / 2.0), 
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