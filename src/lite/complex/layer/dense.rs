use crate::math::cfloat::{Cf32, Cf64};
use crate::math::random::{lcgf32, lcgf64};
use crate::lite::complex::{ComplexParam, ComplexActFunction};
use crate::lite::complex::unit::dense::DenseCNeuron;
use super::{ComplexInputLayer, ComplexLayer};

#[derive(Debug)]
pub struct DenseCInputLayer<CP: ComplexParam> {
    units: Vec<DenseCNeuron<CP>>
}

impl<CP: ComplexParam + Copy> DenseCInputLayer<CP> {
    pub fn new(capacity: usize) -> DenseCInputLayer<CP> {
        DenseCInputLayer {
            units: Vec::with_capacity(capacity)
        }
    }

    pub fn add(&mut self, neuron: DenseCNeuron<CP>) {
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
    pub fn signal(&self, input: &[CP]) -> Vec<CP> {
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

    pub fn wrap(self) -> ComplexInputLayer<CP> {
        ComplexInputLayer::DenseCInputLayer(self)
    }
}

impl DenseCInputLayer<Cf32> {
    pub fn init(
        capacity: usize,
        input_size: usize,
        acti: ComplexActFunction,
        scale: f32,
        seed: &mut u128) -> DenseCInputLayer<Cf32> {
        
        let inputs = if input_size % capacity == 0 && input_size / capacity >= 2 { 
            input_size / capacity 
        } else { 
            panic!("Input size needs to be a multiple of the number of units.") 
        };

        let mut layer: DenseCInputLayer<Cf32> = DenseCInputLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseCNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {
                            Cf32 {
                                x: elm * lcgf32(seed) - (scale / 2.0), 
                                y: elm * lcgf32(seed) - (scale / 2.0)
                            }
                        })
                        .collect(), 
                        Cf32 {
                            x: scale * lcgf32(seed) - (scale / 2.0), 
                            y: scale * lcgf32(seed) - (scale / 2.0)
                        }, 
                    acti.clone()
                )
            );
        }

        layer
    }
}

impl DenseCInputLayer<Cf64> {
    pub fn init(
        capacity: usize,
        input_size: usize,
        acti: ComplexActFunction,
        scale: f64,
        seed: &mut u128) -> DenseCInputLayer<Cf64> {
        
        let inputs = if input_size % capacity == 0 && input_size / capacity >= 2 { 
            input_size / capacity 
        } else { 
            panic!("Input size needs to be a multiple of the number of units.") 
        };

        let mut layer: DenseCInputLayer<Cf64> = DenseCInputLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseCNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {
                            Cf64 {
                                x: elm * lcgf64(seed) - (scale / 2.0), 
                                y: elm * lcgf64(seed) - (scale / 2.0)
                            }
                        })
                        .collect(), 
                        Cf64 {
                            x: scale * lcgf64(seed) - (scale / 2.0), 
                            y: scale * lcgf64(seed) - (scale / 2.0)
                        }, 
                    acti.clone()
                )
            );
        }

        layer
    }
}

#[derive(Debug)]
pub struct DenseCLayer<CP: ComplexParam> {
    units: Vec<DenseCNeuron<CP>>
}

impl<CP: ComplexParam + Copy> DenseCLayer<CP> {
    pub fn new(capacity: usize) -> DenseCLayer<CP> {
        DenseCLayer {
            units: Vec::with_capacity(capacity)
        }
    }

    pub fn add(&mut self, neuron: DenseCNeuron<CP>) {
        self.units.push(neuron);
    }

    /// Returns a new Vec resultant from fowarding an input through a hidden layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    pub fn signal(&self, input: &[CP]) -> Vec<CP> {
        let mut output = Vec::with_capacity(self.units.len());
        for neuron in &self.units {
            output.push(
                neuron
                    .signal(input)
            );
        }

        output
    }

    pub fn wrap(self) -> ComplexLayer<CP> {
        ComplexLayer::DenseCLayer(self)
    }
}

impl DenseCLayer<Cf32> {
    pub fn init(
        capacity: usize,
        inputs: usize,
        acti: ComplexActFunction,
        scale: f32,
        seed: &mut u128) -> DenseCLayer<Cf32> {
        
        let mut layer: DenseCLayer<Cf32> = DenseCLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseCNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {
                            Cf32 {
                                x: elm * lcgf32(seed) - (scale / 2.0), 
                                y: elm * lcgf32(seed) - (scale / 2.0)
                            }
                        })
                        .collect(), 
                    Cf32 {
                            x: scale * lcgf32(seed) - (scale / 2.0), 
                            y: scale * lcgf32(seed) - (scale / 2.0)
                        }, 
                    acti.clone()
                )
            );
        }

        layer
    }
}

impl DenseCLayer<Cf64> {
    pub fn init(
        capacity: usize,
        inputs: usize,
        acti: ComplexActFunction,
        scale: f64,
        seed: &mut u128) -> DenseCLayer<Cf64> {
        
        let mut layer: DenseCLayer<Cf64> = DenseCLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseCNeuron::new(
                    vec![scale; inputs]
                        .into_iter()
                        .map(|elm| {
                            Cf64 {
                                x: elm * lcgf64(seed) - (scale / 2.0), 
                                y: elm * lcgf64(seed) - (scale / 2.0)
                            }
                        })
                        .collect(), 
                    Cf64 {
                            x: scale * lcgf64(seed) - (scale / 2.0), 
                            y: scale * lcgf64(seed) - (scale / 2.0)
                        }, 
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