use crate::math::cfloat::{Cf32, Cf64};
use crate::math::random::{lcgf32, lcgf64};
use crate::lite::complex::{ComplexParam, ComplexActFunction};
use crate::lite::complex::unit::dense::DenseNeuron;
use super::{InputLayer, Layer};

#[derive(Debug)]
pub struct DenseInputLayer<CP: ComplexParam> {
    input_size: usize,
    units: Vec<DenseNeuron<CP>>
}

impl<CP: ComplexParam + Copy> DenseInputLayer<CP> {
    pub fn new(capacity: usize, input_size: usize) -> DenseInputLayer<CP> {
        DenseInputLayer {
            input_size,
            units: Vec::with_capacity(capacity)
        }
    }

    pub fn add(&mut self, neuron: DenseNeuron<CP>) {
        self.units.push(neuron);
    }

    pub fn set_input_size(&mut self, size: usize) {
        if self.input_size == size { panic!("Input size and size not matching.") }

        let entries: usize = self.units
            .iter()
            .map(|elm| {elm.get_weights().len()})
            .sum();

        if size % entries == 0 && size / entries >= 2 { 
            self.input_size = size;
        } else {
            panic!("Entries are not a multiple of the input size")
        }
    }

    /// Returns a new Vec resultant from fowarding a signal through the input layer.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice of the input to foward to the layer. 
    ///             Needs to be in agreement with the number of units and respective neuron inputs.
    pub fn signal(&self, input: &[CP]) -> Vec<CP> {
        if self.input_size == input.len() { panic!("Input size not matching input length.") }

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

    pub fn wrap(self) -> InputLayer<CP> {
        InputLayer::DenseInputLayer(self)
    }
}

impl DenseInputLayer<Cf32> {
    pub fn init(
        capacity: usize,
        input_size: usize,
        acti: ComplexActFunction,
        scale: f32,
        seed: &mut u128) -> DenseInputLayer<Cf32> {
        
        let inputs = if input_size % capacity == 0 && input_size / capacity >= 2 { 
            input_size / capacity 
        } else { 
            panic!("Input size needs to be a multiple of the number of units.") 
        };

        let mut layer: DenseInputLayer<Cf32> = DenseInputLayer::new(capacity, input_size);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
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

impl DenseInputLayer<Cf64> {
    pub fn init(
        capacity: usize,
        input_size: usize,
        acti: ComplexActFunction,
        scale: f64,
        seed: &mut u128) -> DenseInputLayer<Cf64> {
        
        let inputs = if input_size % capacity == 0 && input_size / capacity >= 2 { 
            input_size / capacity 
        } else { 
            panic!("Input size needs to be a multiple of the number of units.") 
        };

        let mut layer: DenseInputLayer<Cf64> = DenseInputLayer::new(capacity, input_size);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
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
pub struct DenseLayer<CP: ComplexParam> {
    units: Vec<DenseNeuron<CP>>
}

impl<CP: ComplexParam + Copy> DenseLayer<CP> {
    pub fn new(capacity: usize) -> DenseLayer<CP> {
        DenseLayer {
            units: Vec::with_capacity(capacity)
        }
    }

    pub fn add(&mut self, neuron: DenseNeuron<CP>) {
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

    pub fn wrap(self) -> Layer<CP> {
        Layer::DenseLayer(self)
    }
}

impl DenseLayer<Cf32> {
    pub fn init(
        capacity: usize,
        inputs: usize,
        acti: ComplexActFunction,
        scale: f32,
        seed: &mut u128) -> DenseLayer<Cf32> {
        
        let mut layer: DenseLayer<Cf32> = DenseLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
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

impl DenseLayer<Cf64> {
    pub fn init(
        capacity: usize,
        inputs: usize,
        acti: ComplexActFunction,
        scale: f64,
        seed: &mut u128) -> DenseLayer<Cf64> {
        
        let mut layer: DenseLayer<Cf64> = DenseLayer::new(capacity);
        for _ in 0..capacity {
            layer.add(
                DenseNeuron::new(
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