use crate::lite::real::Param;
use crate::lite::real::unit::dense::DenseNeuron;


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
    pub fn signal(&self, input: &[P]) -> Result<Vec<P>, UnsetInputError> {
        match self.input_size != input.len() {
            true => {},
            false => { return Err(UnsetInputError::ClashingInput(self.input_size, input.len())); }
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
                    .unwrap()
            );

            // The +1 is to move one place. 
            // We do not want to repeat the last point
            position = demand + 1;

        }

        Ok(output)
    }
}

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
                    .unwrap()
            );
        }

        output
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