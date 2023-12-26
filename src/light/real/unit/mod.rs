pub mod neuron;

use super::Param;
use neuron::Neuron;

pub enum ProcessingUnit<P: Param> {
    Neuron(Neuron<P>)
}