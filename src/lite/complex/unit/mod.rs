pub mod dense;

use super::ComplexParam;
use dense::DenseNeuron;

pub enum ProcessingUnit<CP: ComplexParam> {
    DenseNeuron(DenseNeuron<CP>)
}