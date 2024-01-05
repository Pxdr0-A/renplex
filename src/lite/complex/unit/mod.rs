pub mod dense;

use super::ComplexParam;
use dense::DenseCNeuron;

pub enum ComplexProcessingUnit<CP: ComplexParam> {
    DenseCNeuron(DenseCNeuron<CP>)
}