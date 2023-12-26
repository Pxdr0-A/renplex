pub mod dense;

use super::Param;
use dense::DenseNeuron;

pub enum ProcessingUnit<P: Param> {
    DenseNeuron(DenseNeuron<P>)
}