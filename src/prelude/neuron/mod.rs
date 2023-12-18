pub mod dense;
pub mod param;

use std::marker::PhantomData;

use self::param::Param;


#[derive(Debug, Clone)]
pub enum ActivationFunction {
    SIGMOID,
    TANH,
    RELU,
}


pub trait UnitLike<P: Param> {

    fn new(weights: Vec<P>, bias: P, act_func: ActivationFunction) -> Self;

    fn get_weights(&self) -> &[P];

    fn signal(&self, input: &[P]) -> P;

}

pub enum ProcessingUnit<P, U>
    where
        P: Param,
        U: UnitLike<P> {
    
    VoidP(PhantomData<P>),
    DenseNeuron(U)
}
