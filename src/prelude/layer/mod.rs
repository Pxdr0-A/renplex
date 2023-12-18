pub mod dense;


use std::marker::PhantomData;

use super::neuron::{param::Param, UnitLike};


// maybe supertrait for InputLayerLike?
pub trait LayerLike<P, U> 
    where
        P: Param,
        U: UnitLike<P> {
    
    fn new(n_units: usize) -> Self;

    fn add(&mut self, unit: U);

    fn signal(&self, input: &[P]) -> Vec<P>;

}

pub enum InputLayer<P, U, L>
    where
        P: Param,
        U: UnitLike<P>,
        L: LayerLike<P, U>, {
    
    VoidP(PhantomData<P>),
    VoidU(PhantomData<U>),
    DenseInputLayer(L)
}


pub enum Layer<P, U, L>
    where
        P: Param,
        U: UnitLike<P>,
        L: LayerLike<P, U>, {
    
    VoidP(PhantomData<P>),
    VoidU(PhantomData<U>),
    DenseLayer(L),
}
