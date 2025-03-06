use std::{collections::HashMap, marker::PhantomData};

use num_complex::Complex32;
use tensor::{ComplexRoutines, StaticTensor, Tensor};
// std and dependencies imports
// definition of internal mods

pub mod tensor {
    use num_complex::{Complex32, Complex64};

    // useful type
    pub type Shape = Vec<usize>;
    pub type ShapeSlice<'a> = &'a [usize];

    mod aocl {
        extern "C" {}
    }

    pub trait ComplexRoutines {
        fn matmul<T1: Tensor, T2: Tensor, T3: Tensor>(lhs: &T1, rhs: &T2) -> T3;

        fn tanh<T: Tensor>(t: &T) -> T;
    }

    impl ComplexRoutines for Complex32 {
        fn matmul<T1: Tensor, T2: Tensor, T3: Tensor>(lhs: &T1, rhs: &T2) -> T3 {
            unimplemented!()
        }

        fn tanh<T: Tensor>(t: &T) -> T {
            unimplemented!()
        }
    }

    impl ComplexRoutines for Complex64 {
        fn matmul<T1: Tensor, T2: Tensor, T3: Tensor>(lhs: &T1, rhs: &T2) -> T3 {
            unimplemented!()
        }

        fn tanh<T: Tensor>(t: &T) -> T {
            unimplemented!()
        }
    }

    // tensor definitions, operations, etc.
    pub trait Tensor {
        type C: ComplexRoutines;

        // probabily do not need precision here
        // Initialization
        fn new(_shape: Shape) -> Self;

        // Core Properties of the Tensor
        fn core_ref(&self) -> (&[Self::C], &[usize]);
    }

    pub struct StaticTensor<C: ComplexRoutines, const LEN: usize> {
        _array: [C; LEN],
        _shape: Shape,
    }

    impl<C: ComplexRoutines, const LEN: usize> Tensor for StaticTensor<C, LEN> {
        type C = C;

        fn new(_shape: Shape) -> Self {
            unimplemented!()
        }

        fn core_ref(&self) -> (&[C], &[usize]) {
            (self._array.as_slice(), self._shape.as_slice())
        }
    }
}

// generic functions for defining forward passes like Linear or Convolutional
// might not be a bad idea to have another generic type for the bias
pub trait ModuleCore {
    fn compute<TW: Tensor, TO: Tensor, TI: Tensor>(weights: &TW, bias: &TO, input: &TI) -> TO;

    // for the derivatives and so on
}

pub struct LinearCore;

impl ModuleCore for LinearCore {
    fn compute<TW: Tensor, TO: Tensor, TI: Tensor>(weights: &TW, bias: &TO, input: &TI) -> TO {
        unimplemented!()
    }
}

pub trait Module {
    type Weight: Tensor;
    type Output: Tensor;

    fn forward<T: Tensor>(&self, input: &T) -> Self::Output;

    fn backward<T: Tensor>(&self, grad: &Self::Output, input: &T) -> T;

    fn gradient<T: Tensor>(&self, grad: &Self::Output, input: &T) -> (Self::Weight, Self::Output);
}

pub struct GeneralModule<C, TW, TO>
where
    C: ModuleCore,
    TW: Tensor,
    TO: Tensor,
{
    core: C,
    weights: TW,
    bias: TO,
}

pub type StaticLinear<C, const LENW: usize, const LENO: usize> =
    GeneralModule<LinearCore, StaticTensor<C, LENW>, StaticTensor<C, LENO>>;

// define here more types
pub struct GeneralLayer<C, TW, TO>
where
    C: ModuleCore,
    TW: Tensor,
    TO: Tensor,
{
    plain_layer: GeneralModule<C, TW, TO>,
    activation: PhantomData<TW>,
}

impl<C, TW, TO> Module for GeneralModule<C, TW, TO>
where
    C: ModuleCore,
    TW: Tensor,
    TO: Tensor,
{
    // This one seems to be needed but I do not know about the others?
    type Weight = TW;
    type Output = TO;

    fn forward<T: Tensor>(&self, input: &T) -> Self::Output {
        let (weights, bias) = (&self.weights, &self.bias);
        C::compute(weights, bias, input)
    }

    fn backward<T: Tensor>(&self, grad: &Self::Output, input: &T) -> T {
        unimplemented!()
    }

    fn gradient<T: Tensor>(&self, grad: &Self::Output, input: &T) -> (Self::Weight, Self::Output) {
        unimplemented!()
    }
}

// TACKLE THIS NOW!!

pub mod init {
    pub trait Initialization {}
}

pub mod optimization {
    use crate::{tensor::Tensor, Module};

    pub trait Optimizer {
        fn step<M: Module, W: Tensor, B: Tensor>(&mut self, module: &mut M, gradw: &W, gradb: &B);
    }
}

pub mod modules {}

pub mod activations {
    use crate::tensor::Tensor;

    pub trait Activation {
        fn update_params(&mut self);

        fn compute<T: Tensor>(&self, input: &mut T);

        fn derivative<T: Tensor>(&self, input: &mut T);
    }

    pub struct Tanh;

    impl Activation for Tanh {
        fn update_params(&mut self) {
            unimplemented!()
        }

        fn compute<T: Tensor>(&self, input: &mut T) {
            unimplemented!()
        }

        fn derivative<T: Tensor>(&self, input: &mut T) {
            unimplemented!()
        }
    }

    pub struct Dropout<P> {
        parameters: P,
    }

    impl<P> Activation for Dropout<P> {
        fn update_params(&mut self) {
            unimplemented!()
        }

        fn compute<T: Tensor>(&self, input: &mut T) {
            unimplemented!()
        }

        fn derivative<T: Tensor>(&self, input: &mut T) {
            unimplemented!()
        }
    }
}

mod dataset {}
