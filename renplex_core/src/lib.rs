use std::{collections::HashMap, marker::PhantomData};

use num_complex::Complex32;
use tensor::{StaticTensor, Tensor};
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

pub trait Module {
    type Input: Tensor;
    type Output: Tensor;
    type Weight: Tensor;

    fn init(args: HashMap<String, String>) -> Self;

    fn get_params(&self) -> (&Self::Weight, &Self::Output);

    fn get_mut_params(&mut self) -> (&mut Self::Weight, &mut Self::Output);

    fn get_ff(&self) -> impl Fn(&Self::Input, &Self::Weight, &Self::Output) -> Self::Output;

    fn get_fdx(
        &self,
    ) -> impl Fn(&Self::Output, &Self::Input, &Self::Weight, &Self::Output) -> Self::Input;

    fn get_fdw(
        &self,
    ) -> impl Fn(&Self::Output, &Self::Input, &Self::Weight, &Self::Output) -> (Self::Weight, Self::Output);

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let forward_func = self.get_ff();
        let (weights, bias) = self.get_params();

        forward_func(input, weights, bias)
    }

    fn backward(&self, grad: &Self::Output, input: &Self::Input) -> Self::Input {
        unimplemented!()
    }

    fn gradient(&self, grad: &Self::Output, input: &Self::Input) -> (Self::Weight, Self::Output) {
        unimplemented!()
    }
}

pub struct ModuleLogic<FF, FDx, FDw, TI, TO, TW>
where
    TI: Tensor,
    TO: Tensor,
    TW: Tensor,
    FF: Fn(&TI, &TW, &TO) -> TO,
    FDx: Fn(&TO, &TI, &TW, &TO) -> TI,
    FDw: Fn(&TO, &TI, &TW, &TO) -> (TW, TO),
{
    funcf: FF,
    funcdx: FDx,
    funcdw: FDw,
    weights: TW,
    bias: TO,
    _phantomi: PhantomData<TI>,
}

// and then a module with activation

impl<FF, FDx, FDw, TI, TO, TW> Module for ModuleLogic<FF, FDx, FDw, TI, TO, TW>
where
    TI: Tensor,
    TO: Tensor,
    TW: Tensor,
    FF: Fn(&TI, &TW, &TO) -> TO,
    FDx: Fn(&TO, &TI, &TW, &TO) -> TI,
    FDw: Fn(&TO, &TI, &TW, &TO) -> (TW, TO),
{
    type Input = TI;
    type Output = TO;
    type Weight = TW;

    fn init(args: HashMap<String, String>) -> Self {
        unimplemented!()
    }

    fn get_params(&self) -> (&Self::Weight, &Self::Output) {
        (&self.weights, &self.bias)
    }

    fn get_mut_params(&mut self) -> (&mut Self::Weight, &mut Self::Output) {
        (&mut self.weights, &mut self.bias)
    }

    fn get_ff(&self) -> impl Fn(&TI, &Self::Weight, &Self::Output) -> Self::Output {
        &self.funcf
    }

    fn get_fdx(
        &self,
    ) -> impl Fn(&Self::Output, &Self::Input, &Self::Weight, &Self::Output) -> Self::Input {
        &self.funcdx
    }

    fn get_fdw(
        &self,
    ) -> impl Fn(&Self::Output, &Self::Input, &Self::Weight, &Self::Output) -> (Self::Weight, Self::Output)
    {
        &self.funcdw
    }
}

// maybe a DynModule and a StaticModule
// each one has a precision type that implements Precision trait
// and also input and output tensors
// statics have the addition of two constants for the sizes

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

    // just to add directly to a module
    pub trait Activation {
        fn compute<T: Tensor>(input: &mut T);

        fn derivative<T: Tensor>(input: &mut T);
    }
}

mod dataset {}
