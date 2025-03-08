use std::{collections::HashMap, marker::PhantomData};

use activations::Activation;
use init::StandardInit;
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

        fn core_ref(&self) -> (&[Self::C], &[usize]) {
            (self._array.as_slice(), self._shape.as_slice())
        }
    }
}

type InitArgs = HashMap<&'static str, &'static str>;

// generic functions for defining forward passes like Linear, Convolutional, ...
pub trait StandardCore {
    // put here a generic for standard initialization
    // plan what will happen next
    fn init<TW: Tensor, TB: Tensor>(args: &InitArgs) -> (TW, TB);

    fn compute<TW: Tensor, TB: Tensor, TI: Tensor, TO: Tensor>(
        weights: &TW,
        bias: &TB,
        input: &TI,
    ) -> TO;

    // for the derivatives and so on
}

pub struct LinearCore;

impl StandardCore for LinearCore {
    fn init<TW: Tensor, TB: Tensor>(args: &InitArgs) -> (TW, TB) {
        unimplemented!()
    }

    fn compute<TW: Tensor, TB: Tensor, TI: Tensor, TO: Tensor>(
        weights: &TW,
        bias: &TB,
        input: &TI,
    ) -> TO {
        unimplemented!()
    }
}

pub trait Module {
    type Weight: Tensor;
    type Bias: Tensor;
    type Output: Tensor;

    fn init(args: &InitArgs) -> Self;

    fn get_mut_params(&mut self) -> (&mut Self::Weight, &mut Self::Bias);

    fn forward<T: Tensor>(&self, input: &T) -> Self::Output;

    fn backward<T: Tensor>(&self, grad: &Self::Output, input: &T) -> T;

    fn gradient<T: Tensor>(&self, grad: &Self::Output, input: &T) -> (Self::Weight, Self::Bias);
}

// layer with a core that only accepts standard initialization
pub struct StandardModule<C, TW, TB, TO>
where
    C: StandardCore,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    core: PhantomData<C>,
    weights: TW,
    bias: TB,
    output: PhantomData<TO>,
}

impl<C, TW, TB, TO> Module for StandardModule<C, TW, TB, TO>
where
    C: StandardCore,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    type Weight = TW;
    type Bias = TB;
    type Output = TO;

    fn init(args: &InitArgs) -> Self {
        let (weights, bias) = C::init(args);

        Self {
            core: PhantomData,
            weights,
            bias,
            output: PhantomData,
        }
    }

    fn get_mut_params(&mut self) -> (&mut Self::Weight, &mut Self::Bias) {
        unimplemented!()
    }

    fn forward<T: Tensor>(&self, input: &T) -> Self::Output {
        let (weights, bias) = (&self.weights, &self.bias);
        C::compute(weights, bias, input)
    }

    fn backward<T: Tensor>(&self, grad: &Self::Output, input: &T) -> T {
        unimplemented!()
    }

    fn gradient<T: Tensor>(&self, grad: &Self::Output, input: &T) -> (Self::Weight, Self::Bias) {
        unimplemented!()
    }
}

pub type StaticLinear<C, const LENW: usize, const LENB: usize, const LENO: usize> =
    StandardModule<LinearCore, StaticTensor<C, LENW>, StaticTensor<C, LENB>, StaticTensor<C, LENO>>;

// define here more types
pub struct StandardLayer<C, A, TW, TB, TO>
where
    C: StandardCore,
    A: Activation,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    plain_layer: StandardModule<C, TW, TB, TO>,
    activation: A,
}

impl<C, A, TW, TB, TO> Module for StandardLayer<C, A, TW, TB, TO>
where
    C: StandardCore,
    A: Activation,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    type Weight = TW;
    type Bias = TB;
    type Output = TO;

    fn init(args: &InitArgs) -> Self {
        unimplemented!()
    }

    fn get_mut_params(&mut self) -> (&mut Self::Weight, &mut Self::Bias) {
        unimplemented!()
    }

    fn forward<T: Tensor>(&self, input: &T) -> Self::Output {
        unimplemented!()
    }

    fn backward<T: Tensor>(&self, grad: &Self::Output, input: &T) -> T {
        unimplemented!()
    }

    fn gradient<T: Tensor>(&self, grad: &Self::Output, input: &T) -> (Self::Weight, Self::Bias) {
        unimplemented!()
    }
}

pub mod init {
    use crate::tensor::Tensor;

    pub trait StandardInit {
        fn generate<T: Tensor>(ni: usize, no: usize) -> T;
    }

    pub struct UniformXG;

    impl StandardInit for UniformXG {
        fn generate<T: Tensor>(ni: usize, no: usize) -> T {
            unimplemented!()
        }
    }

    pub struct NormalXG;

    impl StandardInit for NormalXG {
        fn generate<T: Tensor>(ni: usize, no: usize) -> T {
            unimplemented!()
        }
    }

    pub struct NormalHK;

    impl StandardInit for NormalHK {
        fn generate<T: Tensor>(ni: usize, no: usize) -> T {
            unimplemented!()
        }
    }
}

pub mod optimization {
    use crate::{tensor::Tensor, Module};

    pub trait Loss {}

    // I think this is alright
    pub trait Optimizer {
        fn step<M: Module, W: Tensor, B: Tensor>(&mut self, module: &mut M, gradw: &W, gradb: &B);
    }
}

pub mod modules {}

pub mod activations {
    use crate::{tensor::Tensor, InitArgs};

    pub trait Activation {
        // initialization arguments are passed on from parent module
        fn init(args: &InitArgs) -> Self;

        fn update_params(&mut self);

        fn compute<T: Tensor>(&self, input: &mut T);

        fn derivative<T: Tensor>(&self, input: &mut T);
    }

    pub struct Tanh;

    impl Activation for Tanh {
        fn init(_args: &InitArgs) -> Self {
            Tanh
        }

        fn update_params(&mut self) {}

        fn compute<T: Tensor>(&self, input: &mut T) {
            unimplemented!()
        }

        fn derivative<T: Tensor>(&self, input: &mut T) {
            unimplemented!()
        }
    }

    pub struct RawDropout {
        config: Vec<usize>,
        prob: f32,
    }

    impl Activation for RawDropout {
        fn init(args: &InitArgs) -> Self {
            unimplemented!()
        }

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

    pub struct Dropout<A: Activation> {
        act: A,
        config: Vec<usize>,
        prob: f32,
    }

    impl<A: Activation> Activation for Dropout<A> {
        fn init(args: &InitArgs) -> Self {
            unimplemented!()
        }

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
