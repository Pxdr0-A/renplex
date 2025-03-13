use std::{collections::HashMap, marker::PhantomData};

use activations::Activation;
use init::StandardInit;
use tensor::{StaticTensor, Tensor};
// std and dependencies imports
// definition of internal mods

pub mod tensor {
    use routines::ComplexRoutines;

    // useful type
    pub type Shape = Vec<usize>;
    pub type ShapeSlice<'a> = &'a [usize];

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

    mod routines {
        use num_complex::{Complex32, Complex64};

        use super::Tensor;

        mod raw {
            extern "C" {}
        }

        pub trait ComplexRoutines {
            fn matmul<T1: Tensor, T2: Tensor, T3: Tensor>(lhs: &T1, rhs: &T2) -> T3;
        }

        // find a way to copy the logic from one to the other ((16) -> 32 -> 64 -> (128))
        // macro that passes the routines?
        impl ComplexRoutines for Complex32 {
            fn matmul<T1: Tensor, T2: Tensor, T3: Tensor>(lhs: &T1, rhs: &T2) -> T3 {
                unimplemented!()
            }
        }

        impl ComplexRoutines for Complex64 {
            fn matmul<T1: Tensor, T2: Tensor, T3: Tensor>(lhs: &T1, rhs: &T2) -> T3 {
                unimplemented!()
            }
        }
    }
}

type InitArgs = HashMap<&'static str, &'static str>;

// generic functions like Linear, Convolutional, ... that involve a standard initialization based on input and output feature numbers
pub trait StandardCore {
    fn io_features(args: &InitArgs) -> (usize, usize);

    fn compute<TW: Tensor, TB: Tensor, TI: Tensor, TO: Tensor>(
        weights: &TW,
        bias: &TB,
        input: &TI,
    ) -> TO;

    // for the derivatives and so on
}

pub struct LinearCore;

impl StandardCore for LinearCore {
    fn io_features(args: &InitArgs) -> (usize, usize) {
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
    type Init;

    fn init(args: &InitArgs) -> Self;

    fn get_mut_params(&mut self) -> (&mut Self::Weight, &mut Self::Bias);

    fn forward<T: Tensor>(&self, input: &T) -> Self::Output;

    fn backward<T: Tensor>(&self, grad: &Self::Output, input: &T) -> T;

    fn gradient<T: Tensor>(&self, grad: &Self::Output, input: &T) -> (Self::Weight, Self::Bias);
}

// layer with a core that only accepts standard initialization
pub struct StandardModule<SC, SI, TW, TB, TO>
where
    SC: StandardCore,
    SI: StandardInit,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    core: PhantomData<SC>,
    standard_init: PhantomData<SI>,
    weights: TW,
    bias: TB,
    output: PhantomData<TO>,
}

impl<SC, SI, TW, TB, TO> Module for StandardModule<SC, SI, TW, TB, TO>
where
    SC: StandardCore,
    SI: StandardInit,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    type Weight = TW;
    type Bias = TB;
    type Output = TO;
    type Init = SI;
    // Initialization should enter here as a generic

    fn init(args: &InitArgs) -> Self {
        let (ni, no) = SC::io_features(args);
        let weights = SI::generate(ni, no);
        let bias = Self::Bias::new(vec![no]);

        Self {
            core: PhantomData,
            standard_init: PhantomData,
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
        SC::compute(weights, bias, input)
    }

    fn backward<T: Tensor>(&self, grad: &Self::Output, input: &T) -> T {
        unimplemented!()
    }

    fn gradient<T: Tensor>(&self, grad: &Self::Output, input: &T) -> (Self::Weight, Self::Bias) {
        unimplemented!()
    }
}

pub type StaticLinear<C, SI, const LENW: usize, const LENB: usize, const LENO: usize> =
    StandardModule<
        LinearCore,
        SI,
        StaticTensor<C, LENW>,
        StaticTensor<C, LENB>,
        StaticTensor<C, LENO>,
    >;

// define here more types
pub struct StandardLayer<SC, SI, A, TW, TB, TO>
where
    SC: StandardCore,
    SI: StandardInit,
    A: Activation,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    plain_layer: StandardModule<SC, SI, TW, TB, TO>,
    activation: A,
}

impl<SC, SI, A, TW, TB, TO> Module for StandardLayer<SC, SI, A, TW, TB, TO>
where
    SC: StandardCore,
    SI: StandardInit,
    A: Activation,
    TW: Tensor,
    TB: Tensor,
    TO: Tensor,
{
    type Weight = TW;
    type Bias = TB;
    type Output = TO;
    type Init = SI;

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
