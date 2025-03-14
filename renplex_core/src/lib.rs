use std::collections::HashMap;
// std and dependencies imports
// definition of internal mods

type InitArgs = HashMap<&'static str, &'static str>;

pub mod tensor {
    use routines::ComplexRoutine;

    // useful type
    pub type Shape = Vec<usize>;

    // tensor definitions, operations, etc.
    // I think the trait needs to have the type as generic
    pub trait Tensor<C: ComplexRoutine> {
        // probabily do not need precision here
        // Initialization
        fn new(_shape: Shape) -> Self;

        // Core Properties of the Tensor
        fn core_ref(&self) -> (&[C], &[usize]);
    }

    pub struct StaticTensor<C: ComplexRoutine, const LEN: usize> {
        _array: [C; LEN],
        _shape: Shape,
    }

    impl<C: ComplexRoutine, const LEN: usize> Tensor<C> for StaticTensor<C, LEN> {
        fn new(_shape: Shape) -> Self {
            unimplemented!()
        }

        fn core_ref(&self) -> (&[C], &[usize]) {
            (self._array.as_slice(), self._shape.as_slice())
        }
    }

    pub mod routines {
        use num_complex::{Complex32, Complex64, ComplexFloat};

        use super::Tensor;

        mod raw {
            extern "C" {}
        }

        pub trait ComplexRoutine
        where
            Self: ComplexFloat,
        {
            fn matmul<T1: Tensor<Self>, T2: Tensor<Self>, T3: Tensor<Self>>(
                lhs: &T1,
                rhs: &T2,
            ) -> T3;

            fn scale<T: Tensor<Self>>(targ: &mut T, val: Self);
        }

        impl ComplexRoutine for Complex32 {
            fn matmul<T1: Tensor<Self>, T2: Tensor<Self>, T3: Tensor<Self>>(
                lhs: &T1,
                rhs: &T2,
            ) -> T3 {
                unimplemented!()
            }

            fn scale<T: Tensor<Self>>(targ: &mut T, val: Self) {
                unimplemented!()
            }
        }

        impl ComplexRoutine for Complex64 {
            fn matmul<T1: Tensor<Self>, T2: Tensor<Self>, T3: Tensor<Self>>(
                lhs: &T1,
                rhs: &T2,
            ) -> T3 {
                unimplemented!()
            }

            fn scale<T: Tensor<Self>>(targ: &mut T, val: Self) {
                unimplemented!()
            }
        }
    }
}

pub mod module_cores {

    use crate::{
        tensor::{routines::ComplexRoutine, Tensor},
        InitArgs,
    };

    pub struct LinearCore;

    // generic functions like Linear, Convolutional, ... that involve a standard initialization based on input and output feature numbers
    pub trait StandardCore {
        fn io_features(args: &InitArgs) -> (usize, usize);

        fn forward<C: ComplexRoutine, TW: Tensor<C>, TB: Tensor<C>, TI: Tensor<C>, TO: Tensor<C>>(
            weights: &TW,
            bias: &TB,
            input: &TI,
        ) -> TO;

        fn backward<C: ComplexRoutine, TW: Tensor<C>, TB: Tensor<C>, TI: Tensor<C>, TO: Tensor<C>>(
            weights: &TW,
            bias: &TB,
            input: &TI,
            grad: &TO,
        ) -> TI;

        fn gradient<C: ComplexRoutine, TW: Tensor<C>, TB: Tensor<C>, TI: Tensor<C>, TO: Tensor<C>>(
            weights: &TW,
            bias: &TB,
            input: &TI,
            grad: &TO,
        ) -> (TW, TB);
    }

    impl StandardCore for LinearCore {
        fn io_features(args: &InitArgs) -> (usize, usize) {
            unimplemented!()
        }

        fn forward<
            C: ComplexRoutine,
            TW: Tensor<C>,
            TB: Tensor<C>,
            TI: Tensor<C>,
            TO: Tensor<C>,
        >(
            weights: &TW,
            bias: &TB,
            input: &TI,
        ) -> TO {
            unimplemented!()
        }

        fn backward<
            C: ComplexRoutine,
            TW: Tensor<C>,
            TB: Tensor<C>,
            TI: Tensor<C>,
            TO: Tensor<C>,
        >(
            weights: &TW,
            bias: &TB,
            input: &TI,
            grad: &TO,
        ) -> TI {
            unimplemented!()
        }

        fn gradient<
            C: ComplexRoutine,
            TW: Tensor<C>,
            TB: Tensor<C>,
            TI: Tensor<C>,
            TO: Tensor<C>,
        >(
            weights: &TW,
            bias: &TB,
            input: &TI,
            grad: &TO,
        ) -> (TW, TB) {
            unimplemented!()
        }
    }
}

pub mod modules {
    use std::marker::PhantomData;

    use crate::{
        activations::Activation,
        init::StandardInit,
        module_cores::{LinearCore, StandardCore},
        tensor::{routines::ComplexRoutine, StaticTensor, Tensor},
        InitArgs,
    };

    pub type StaticLinearModule<C, SI, const LENW: usize, const LENB: usize, const LENO: usize> =
        StandardModule<
            C,
            LinearCore,
            SI,
            StaticTensor<C, LENW>,
            StaticTensor<C, LENB>,
            StaticTensor<C, LENO>,
        >;

    pub type StaticLinearLayer<C, SI, A, const LENW: usize, const LENB: usize, const LENO: usize> =
        StandardLayer<
            C,
            LinearCore,
            SI,
            A,
            StaticTensor<C, LENW>,
            StaticTensor<C, LENB>,
            StaticTensor<C, LENO>,
        >;

    pub trait Module {
        type Precision: ComplexRoutine;
        type Weight: Tensor<Self::Precision>;
        type Bias: Tensor<Self::Precision>;
        type Output: Tensor<Self::Precision>;
        type Init;
        // might be interesting to have here the type of the core also

        fn init(args: &InitArgs) -> Self;

        fn get_mut_params(&mut self) -> (&mut Self::Weight, &mut Self::Bias);

        fn forward<T: Tensor<Self::Precision>>(&self, input: &T) -> Self::Output;

        fn backward<T: Tensor<Self::Precision>>(&self, input: &T, grad: &Self::Output) -> T;

        fn gradient<T: Tensor<Self::Precision>>(
            &self,
            input: &T,
            grad: &Self::Output,
        ) -> (Self::Weight, Self::Bias);
    }

    // layer with a core that only accepts standard initialization
    pub struct StandardModule<C, SC, SI, TW, TB, TO>
    where
        C: ComplexRoutine,
        SC: StandardCore,
        SI: StandardInit,
        TW: Tensor<C>,
        TB: Tensor<C>,
        TO: Tensor<C>,
    {
        precision: PhantomData<C>,
        core: PhantomData<SC>,
        standard_init: PhantomData<SI>,
        weights: TW,
        bias: TB,
        output: PhantomData<TO>,
    }

    impl<C, SC, SI, TW, TB, TO> Module for StandardModule<C, SC, SI, TW, TB, TO>
    where
        C: ComplexRoutine,
        SC: StandardCore,
        SI: StandardInit,
        TW: Tensor<C>,
        TB: Tensor<C>,
        TO: Tensor<C>,
    {
        type Precision = C;
        type Weight = TW;
        type Bias = TB;
        type Output = TO;
        type Init = SI;

        fn init(args: &InitArgs) -> Self {
            let (ni, no) = SC::io_features(args);
            let weights = SI::generate(ni, no);
            let bias = Self::Bias::new(vec![no]);

            Self {
                precision: PhantomData,
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

        fn forward<T: Tensor<Self::Precision>>(&self, input: &T) -> Self::Output {
            SC::forward(&self.weights, &self.bias, input)
        }

        fn backward<T: Tensor<Self::Precision>>(&self, input: &T, grad: &Self::Output) -> T {
            unimplemented!()
        }

        fn gradient<T: Tensor<Self::Precision>>(
            &self,
            input: &T,
            grad: &Self::Output,
        ) -> (Self::Weight, Self::Bias) {
            unimplemented!()
        }
    }

    // define here more types
    pub struct StandardLayer<C, SC, SI, A, TW, TB, TO>
    where
        C: ComplexRoutine,
        SC: StandardCore,
        SI: StandardInit,
        A: Activation,
        TW: Tensor<C>,
        TB: Tensor<C>,
        TO: Tensor<C>,
    {
        plain_layer: StandardModule<C, SC, SI, TW, TB, TO>,
        activation: A,
    }

    impl<C, SC, SI, A, TW, TB, TO> Module for StandardLayer<C, SC, SI, A, TW, TB, TO>
    where
        C: ComplexRoutine,
        SC: StandardCore,
        SI: StandardInit,
        A: Activation,
        TW: Tensor<C>,
        TB: Tensor<C>,
        TO: Tensor<C>,
    {
        type Precision = C;
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

        fn forward<T: Tensor<Self::Precision>>(&self, input: &T) -> Self::Output {
            unimplemented!()
        }

        fn backward<T: Tensor<Self::Precision>>(&self, input: &T, grad: &Self::Output) -> T {
            unimplemented!()
        }

        fn gradient<T: Tensor<Self::Precision>>(
            &self,
            input: &T,
            grad: &Self::Output,
        ) -> (Self::Weight, Self::Bias) {
            unimplemented!()
        }
    }
}

pub mod activations {
    use crate::{
        tensor::{routines::ComplexRoutine, Tensor},
        InitArgs,
    };

    pub trait Activation {
        // initialization arguments are passed on from parent module
        fn init(args: &InitArgs) -> Self;

        fn update_params(&mut self);

        fn forward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T);

        fn backward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T);
    }

    pub struct Tanh;

    impl Activation for Tanh {
        fn init(_args: &InitArgs) -> Self {
            Tanh
        }

        fn update_params(&mut self) {}

        fn forward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T) {
            unimplemented!()
        }

        fn backward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T) {
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

        fn forward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T) {
            unimplemented!()
        }

        fn backward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T) {
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

        fn forward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T) {
            unimplemented!()
        }

        fn backward<C: ComplexRoutine, T: Tensor<C>>(&self, input: &mut T) {
            unimplemented!()
        }
    }
}

pub mod init {
    use crate::tensor::{routines::ComplexRoutine, Tensor};

    pub trait StandardInit {
        fn generate<C: ComplexRoutine, T: Tensor<C>>(ni: usize, no: usize) -> T;
    }

    pub struct UniformXG;

    impl StandardInit for UniformXG {
        fn generate<C: ComplexRoutine, T: Tensor<C>>(ni: usize, no: usize) -> T {
            unimplemented!()
        }
    }

    pub struct NormalXG;

    impl StandardInit for NormalXG {
        fn generate<C: ComplexRoutine, T: Tensor<C>>(ni: usize, no: usize) -> T {
            unimplemented!()
        }
    }

    pub struct NormalHK;

    impl StandardInit for NormalHK {
        fn generate<C: ComplexRoutine, T: Tensor<C>>(ni: usize, no: usize) -> T {
            unimplemented!()
        }
    }
}

pub mod optimization {
    use crate::{
        tensor::{routines::ComplexRoutine, Tensor},
        InitArgs,
    };

    // the loss might not need precision right now
    // i do not think there are parametrized losses whose parameter types
    // need to be synced with the data
    pub trait Loss {
        fn forward<C: ComplexRoutine, T: Tensor<C>>(input: &T) -> T;

        fn backward<C: ComplexRoutine, T: Tensor<C>>(input: &T) -> T;
    }

    // You should be able to just say MSE
    // Maybe it is possible without the pahntom data
    pub struct MSE;

    impl Loss for MSE {
        fn forward<C: ComplexRoutine, T: Tensor<C>>(input: &T) -> T {
            unimplemented!()
        }

        fn backward<C: ComplexRoutine, T: Tensor<C>>(input: &T) -> T {
            unimplemented!()
        }
    }

    pub trait Optimizer {
        // needs this because the parameters inside the optimizer need to sync with the precision type of the data
        // this is equivalent to the module trait
        type Precision: ComplexRoutine;

        fn init(args: &InitArgs) -> Self;

        fn step<W: Tensor<Self::Precision>, B: Tensor<Self::Precision>>(
            &mut self,
            weights: &mut W,
            bias: &mut B,
            gradw: &W,
            gradb: &B,
        );
    }

    pub struct CSGD<CP: ComplexRoutine> {
        lr: CP,
    }

    impl<C: ComplexRoutine> Optimizer for CSGD<C> {
        type Precision = C;

        fn init(args: &InitArgs) -> Self {
            unimplemented!()
        }

        fn step<W: Tensor<Self::Precision>, B: Tensor<Self::Precision>>(
            &mut self,
            weights: &mut W,
            bias: &mut B,
            gradw: &W,
            gradb: &B,
        ) {
            unimplemented!()
        }
    }
}

mod dataset {}
