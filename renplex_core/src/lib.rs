use std::{collections::HashMap, fmt::Debug};
// std and dependencies imports
// definition of internal mods

pub mod tensor {
    use num_complex::Complex;
    use std::fmt::Debug;

    // useful type
    pub type Shape = Vec<usize>;
    pub type ShapeSlice<'a> = &'a [usize];

    mod aocl {
        extern "C" {}
    }

    // native operations with scalar precision mapped to complex
    pub trait Precision
    where
        Self: Debug + Copy + Clone + Default,
    {
        // Relevant linear algebra routines
        // Taking the type, and wrapping it up in the complex number
        fn operation(lhs: &[Complex<Self>], rhs: &[Complex<Self>]);
    }

    impl Precision for f32 {
        fn operation(_lhs: &[Complex<Self>], _rhs: &[Complex<Self>]) {
            unimplemented!()
        }
    }

    impl Precision for f64 {
        fn operation(_lhs: &[Complex<Self>], _rhs: &[Complex<Self>]) {
            unimplemented!()
        }
    }

    // tensor definitions, operations, etc.
    pub trait Tensor
    where
        Self: Sized + Clone + Debug,
    {
        // probabily do not need precision here
        type Prec: Precision;
        const LEN: usize;

        // Initialization: just zeros.
        // Try to do further initializations after core props.
        fn new(shape: Shape) -> Self;

        // Core Properties of the Tensor
        // shape, body, etc.
        fn shape(&self) -> ShapeSlice;

        // Further Initializations based on zeros

        // Operations
        fn matmul(&self) {
            unimplemented!()
        }
    }

    #[derive(Debug, Clone)]
    pub struct DynTensor<T: Precision> {
        _array: Vec<Complex<T>>,
        _shape: Shape,
    }

    impl<T: Precision> Tensor for DynTensor<T> {
        type Prec = T;
        const LEN: usize = 0;

        fn new(_shape: Shape) -> Self {
            Self {
                _array: Vec::new(),
                _shape: Vec::new(),
            }
        }

        fn shape(&self) -> ShapeSlice {
            self._shape.as_slice()
        }
    }

    #[derive(Debug, Clone)]
    pub struct StaticTensor<T: Precision, const LEN: usize> {
        _array: [Complex<T>; LEN],
        _shape: Shape,
    }

    impl<T: Precision, const LEN: usize> Tensor for StaticTensor<T, LEN> {
        type Prec = T;
        const LEN: usize = LEN;

        fn new(_shape: Shape) -> Self {
            Self {
                _array: [Complex::default(); LEN],
                _shape: Vec::new(),
            }
        }

        fn shape(&self) -> ShapeSlice {
            self._shape.as_slice()
        }
    }
}

mod dataset {}

pub mod init {
    #[derive(Debug)]
    pub enum Intialization {
        None,
        Square,
    }
}

pub mod optimization {
    pub trait Loss {
        type Input;
        type Output;

        fn compute(input: Self::Input, target: Self::Output) -> Self::Output;

        fn derivative(input: Self::Input, target: Self::Output) -> Self::Output;
    }
}

// imports that come from internal modules
use tensor::{Precision, ShapeSlice, Tensor};

// maybe a DynModule and a StaticModule
// each one has a precision type that implements Precision trait
// and also input and output tensors
// statics have the addition of two constants for the sizes

pub trait Module {
    type Prec: Precision;
    type Input: Tensor;
    type Output: Tensor;

    fn init(args: HashMap<String, String>) -> Self;

    fn forward(&self, input: Self::Input) -> Self::Output;

    fn backward(&mut self, input: Self::Input, grad: Self::Output) -> Self::Input;

    fn inpsp(&self) -> ShapeSlice {
        unimplemented!()
    }

    fn outsp(&self) -> ShapeSlice {
        unimplemented!()
    }
}

// maybe Dyn and Static module are not a bad idea

pub mod modules {
    use crate::{
        tensor::{DynTensor, Precision, StaticTensor, Tensor},
        Module,
    };

    pub struct DynLinear<P: Precision> {
        _weights: DynTensor<P>,
        _bias: DynTensor<P>,
    }

    impl<P: Precision> Module for DynLinear<P> {
        type Prec = P;
        // template for a dynamic layer
        type Input = DynTensor<P>;
        type Output = DynTensor<P>;

        fn init(args: std::collections::HashMap<String, String>) -> Self {
            unimplemented!()
        }

        fn forward(&self, input: Self::Input) -> Self::Output {
            unimplemented!()
        }

        fn backward(&mut self, input: Self::Input, grad: Self::Output) -> Self::Input {
            unimplemented!()
        }
    }

    #[derive(Debug)]
    pub struct StaticLinear<
        P: Precision,
        const LENW: usize,
        const LENB: usize,
        const LENI: usize,
        const LENO: usize,
    > {
        _weights: StaticTensor<P, LENW>,
        _bias: StaticTensor<P, LENB>,
    }

    impl<
            P: Precision,
            const LENW: usize,
            const LENB: usize,
            const LENI: usize,
            const LENO: usize,
        > Module for StaticLinear<P, LENW, LENB, LENI, LENO>
    {
        type Prec = P;
        type Input = StaticTensor<P, LENI>;
        type Output = StaticTensor<P, LENO>;

        fn init(args: std::collections::HashMap<String, String>) -> Self {
            unimplemented!()
        }

        fn forward(&self, input: Self::Input) -> Self::Output {
            unimplemented!()
        }

        fn backward(
            &mut self,
            input: StaticTensor<Self::Prec, LENI>,
            grad: StaticTensor<Self::Prec, LENO>,
        ) -> StaticTensor<Self::Prec, LENI> {
            unimplemented!()
        }
    }
}

// stuff that does not change the input dims
// only has the problem if it is the first one on the supermodule
pub trait Activation
where
    Self: Debug,
{
    // this is a problem! it needs input and output type, otherwise, derive macro
    // cannot define input and output of the network
    fn init(args: HashMap<String, String>) -> Self;

    fn forward<T: Tensor>(&self, input: T) -> T;

    fn backward<T: Tensor>(&mut self, input: T, grad: T) -> T;

    fn inpsp(&self) -> ShapeSlice {
        unimplemented!()
    }

    fn outsp(&self) -> ShapeSlice {
        unimplemented!()
    }
}

pub mod activations {
    use crate::{tensor::Tensor, Activation};

    #[derive(Debug)]
    pub struct Tanh {}

    impl Activation for Tanh {
        fn init(_args: std::collections::HashMap<String, String>) -> Self {
            unimplemented!()
        }

        fn forward<T: Tensor>(&self, _input: T) -> T {
            unimplemented!()
        }

        fn backward<T: Tensor>(&mut self, _input: T, _grad: T) -> T {
            unimplemented!()
        }
    }
}
