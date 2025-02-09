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
        type Prec: Precision;
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
use tensor::{DynTensor, Precision, ShapeSlice, Tensor};

// maybe a DynModule and a StaticModule
// each one has a precision type that implements Precision trait
// and also input and output tensors
// statics have the addition of two constants for the sizes

pub trait DynModule
where
    Self: Debug,
{
    type Prec: Precision;
    type Input: Tensor;
    type Output: Tensor;

    fn init(args: HashMap<String, String>) -> Self;

    // see if this is the best way
    // slices are enough for activations and also maybe here
    // you only need slice and a shape (or vec and shape)
    fn forward(&self, input: DynTensor<Self::Prec>) -> Self::Output;

    fn backward(&mut self, input: Self::Input, grad: &Self::Output) -> Self::Output;

    fn inpsp(&self) -> ShapeSlice {
        unimplemented!()
    }

    fn outsp(&self) -> ShapeSlice {
        unimplemented!()
    }
}

// useful for stuff that changes the shape of the input data
pub trait Module
where
    Self: Debug,
{
    type Input: Tensor;
    type Output: Tensor;

    fn init(args: HashMap<String, String>) -> Self;

    fn forward(&self, input: &Self::Input) -> Self::Output;

    fn backward(&mut self, input: Self::Input, grad: &Self::Output) -> Self::Output;

    fn inpsp(&self) -> ShapeSlice {
        unimplemented!()
    }

    fn outsp(&self) -> ShapeSlice {
        unimplemented!()
    }
}

// useful for stuff that does not change the input shape
// can be trained if needed, just does not change the shape
pub trait Activation
where
    Self: Debug,
{
}

pub fn connectivity<M1: Module, M2: Module>(_m1: M1, _m2: M2) -> bool {
    unimplemented!()
}

pub mod module {
    use crate::{tensor::Tensor, Module};
    use std::{collections::HashMap, marker::PhantomData};

    #[derive(Debug)]
    pub struct Linear<TW: Tensor, TB: Tensor, TI: Tensor, TO: Tensor> {
        _weights: TW,
        _bias: TB,
        _tensorin: PhantomData<TI>,
        _tensorout: PhantomData<TO>,
    }

    impl<TW: Tensor, TB: Tensor, TI: Tensor, TO: Tensor> Module for Linear<TW, TB, TI, TO> {
        type Input = TI;
        type Output = TO;

        fn init(_args: HashMap<String, String>) -> Self {
            Self {
                _weights: Tensor::new(Vec::new()),
                _bias: Tensor::new(Vec::new()),
                _tensorin: PhantomData,
                _tensorout: PhantomData,
            }
        }

        fn forward(&self, _input: &Self::Input) -> Self::Output {
            println!("Linear pass");
            Tensor::new(Vec::new())
        }

        fn backward(&mut self, _input: Self::Input, _grad: &Self::Output) -> Self::Output {
            unimplemented!()
        }
    }
}

pub mod activation {
    #[derive(Debug)]
    pub struct Tanh {}
}
