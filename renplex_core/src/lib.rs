// std and dependencies imports
use std::{collections::HashMap, fmt::Debug};

// definition of internal mods

pub mod tensor {
    use num_complex::Complex;
    use std::fmt::Debug;

    mod aocl {
        extern "C" {}
    }

    mod mkl {
        extern "C" {}
    }

    // useful type
    pub type Shape = Vec<usize>;

    // native operations with scalar precision maped to complex
    pub trait Precision
    where
        Self: Debug + Copy + Clone + Default,
    {
        // Relevant linear algebra routines
        // Taking the type, and wrapping it up in the complex number
        fn operation(lhs: &[Complex<Self>], rhs: &[Complex<Self>]);
    }

    // impl Precision for f16 {}

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

    // impl Precision for f128 {}

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
use tensor::Tensor;

// this is only useful when you want to go from dyn to static and vice-versa
pub trait Module
where
    Self: Debug,
{
    type Input: Tensor;
    type Output: Tensor;

    fn init(args: HashMap<String, String>) -> Self;

    fn forward(&self, input: &Self::Input) -> Self::Output;

    fn backward(&self, input: Self::Input) -> Self::Output;

    fn update(&mut self, input: &Self::Input, grad: &Self::Output);
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

        fn backward(&self, _input: Self::Input) -> Self::Output {
            unimplemented!()
        }

        fn update(&mut self, _input: &Self::Input, _grad: &Self::Output) {
            unimplemented!()
        }
    }
}

pub mod activation {
    use std::{collections::HashMap, marker::PhantomData};

    use crate::{tensor::Tensor, Module};

    #[derive(Debug)]
    pub struct Tanh<T: Tensor> {
        _tensorio: PhantomData<T>,
    }

    impl<T: Tensor> Module for Tanh<T> {
        type Input = T;
        type Output = T;

        fn init(_args: HashMap<String, String>) -> Self {
            Self {
                _tensorio: PhantomData,
            }
        }

        fn forward(&self, _input: &Self::Input) -> Self::Output {
            println!("Tanh Activation");
            Tensor::new(Vec::new())
        }

        fn backward(&self, _input: Self::Input) -> Self::Output {
            unimplemented!()
        }

        fn update(&mut self, _input: &Self::Input, _grad: &Self::Output) {
            unimplemented!()
        }
    }
}
