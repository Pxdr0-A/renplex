use super::CPrec;
use rand::rngs::ThreadRng;

pub type Shape = (usize, usize);

pub trait Matrix
where
    Self: Sized + Clone,
{
    // Base Implementations

    fn zeros(shape: Shape) -> Self;

    fn ones(shape: Shape) -> Self;

    fn ident(shape: Shape) -> Self;

    fn prandom(shape: Shape, thread: &mut ThreadRng) -> Self;

    fn get_shape(&self) -> Shape;

    fn get_body(&self) -> &[CPrec];

    fn get_mut_body(&mut self) -> &mut [CPrec];

    // Derived Implementations
}

#[derive(Debug, Clone)]
pub struct StaticMatrix<const LEN: usize> {
    array: [CPrec; LEN],
    shape: Shape,
}

impl<const LEN: usize> Matrix for StaticMatrix<LEN> {
    fn zeros(shape: Shape) -> Self {
        Self {
            array: [CPrec::ZERO; LEN],
            shape,
        }
    }

    fn ones(shape: Shape) -> Self {
        Self {
            array: [CPrec::ONE; LEN],
            shape,
        }
    }

    fn ident(shape: (usize, usize)) -> Self {
        let mut src = Self::zeros(shape);
        let srcsp = src.get_shape();
        let srcbd = src.get_mut_body();
        for i in 0..(srcsp.0) {
            srcbd[i * srcsp.1 + i] += CPrec::ONE;
        }

        src
    }

    fn prandom(_shape: Shape, _thread: &mut ThreadRng) -> Self {
        unimplemented!()
    }

    fn get_shape(&self) -> Shape {
        unimplemented!()
    }

    fn get_body(&self) -> &[CPrec] {
        unimplemented!()
    }

    fn get_mut_body(&mut self) -> &mut [CPrec] {
        unimplemented!()
    }
}

#[macro_export]
macro_rules! bin_combs {
    ($bits:expr) => {{
        const MAT_LEN: usize = 2_usize.pow($bits as u32);
        let mut combs = [char::default(); $bits * MAT_LEN];
        for bit_seq in 0..MAT_LEN {
            let binary_string = format!("{:0width$b}", bit_seq, width = $bits);
            let binary_chars = binary_string.chars().collect::<Vec<char>>();

            let init = bit_seq * $bits;
            let end = init + $bits;
            combs[init..end].copy_from_slice(binary_chars.as_slice());
        }

        combs
    }};
}
