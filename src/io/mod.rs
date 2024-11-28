use crate::math::CPrec;

pub struct Input<const LEN: usize, const DIM: usize> {
    data: [CPrec; LEN],
    shape: [usize; DIM],
}

pub struct Output<const LEN: usize, const DIM: usize> {
    data: [CPrec; LEN],
    shape: [usize; DIM],
}
