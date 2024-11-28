use super::CPrec;
use rand::rngs::ThreadRng;

pub type Shape = Vec<usize>;

pub trait Tensor
where
    Self: Sized + Clone,
{
    // Initializations

    fn zeros(shape: Shape) -> Self;

    fn ones(shape: Shape) -> Self;

    fn ident(shape: Shape) -> Self;

    fn prandom(shape: Shape, thread: &mut ThreadRng) -> Self;

    // Core Properties of the Tensor

    fn get_shape(&self) -> &[CPrec];

    fn get_body(&self) -> &[CPrec];

    fn get_mut_body(&mut self) -> &mut [CPrec];

    // Operations
}

#[derive(Debug, Clone)]
pub struct StaticTensor<const LEN: usize> {
    _array: [CPrec; LEN],
    _shape: Shape,
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
