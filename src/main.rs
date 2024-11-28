use std::ffi::c_int;

#[link(name = "blis", kind = "static")]
extern "C" {
    fn dgemv_(
        trans: *const u8,   // Whether to transpose matrix A ('N' for No, 'T' for Transpose)
        m: *const c_int,    // Number of rows in A
        n: *const c_int,    // Number of columns in A
        alpha: *const f64,  // Scalar multiplier for A * x
        a: *const f64,      // Matrix A
        lda: *const c_int,  // Leading dimension of A
        x: *const f64,      // Vector x
        incx: *const c_int, // Stride of x
        beta: *const f64,   // Scalar multiplier for y
        y: *mut f64,        // Vector y (output)
        incy: *const c_int, // Stride of y
    );
}

fn main() {
    // Matrix A (2x2)
    let a = vec![5.0, 2.0, 3.0, 4.0];

    // Vector x (2 elements)
    let x = vec![1.0, 1.0];

    // Vector y (initially zeros, will contain the result)
    let mut y = vec![0.0, 0.0];

    // Parameters for dgemv
    let trans = b'N'; // No transpose ('N')
    let m = 2; // Number of rows in A
    let n = 2; // Number of columns in A
    let lda = m; // Leading dimension of A (set to number of rows of A)
    let incx = 1; // Stride of x
    let incy = 1; // Stride of y
    let alpha = 1.0; // Scalar multiplier for A * x
    let beta = 0.0; // Scalar multiplier for y (0 means y starts as 0)

    // Convert values to pointers for FFI
    let trans_ptr = &trans as *const u8;
    let m_ptr = &m as *const c_int;
    let n_ptr = &n as *const c_int;
    let alpha_ptr = &alpha as *const f64;
    let a_ptr = a.as_ptr();
    let lda_ptr = &lda as *const c_int;
    let x_ptr = x.as_ptr();
    let incx_ptr = &incx as *const c_int;
    let beta_ptr = &beta as *const f64;
    let y_ptr = y.as_mut_ptr();
    let incy_ptr = &incy as *const c_int;

    // Unsafe block to call the external function
    unsafe {
        dgemv_(
            trans_ptr, m_ptr, n_ptr, alpha_ptr, a_ptr, lda_ptr, x_ptr, incx_ptr, beta_ptr, y_ptr,
            incy_ptr,
        );
    }

    // Print the resulting vector y
    println!("Resulting vector y: {:?}", y);
}
