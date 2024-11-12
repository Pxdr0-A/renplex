use std::ffi::c_int;

#[link(name="blas", kind="static")]
#[link(name="openblas", kind="static")]
#[link(name="lapack", kind="static")]
#[link(name="gfortran")]
extern "C" {
  fn dgemv_(
    trans: *const u8,        // Whether to transpose matrix A ('N' for No, 'T' for Transpose)
    m: *const c_int,         // Number of rows in A
    n: *const c_int,         // Number of columns in A
    alpha: *const f64,       // Scalar multiplier for A * x
    a: *const f64,           // Matrix A
    lda: *const c_int,       // Leading dimension of A
    x: *const f64,           // Vector x
    incx: *const c_int,      // Stride of x
    beta: *const f64,        // Scalar multiplier for y
    y: *mut f64,             // Vector y (output)
    incy: *const c_int,      // Stride of y
  );

  fn dgesv_(
    n: *const c_int,          // Order of matrix A
    nrhs: *const c_int,       // Number of right-hand sides
    a: *mut f64,              // Matrix A
    lda: *const c_int,        // Leading dimension of A
    ipiv: *mut c_int,         // Pivot indices
    b: *mut f64,              // Right-hand side matrix B (also solution matrix)
    ldb: *const c_int,        // Leading dimension of B
    info: *mut c_int,         // Output status code
  );
}