fn main() {
    // attempting to connect fortran statically
    // for now it is not working
    println!("cargo:rustc-link-search=native=/opt/AMD/aocl/aocl-linux-gcc-5.0.0/gcc/lib");
    // println!("cargo:rustc-link-search=native=/opt/intel/oneapi/mkl/2025.0/lib")
}
