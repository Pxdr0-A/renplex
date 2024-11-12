fn main() {
  // attempting to connect fortran statically
  // for now it is not working
  println!("cargo:rustc-link-search=native=/usr/lib/gcc/x86_64-redhat-linux/14");
}