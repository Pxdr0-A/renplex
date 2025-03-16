use proc_macro::TokenStream;

#[proc_macro_derive(SuperModuleMacro)]
pub fn super_module_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();

    impl_supermodule_macro(&ast)
}

fn impl_supermodule_macro(ast: &syn::DeriveInput) -> TokenStream {
    unimplemented!()
}
