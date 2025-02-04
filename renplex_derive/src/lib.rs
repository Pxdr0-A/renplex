use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, Fields, Ident, Type};

#[proc_macro_derive(SuperModuleMacro)]
pub fn super_module_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();

    impl_supermodule_macro(&ast)
}

fn impl_supermodule_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let field_idents = if let Data::Struct(data_struct) = &ast.data {
        match &data_struct.fields {
            Fields::Named(fields_named) => fields_named
                .named
                .iter()
                .filter_map(|field| field.ident.clone())
                .collect::<Vec<Ident>>(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let _field_tys = if let Data::Struct(data_struct) = &ast.data {
        match &data_struct.fields {
            Fields::Named(fields_named) => fields_named
                .named
                .iter()
                .map(|field| field.ty.clone())
                .collect::<Vec<Type>>(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    // pre-processing token streams for the implementation

    // field identifiers
    let fields = field_idents.iter().map(|field| {
        quote! { #field }
    });
    // init method arguments
    let init_args = field_idents.iter().map(|field| {
        let arg_name = field;
        let arg_type = quote! { HashMap<String, String> };

        quote! { #arg_name: #arg_type }
    });

    // implement functionalities
    let gen = quote! {
        impl #name {
            pub fn init(#(#init_args),*) -> Self {
                // if the static tensor sizes do not align, than it will probabily not compile (in forward)
                Self {
                    #(#fields: Module::init(#fields)),*
                }
                // when the network is initialized, check shape agreement between io's
                // how do you do this with activation functions
            }
        }
    };

    gen.into()
}
