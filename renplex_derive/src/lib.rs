use proc_macro::TokenStream;
use syn::{Data, Fields};

#[proc_macro_derive(ImplNetwork)]
pub fn super_module_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();

    impl_supermodule_macro(&ast)
}

fn impl_supermodule_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;

    // Get the struct data
    let data = match &ast.data {
        Data::Struct(data) => data,
        _ => panic!("GetLastFieldType can only be used on structs"),
    };

    let fields = match data.fields {
        Fields::Named(ref fields) => &fields.named,
        Fields::Unnamed(ref fields) => &fields.unnamed,
        Fields::Unit => panic!("GetLastFieldType cannot be used on unit structs"),
    };

    // Get the type of the last field
    let last_field_type = fields
        .iter()
        .last()
        .map(|field| &field.ty)
        .expect("Struct has no fields");

    let gen = quote::quote! {
        impl Network for #name {
            type Precision = Precision;
            type Output = <#last_field_type as Module>::Output;
            type LossFunc = MyLossFunc;
            type Opt = MyOptimizer;

            fn init(args: &InitArgs) -> Self {
                unimplemented!()
            }

            fn predict<T: Tensor<Self::Precision>>(&self, input: &T) -> Self::Output {
                unimplemented!()
            }

            fn train<T: Tensor<Self::Precision>>(
                &mut self,
                input: &T,
                target: &Self::Output,
                opt: &mut Self::Opt,
            ) {
                unimplemented!()
            }
        }
    };

    gen.into()
}
