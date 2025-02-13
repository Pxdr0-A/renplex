# Important Tasks

[x] - Activation Functions do not need specific Tensors.
[x] - See the shape slice thing (Potentially not worth it for now).
[x] - Maybe it is more straight-forward to define a DynModule and StaticModule (Potentially).
[ ] - Define connectivity function.
[ ] - Create Aliases for the types. Static and Dyn Linear for instance.
[ ] - Find a strategy to do the intercept.
    [ ] - One method for activating each module until the desired one.
[ ] - Propagating slices?
[x] - Activation fucntions as closures?! (No!)
[ ] - Define the input type of the network?
    [ ] - Is it really necessary to define input and output type of the network?
    [x] - This will influence the activation functions? (Not really. Only if the network is full with activation functions with nothing to infer type with)
