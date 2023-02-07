# Define root Nodes ---------------------------------------------- #

include("node.jl") 		 
include("parameters.jl")

# Nodes for input neurons.
x = Node(value=0.0, tpos=TapePosition(class=1, position=1)) 
	     	 
# Nodes for network parameters.
# We will use a network with two 20-neuron hidden layers.
layers = [1,12,12,1]
θ      = Parameters(layers)

# Add non-root Nodes by tracking computations ------------------- #

include("activation_function.jl")    
include("record_steps.jl") 
include("record.jl") 


# Obtain record data from tracking the network's feedforward swipe.
ff_rec, activations = record_ff([x], θ) 

Φ = ff_rec.derivatives[1] 

ad_rec_1 = record_ad(ff_rec, Φ.tpos.position) 

Φx       = ad_rec_1.derivatives[1]
ad_rec_2 = record_ad(ad_rec_1, Φx.tpos.position)

Φxx = ad_rec_2.derivatives[1]


include("autodiff.jl") 
autodiff!(ad_rec_2, Φxx.tpos.position)


records = RecordCollection(ff_rec, activations, [ad_rec_1, ad_rec_2])

# model fitting ---------------------------------------------------- #

include("grads.jl")	# hand crafted per-sample loss gradient functions
include("fitting.jl")

# Sample training data.
res_samples = rand(collect(0.01:0.01:0.99), 50);
bdr_samples = [0.0, 1.0];

# Call the `fit!` function which deploys the Adam optimizer for successive
# parameter updates. It returns the reported residual and boundary losses.
res_losses, bdr_losses = fit!(records, res_samples, bdr_samples, epochs = 1000);

# uncomment to plot loss trajectory
# plot(collect(1:length(res_losses)), res_losses, label="residual loss")
# plot!(collect(1:length(bdr_losses), bdr_losses, label="boundary loss")


# model evaluation -------------------------------------------------- #

include("predict.jl")

# sample test data
X_test = collect(0:0.001:1.0)

# obtain the model's prediction
Y_pred = predict(X_test, θ)

# get true values from exact solution
u(x) = 5*sin(2*x+1)
Y_true = u.(X_test)

# compute MSE
mse =  (1/length(X_test))*sum(abs2.(Y_pred - Y_true))
println("MSE: ", mse)
