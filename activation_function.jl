# Hyperbolic tangent
σ(z::Float64)     = tanh(z)  
σ_p(z::Float64)   = sech(z)^2
σ_pp(z::Float64)  = -2.0 * σ(z) * σ_p(z)            
σ_ppp(z::Float64) = 4.0  * σ(z)^2 * σ_p(z) - 2.0 * σ_p(z)^2

#= Sigmoid
σ(z::Float64)     = 1/(1+exp(-z))
σ_p(z::Float64)   = exp(-z)/(1+exp(-z))^2
σ_pp(z::Float64)  = 0.0            
σ_ppp(z::Float64) = 0.0
=#

#= Swish
sigmoid(z::Float64)    = 1/(1+exp(-z))
sigmoid_p(z::Float64)  = exp(-z)/(1+exp(-z))^2
sigmoid_pp(z::Float64) = (2*exp(-2*z))/(exp(-z) + 1)^3 - exp(-z)/(exp(-z) + 1)^2
σ(z::Float64)          = z*sigmoid(z)  
σ_p(z::Float64)        = sigmoid(z)+z*sigmoid(z)*(1-sigmoid(z))
σ_pp(z::Float64)       = sigmoid(z)*(1 - 2*z*sigmoid_p(z)) + (z+1)*sigmoid_p(z) - sigmoid(z)^2         
σ_ppp(z::Float64)      = (z*(1 - 2*sigmoid(z)) + 1)*sigmoid_pp(z) - 2*z*sigmoid_p(z)^2 + (2 - 4*sigmoid(z))*sigmoid_p(z)
=#
