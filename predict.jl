function predict(X::Vector{Float64}, θ::Parameters)
	L = length(θ.weights)+1
	predictions = zeros(length(X))
	for i in eachindex(X)
		activated = [X[i]]
		for l = 1:(L-1)
			w_l = [w.value for w in θ.weights[l]]'
			b_l = [b.value for b in θ.biases[l]]
			activated = w_l * activated
			activated = activated .+ b_l
			if l < L-1
				activated = σ.(activated)
			end
		end
		predictions[i] = activated[1]
	end
	return predictions
end
