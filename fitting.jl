# uncomment these two lines for visual representation of training progress.
using ProgressBars
using Printf


function adam_update!(node::Node, t::Int64)
	# Hyperparameters as suggested by the creators of Adam.
	α   = 0.01
	β_1 = 0.99
	β_2 = 0.999
	ε   = 10^(-9)
	# Following the instructions given in the original Adam paper ...
	# ... update moments
	node.mt = β_1*node.mt + (1-β_1)*node.tgradval     
	node.vt = β_2*node.vt + (1-β_2)*abs2(node.tgradval)  
	# ... bias correction
	node.mhat = node.mt / (1-β_1^t) 
	node.vhat = node.vt / (1-β_2^t)
	# ... update parameters
	node.value -= α*node.mhat/(sqrt(node.vhat) + ε) 
	# Resetting gradients. They will be freshly accumulated  
	# for the next descent step.
	node.tgradval = 0.0 
end

function adamstep!(θ::Parameters, t::Int64)
# This methods simply loops through the whole set of parameter Nodes
# and performs an 'adam_update!' for each such Node.
        @inbounds for l in eachindex(θ.weights)
        	@inbounds for i in eachindex(θ.weights[l])
			adam_update!(θ.weights[l][i], t)
		end
		@inbounds for i in eachindex(θ.biases[l])
			adam_update!(θ.biases[l][i], t)
		end
	end
end

function fit!(records::RecordCollection, 
             res_samples::Vector{Float64},
	     bdr_samples::Vector{Float64},
             ;epochs = 1
             )  
	res_losses = Float64[]
	bdr_losses = Float64[]	
	# uncomment these two lines when using "ProgressBars"
	progress = ProgressBar(1:epochs)
	@inbounds for epoch in progress
	# @inbounds for epoch = 1:epochs
		res_loss = addgrad_resloss!(records, res_samples)
		bdr_loss = addgrad_bdrloss!(records, bdr_samples)
		push!(res_losses, res_loss)
		push!(bdr_losses, bdr_loss)
		adamstep!(records.ffrecord.θ, epoch)
		# uncomment when using "ProgressBars"
		 set_description(progress, string(@sprintf("losses. residual: %.5f, boundary: %.5f", res_loss, bdr_loss)))   
	end
	return res_losses, bdr_losses
end
