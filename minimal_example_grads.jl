function addgrad_resloss!(records::RecordCollection, samples::Vector{Float64})
	# Pick input Node.
	x    = records.activations[1][1]
	# Pick Node representing the required network derivative.
	Φxx  = records.adrecords[2].derivatives[1]
	# Initialize loss value to be reported.
	loss = 0.0
	# The Number of training samples will be required for proper scaling.
	N    = length(samples)
	# Loop over the traing samples
	@inbounds for xi in samples
		# Use current sample as DAG input.
		setvalue!(x, xi)
		# Update DAG with new input value.
		update_records!(records)
		# Use selected derivative Nodes to obtain the number Φ''(xi).
		ΔΦ = Φxx.value
		# Evaluate source term with current sample.
		f  = -20*sin(2*xi+1)
		# Add sample loss to total residual loss.
		loss += (1/N)*0.5*abs2(ΔΦ - f)
		# Add the sample gradient to the total gradient.
	        autodiff!(records.adrecords[2], Φxx.tpos.position, scale=(1/N)*(ΔΦ - f))
	end
	# Report residual loss for current set of parameters.
	return loss
end

# Similar for the boundary loss:
function addgrad_bdrloss!(records::RecordCollection, samples::Vector{Float64})
	x = records.activations[1][1]
	Φ = records.ffrecord.derivatives[1]
	N = length(samples)
	loss = 0.0
	@inbounds for xi in samples
		setvalue!(x, xi)
		update_ffrecord!(records.ffrecord, records.activations)
		g = (xi == 0.0 ? 5*sin(1) : 5*sin(3))
		loss += (1/N)*0.5*abs2(Φ.value - g)
	        autodiff!(records.ffrecord, Φ.tpos.position, scale=(1/N)*(Φ.value - g))
	end
	return loss
end
