using Distributions

struct Parameters
        # ------------------
	# list of attributes
	weights::Vector{Matrix{Node}}
	biases::Vector{Vector{Node}}
	# ------------------
	# constructor method
	function Parameters(layers::Vector{Int64})
	# This method first creates all the weight and bias matrices and vectors as FLOAT64
	# arrays and then "copies" them as arrays of the "NODE" type.
	        # -----------------------------------------------
		# Sample values and store them in FLOAT64 arrays.
		w = Vector{Matrix{Float64}}(undef, length(layers)-1)
		b = Vector{Vector{Float64}}(undef, length(layers)-1)
		@inbounds for i = 1:length(layers)-1
			# Apply Glorot's method for parameter initialization ...
			# ... define distribution 
			fan_in  = layers[i]
			fan_out = layers[i+1]
			fan_avg = (fan_in + fan_out)/2
			var = 1.0 / fan_avg
			d   = Normal(0.0, sqrt(var))
			# ... sample values
			w[i] = rand(d,layers[i],layers[i+1]) 
			b[i] = rand(d,layers[i+1])
		end
		# --------------------------
		# Copy data into NODE arrays.
		w_nodes = Vector{Matrix{Node}}(undef, length(layers)-1)
		b_nodes = Vector{Vector{Node}}(undef, length(layers)-1)
		@inbounds for l = 1:length(layers)-1
		rows, cols = size(w[l])
		w_nodes[l] = similar(w[l], Node)
		b_nodes[l] = similar(b[l], Node)		
			@inbounds for j = 1:cols, i = 1:rows
				# Select sampled value
				weight = w[l][i,j]
                        	bias   = b[l][j]
                        	# Create corresponding Node representation 
				w_nodes[l][i,j] = Node(value = weight, 
				                       tpos  = TapePosition(class  = 2,
				                                            layer  = l,
				                                            row    = i,
				                                            column = j
				                                            ), 
				               	       mt   = 0.0, 
					               vt   = 0.0, 
				                       mhat = 0.0, 
				                       vhat = 0.0
				                      )
			        b_nodes[l][j]   = Node(value = bias, 
				                       tpos  = TapePosition(class = 3,
				                                            layer = l,
				                                            row   = j,
				                                           ), 
				               	       mt   = 0.0, 
					               vt   = 0.0, 
				                       mhat = 0.0, 
				                       vhat = 0.0
				                      )
			end
		end
		new(w_nodes,b_nodes)
	end	
end
