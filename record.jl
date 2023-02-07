struct Record
	# list of attributes
	inputs::Vector{Node}        # input neuron Nodes ('x','y',...)
	θ::Parameters	            # network parameter Nodes
	intape::Vector{Node}        # tape we feed into the AD algorithm
	outtape::Vector{Node}       # tape holding operations performed by the AD algorithm 
	valid::Vector{Int64}        # ?
	n_s::Int64                  # seed used by the AD algorithm
	derivatives::Vector{Node}   # Nodes representing derivatives of seed w.r.t. input neuron Nodes 
	# constructor
	function Record(inputs, θ, outtape; 
			intape      = [],
	                valid       = Int64[], 
	                n_s         = 0, 
	                derivatives = Node[]
	               )
		new(inputs, θ, intape, outtape, valid, n_s, derivatives) 
	end
end

function findnode(tpos::TapePosition, record::Record)
	if tpos.class == 4
		return record.outtape[tpos.position]
	elseif tpos.class == 1
		return record.inputs[tpos.position]
	elseif tpos.class == 2
		return record.θ.weights[tpos.layer][tpos.row,tpos.column]
	elseif tpos.class == 3
		return record.θ.biases[tpos.layer][tpos.row]
	end
end

struct RecordCollection
	# list of attributes
	ffrecord::Record
	activations::Vector{Vector{Node}}
	adrecords::Vector{Record}
end

function update_records!(records::RecordCollection)

	update_ffrecord!(records.ffrecord, records.activations)
	
	@inbounds for j in eachindex(records.adrecords)
		update_adrecord!(records.adrecords[j])
	end
end

function record_ff(inputs::Vector{Node}, θ::Parameters)

	# Preallocate memory for activation Node vectors
	activations = init_activations_vector(inputs, θ)
	# Initialize the tape for recording "zeroth order" AD.
	outtape = Node[]
	# Loop through layers and activate the l-th layer using the 
	# activations from the (l-1)-th layer.																		
	for l = 2:length(θ.weights)+1
		activations[l] = mvproduct!(θ.weights[l-1], activations[l-1], outtape)
		activations[l] = hdmsum!(activations[l], θ.biases[l-1], outtape)
		# Apply non-linear activation only for hidden layers.
		if l < length(θ.weights)+1
			activations[l] = hdmact!(activations[l], outtape) 
		end              
	end    
			                    
	record = Record(inputs, θ, outtape, derivatives = [last(outtape)])            
                                                                                       
	return record, activations		
end


function init_activations_vector(inputs::Vector{Node}, θ::Parameters)
	
	activations = Vector{Node}[]      
	push!(activations, Vector{Node}(undef, 2))
	for l = 2:length(θ.weights)+1   
		dim = length(θ.biases[l-1])
		push!(activations, Vector{Node}(undef, dim))                           
	end 
	activations[1] = inputs
	return activations

end


function record_ad(record::Record, n_s::Int64)
	# Reset TGRADPOS attributes of the Nodes from the previous record.
	resetpos!(record)
	# To increase the order of differentiation, we want to track the 
	# operations of the AD algorithm running on the outtape from 
	# the previous AD-Record. 
	intape = record.outtape
	# Prepate the new OUTTAPE, where the current AD-Recorder will record
	# what the AD algorithm does on INTAPE. 
	seed_tgnode = Node(value=1.0, tpos=TapePosition(class=4, position=n_s))   
	outtape      = vcat(intape[1:n_s-1], [seed_tgnode])
	# The AD algorithm may hit Nodes from the previous computation 
	# that did not contribute to the seed Node. The AD-Recorder will
	# keep track of the indices of these Nodes.
	invalid = Int64[] 
	# Tell the seed Node, where its total gradient Node is located.
	seed          = intape[n_s]
	seed.tgradpos = length(outtape)
        # Record the operations of the AD algorithm, whose inputs are N_S and INTAPE.
	for n = n_s:-1:1	
		node = intape[n]
		# If a Node did not appear as a parent of another Node yet, 
		# that Node is not part of the DAG that underlies the computation 
		# of the seed Node.
		if unseen(node)			
			push!(invalid, n)		
		else
		        record_backprop!(node, record, outtape)
		end
	end
	# Remove tape positions of invalid Nodes from valid tape position.
	valid = setdiff(collect(n_s:-1:1), invalid)
	# Collect Nodes that represent the final total gradient updates
	# for the non-parameter root Nodes.
	derivatives = [outtape[input.tgradpos] for input in record.inputs]
	
	return Record(record.inputs, record.θ, outtape, intape = intape, valid = valid, n_s = n_s, derivatives = derivatives)	
end

function record_backprop!(node::Node, record::Record, outtape::Vector{Node})
	tgnode = outtape[node.tgradpos]
	for j in eachindex(node.pdata.tpos)
		p_j = node.pdata.tpos[j]
		g_j = node.pdata.gradfuns[j]
		# Check if there is a function to evaluate.
		if g_j > 0
			# Create local gradient Node, i.e. track the evaluation of the
			# local gradient function.
			lgnode = record_lgnode!(g_j, node.pdata.tpos, record, outtape)
			# Locate the parent Node.
			pnode  = findnode(p_j, record)
			# Track the parent Node's total gradient update.
			contribute!(tgnode, lgnode, pnode, outtape)   
		end
	end	
end

function record_lgnode!(g_i::Int8, args::Vector{TapePosition}, record::Record, outtape::Vector{Node})    
   	if g_i == 1
   		node1   = findnode(args[1], record)
   		node2   = findnode(args[2], record)
		lg_node = one!(node1, node2, outtape) 
	elseif g_i == 2
   		node1   = findnode(args[1], record)
   		node2   = findnode(args[2], record)
		lg_node = pi_one!(node1, node2, outtape) 
	elseif g_i == 3
   		node1   = findnode(args[1], record)
   		node2   = findnode(args[2], record)
		lg_node = pi_two!(node1, node2, outtape) 
	elseif g_i == 4
		node    = findnode(args[1], record)
		lg_node = act_p!(node, outtape)
	elseif g_i == 5
		node    = findnode(args[1], record)
		lg_node = act_pp!(node, outtape)
	elseif g_i == 6
		node    = findnode(args[1], record)
		lg_node = act_ppp!(node, outtape)
	end
end 

function contribute!(tgnode::Node, lgnode::Node, pnode::Node, outtape::Vector{Node})
	multiply!(tgnode, lgnode, outtape)
	if !unseen(pnode)
		add!(last(outtape), outtape[pnode.tgradpos], outtape)  
	end	
	# Tell the parent Node where its most recent tota gradient Node is located.
	pnode.tgradpos = length(outtape)
end







function update_ffrecord!(record::Record, activations::Vector{Vector{Node}})     
         
        
        L = length(activations)
                                                                         
   	pos = 1   
	@inbounds for l = 2:L-1
		pos = mvupdate!(activations[l], 
		                record.θ.weights[l-1], 
		                activations[l-1], 
		                record.outtape, 
		                pos
		               )   
		pos = biasupdate!(activations[l], 
		                  record.θ.biases[l-1], 
		                  record.outtape, 
		                  pos
		                 )     
		pos = actupdate!(activations[l], 
			         record.outtape, 
			         pos
			        )

	end
	pos = mvupdate!(activations[L], 
		        record.θ.weights[L-1], 
		        activations[L-1], 
		        record.outtape, 
		        pos
		       )   
	pos = biasupdate!(activations[L], 
		          record.θ.biases[L-1], 
		          record.outtape, 
		          pos
		         )   
	return nothing
end


function update_adrecord!(record::Record) 


	resetpos!(record)
	
	pos = record.n_s
	record.intape[record.n_s].tgradpos = pos
	
	@inbounds for i in record.valid
		node = record.intape[i]
		pos  = overwrite_backward!(node, record, pos)
	end 
	
	return nothing
end

function overwrite_backward!(node::Node, record::Record, pos::Int64)
	tg_node = record.outtape[node.tgradpos]     
  	@inbounds for i in eachindex(node.pdata.tpos)
		n_i = node.pdata.tpos[i]
		g_i = node.pdata.gradfuns[i]
		if g_i > 0
			pos     = overwrite_lg_node!(g_i, node.pdata.tpos, record, pos)
			lg_node = record.outtape[pos]
			parent  = findnode(n_i, record)
			pos     = overwrite_contribution!(tg_node.value, 
			                                  lg_node.value, 
			                                  parent, 
			                                  record.outtape, 
			                                  pos
			                                 )
		end
	end
	return pos
end

function overwrite_lg_node!(g_i::Int8, args::Vector{TapePosition}, record::Record, pos::Int64)         
  	pos += 1
	if g_i == 1				
		record.outtape[pos].value = 1.0
	elseif g_i == 2
		node1 = findnode(args[1], record)	
		record.outtape[pos].value = node1.value
	elseif g_i == 3     
		node2 = findnode(args[2], record)	
		record.outtape[pos].value = node2.value	          
	elseif g_i == 4
		node = findnode(args[1], record)
		record.outtape[pos].value           = σ_p(node.value) 	
		record.outtape[pos].pdata.lgrads[1] = σ_pp(node.value)
	elseif g_i == 5
		node = findnode(args[1], record)
		record.outtape[pos].value           = σ_pp(node.value) 
		record.outtape[pos].pdata.lgrads[1] = σ_ppp(node.value)		
	end
	return pos
end 

function overwrite_contribution!(totalgrad::Float64,
                                 localgrad::Float64,
	                         parent::Node,
	                         outtape::Vector{Node},
	                         pos::Int64
	                        ) 
	pos += 1
	contribution = totalgrad * localgrad   
	outtape[pos].value      = contribution
	outtape[pos].pdata.lgrads[1] = localgrad 
	outtape[pos].pdata.lgrads[2] = totalgrad   
	if !unseen(parent)
		pos += 1
		outtape[pos].value = outtape[parent.tgradpos].value + contribution
	end
	parent.tgradpos = pos	
	return pos
end

function resetpos!(record::Record)
	@inbounds for i in eachindex(record.inputs)
		record.inputs[i].tgradpos = 0
	end
	
	@inbounds for i in eachindex(record.θ.weights)
		resetpos!(record.θ.weights[i])
		resetpos!(record.θ.biases[i])
	end
	
	@inbounds for i in eachindex(record.intape)   # need both in and outtape ?
			record.intape[i].tgradpos = 0
	end
	
	@inbounds for i in eachindex(record.outtape)
			record.outtape[i].tgradpos = 0
	end
end

function resetpos!(w::Matrix{Node})
	@inbounds for i in eachindex(w)
		w[i].tgradpos = 0
	end
end

function resetpos!(b::Vector{Node})
	@inbounds for i in eachindex(b)
		b[i].tgradpos = 0
	end
end

function unseen(node::Node)
	return node.tgradpos == 0
end
