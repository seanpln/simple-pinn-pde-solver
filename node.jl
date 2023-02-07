struct TapePosition
	class::Union{Nothing, Int8}
	layer::Union{Nothing, Int8}
	position::Union{Nothing, Int64}
	row::Union{Nothing, Int64}
	column::Union{Nothing, Int64}
	function TapePosition(;
	                      class    = nothing,
			      layer    = nothing,
			      position = nothing,
			      row      = nothing,
			      column   = nothing
			     )
		new(class, layer, position, row, column)
	end
end


struct ParentData
	# list of attributes
	tpos::Vector{TapePosition}
	lgrads::Vector{Float64}      # local gradients (used by 'autodiff')
	gradfuns::Vector{Int8}       # gradient function labels (used by 'adrecord')
end

mutable struct Node
	# list of attributes
	tpos::TapePosition           
	tgradpos::Int64               # total gradient's position  (tracked by 'adrecord')                  
	value::Float64               
        tgradval::Float64             # total gradient value (set by 'autodiff')	
	pdata::ParentData             # parent data
	# only for parameter nodes:
	mt::Union{Float64,Nothing}    # adam: first moment
	vt::Union{Float64,Nothing}    # adam: second moment
	mhat::Union{Float64,Nothing}  # adam: first moment: bias correction
	vhat::Union{Float64,Nothing}  # adam: second moment: bias correction
	# constructor method
	function Node(;
		      value    = 0.0,
		      tpos     = TapePosition(),
		      tgradpos = 0,
		      tgradval = 0.0,
		      pdata    = ParentData(TapePosition[], Float64[], Int8[]),      
		      mt       = nothing,
		      vt       = nothing,
	              mhat     = nothing,
	              vhat     = nothing            
	             )
		new(tpos, tgradpos, value, tgradval, pdata, mt, vt, mhat, vhat)
	end
end


# Methods for manipulating and reading Node attributes 

function class(node::Node)
	if node.tpos.class == 1
		return "root Node: input"
	elseif node.tpos.class == 2
		return "root Node: weight"
	elseif node.tpos.class == 3
		return "root Node: bias"
	elseif node.tpos.class == 3
		return "non-root Node"
	end
end

function setvalue!(node::Node, value::Float64)
# Sets a Node's value to the one specified by 'value'.
	node.value = value
end

function settotalgrad!(node::Node, value::Float64)
# Sets a Node's total gradient value to the one specified by 'value'.
	node.tgradval = value
end
