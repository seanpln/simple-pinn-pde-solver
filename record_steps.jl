function one!(node1::Node, node2::Node, outtape::Vector{Node}) 
 # funid = 1   
 	push!(outtape, Node(tpos  = TapePosition(class=4, position=length(outtape)+1),
 	                    value = 1.0,
 	                    pdata = ParentData([node1.tpos, node2.tpos],
	                                       [0.0, 0.0],
	                                       [-1, -1]
	                                      )
 	     )             )        
	return last(outtape)
end

function pi_one!(node1::Node, node2::Node, outtape::Vector{Node})
# funid = 2
	push!(outtape, Node(tpos  = TapePosition(class=4, position=length(outtape)+1),
	      		    value = node1.value,
	                    pdata = ParentData([node1.tpos, node2.tpos],
	                                       [1.0, 0.0],
	                                       [1, -1]
	                                      ) 
	                   )
	     )
	return last(outtape)
end
      
function pi_two!(node1::Node, node2::Node, outtape::Vector{Node})
# funid = 3
	push!(outtape, Node(tpos  = TapePosition(class=4, position=length(outtape)+1),
	                    value = node2.value,
	                    pdata = ParentData([node1.tpos, node2.tpos],
	                                       [0.0, 1.0],
	                                       [-1, 1]
	                                      )
	                   )
	     )
	return last(outtape)
end

function add!(node1::Node, node2::Node, outtape::Vector{Node})
	push!(outtape, Node(tpos  = TapePosition(class=4, position=length(outtape)+1),
	                    value = node1.value + node2.value,
	                    pdata = ParentData([node1.tpos, node2.tpos],
	                                       [1.0, 1.0],
	                                       [1, 1]
	                                      )
	                   )
	     )
	return last(outtape)
end

function multiply!(node1::Node, node2::Node, outtape::Vector{Node})
	push!(outtape, Node(tpos = TapePosition(class=4, position=length(outtape)+1),
	                    value = node1.value * node2.value,
	                    pdata = ParentData([node1.tpos, node2.tpos],
	                                       [node2.value, node1.value],
	                                       [3, 2]
					       )
			    )
	     )
	return last(outtape)
end

function mvproduct!(mat::Matrix{Node}, vec::Vector{Node}, outtape::Vector{Node})
	
	nbr_cols = size(mat)[2]
	result   = Vector{Node}(undef, nbr_cols)
	
	@inbounds for i = 1:nbr_cols
		col = @views mat[:,i]
		tmp = multiply!(vec[1], col[1], outtape)
		@inbounds for j = 2:length(vec)
			multiply!(vec[j], col[j], outtape)
			tmp = add!(last(outtape), tmp, outtape)
		end
		result[i] = tmp
	end
	
	return result
end

function hdmsum!(vec1::Vector{Node}, vec2::Vector{Node}, outtape::Vector{Node})

	@inbounds for i in eachindex(vec1)
		vec1[i] = add!(vec1[i], vec2[i], outtape)
	end

	return vec1
end

function act!(node::Node, outtape::Vector{Node})
	push!(outtape, Node(tpos = TapePosition(class=4, position=length(outtape)+1),
	                    value = σ(node.value),
	                    pdata = ParentData([node.tpos],
	                                       [σ_p(node.value)],
	                                       [4]
					       )
			    )
	     )
	return last(outtape)
end

function act_p!(node::Node, outtape::Vector{Node})       
# funid = 4                      
	push!(outtape, Node(tpos = TapePosition(class=4, position=length(outtape)+1),
	                    value = σ_p(node.value),
	                    pdata = ParentData([node.tpos],
	                                       [σ_pp(node.value)],
	                                       [5]
					       )
			    )
	     )
	return last(outtape)
end

function act_pp!(node::Node, outtape::Vector{Node})   
# funid = 5                   
	push!(outtape, Node(tpos = TapePosition(class=4, position=length(outtape)+1),
	                    value = σ_pp(node.value),
	                    pdata = ParentData([node.tpos],
	                                       [σ_ppp(node.value)],
	                                       [6]
					       )
			    )
	     )
	return last(outtape)
end

function act_ppp!(node::Node, outtape::Vector{Node}) 
# funid = 6                         
	push!(outtape, Node(tpos = TapePosition(class=4, position=length(outtape)+1),
	                    value = σ_ppp(node.value),
	                    pdata = ParentData([node.tpos],
	                                       [0.0],
	                                       [-1]
					       )
			    )
	     )
	return last(outtape)
end


function hdmact!(vec::Vector{Node}, outtape::Vector{Node})
	@inbounds for i in eachindex(vec)
		vec[i] = act!(vec[i], outtape)
	end
	return vec
end

# Non-allocating Record-step methods (for updating a DAG)

function mvupdate!(result::Vector{Node}, # to be updated (result exists from previous swipe)
                   mat::Matrix{Node},   
                   vec::Vector{Node},    # used in mv product
                   outtape::Vector{Node},
                   pos::Int64            # position on outtape befor mv product is applied
                  )  
                  
	nbr_cols = size(mat)[2]

	@inbounds for i = 1:nbr_cols
		
		result[i].value = 0.0 # reset entry of v1
		col = @view mat[:,i]
		#col = mat[:,i]
		
		#pos += 1
		# "tmp"
		outtape[pos].value = vec[1].value * col[1].value	
		outtape[pos].pdata.lgrads[1] = col[1].value       # new!
		outtape[pos].pdata.lgrads[2] = vec[1].value       # new!
		tmp_pos = pos
		
		pos += 1
		
		@inbounds for j = 2:length(vec)
			
			outtape[pos].value = vec[j].value * col[j].value # update tape data
			outtape[pos].pdata.lgrads[1] = col[j].value       # new!
			outtape[pos].pdata.lgrads[2] = vec[j].value       # new!
			pos += 1
			
			# "last(outtape) + tmp"
			outtape[pos].value = outtape[pos-1].value + outtape[tmp_pos].value
			tmp_pos = pos
			pos += 1
		end
		
		result[i].value = outtape[tmp_pos].value
	end
	
   	return pos                
end

function biasupdate!(v_1::Vector{Node}, # to be updated 
                     v_2::Vector{Node},
		     outtape::Vector{Node},
		     pos::Int64
                    )
                    
	@inbounds for i in eachindex(v_1)
		v_1[i].value += v_2[i].value
		outtape[pos].value = v_1[i].value
		pos += 1
	end
	return pos
end

function actupdate!(vec::Vector{Node}, 
		    outtape::Vector{Node},
		    pos::Int64
                   )
	@inbounds for i in eachindex(vec)
		oldv     = vec[i].value
		vec[i].value = σ(vec[i].value)	   # update activation data

		outtape[pos].value      = vec[i].value  # update tape data
		outtape[pos].pdata.lgrads[1] = σ_p(oldv) # new!
		pos += 1
	end
	return pos
end
