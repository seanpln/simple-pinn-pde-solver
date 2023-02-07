function autodiff!(record::Record, n_s::Int64; scale = 1.0) 

	record.outtape[n_s].tgradval = scale
	@inbounds for i = n_s:-1:1
		node = record.outtape[i]
		backprop!(node, record)
		node.tgradval = 0.0
	end
end

function backprop!(node::Node, record::Record)

  	@inbounds for i in eachindex(node.pdata.tpos)
  	
  		localgrad = node.pdata.lgrads[i]
	
		if localgrad != 0.0
			contribution = node.tgradval * localgrad
			parent       = findnode(node.pdata.tpos[i], record)
			parent.tgradval    += contribution  
		end
	end
end
