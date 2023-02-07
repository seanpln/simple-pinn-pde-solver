# simple-pinn-pde-solver

## 1. The PINN method

A PINN solver works by training a surrogate model $\Psi$ for the state $u:\overline{\Omega} \subset \mathbb{R}^d \to \mathbb{R}$ of a system

$$
\begin{cases}
\mathcal{F}[u] = f & \text{on } \Omega\\
\mathcal{B}[u] = g & \text{on } \partial{\Omega},\\
\end{cases}
$$

whose right hand side data $(f,g)$ is assumed to be sufficiently regular such that pointwise evaluation is well-defined. The model is of the form

$$\Psi_{\theta}(\boldsymbol{x}) = A(\boldsymbol{x}) + B(\boldsymbol{x})\Phi_{\theta}(\boldsymbol{x})$$

and parametrized by a set of weights and biases $\theta$ of a feedforwad neural network $\Phi_{\theta}$ (the "PINN"). One can either try to find fitting functions $A,B:\overline{\Omega} \to \mathbb{R}$ that achieve $\mathcal{B}[\Psi_\theta] = g$ and hence lead to a model that matches the boundary conditions exactly, or incorporate a boundary term in the loss function that determines the model's empirical risk $\hat{\mathcal{R}}$. In the latter case, a common choice is

$$ \hat{\mathcal{R}}[\theta] = \frac{\lambda_\text{res}}{N_\text{res}}\sum_{i=1}^{N_\text{res}}{\frac{1}{2}(\mathcal{F}\[\Phi_{\theta}\](\boldsymbol{x}_i)-f(\boldsymbol{x}_i))^2} +  \frac{\lambda_\text{bdr}}{N_\text{bdr}}\sum_{i=1}^{N_\text{bdr}}{\frac{1}{2}(\mathcal{B}\[\Phi_{\theta}\](\boldsymbol{x}_i)-g(\boldsymbol{x}_i))^2}$$

which is a sum of weighted MSEs over training samples. (Note that when we work with a boundary component, we set $A = 1$ and $B = 0$, i.e. $\Psi_{\theta} = \Phi_{\theta}$.)

The model fitting is gradient based and requires the expressions $\nabla_\theta \hat{ \mathcal{R} }$ in order to update parameters. These gradients can be computed with a sufficient efficiency by feeding the computational graph (DAG) that underlies $\hat{ \mathcal{R} }$ into a reverse-mode automatic differentiation (AD) algorithm.

## 2. Working example 

For a minimal working example, consider the following Poisson problem with Dirichlet boundary conditions

$$
\begin{cases}
u''(x) = -20\sin(2x+1), & x \in (0,1),\\
u(0) = 5\sin(1),\\
u(1) = 5\sin(3),\\
\end{cases}
$$

whose exact solution is given as $u(x) = 5\sin(2x+1)$.
 
### 2.1. Constructing the DAGs

We require

```julia
include("node.jl")
include("parameters.jl")
```
to define the root Nodes for a fully connected feedforward neural network.

```julia
# Nodes for input neurons.
x = Node(value=0.0, tpos=TapePosition(class=1, position=1)) 	     	 
# Nodes for network parameters.
# We will use a network with two 10-neuron hidden layers.
layers = [1,10,10,1]
θ      = Parameters(layers)
```

After defining the root Nodes, we add the non-root Nodes by recording all the operations that are required to obtain the expression $\Phi_\theta''(0.0)$. For this, we use the code from

```julia
include("activation_function.jl")
include("record_steps.jl")
include("record.jl")
```

The first "portion" of the DAG representing $\Phi_\theta''(0.0)$ comes from tracking the feedforward swipe of the neural network, i.e. the computation $\Phi_\theta(0.0)$.

```julia
ff_rec, activations = record_ff([x, y], θ) 
```

The output Nodes from any record will be contained in an attribute we call `derivatives`. In particular, the value $\Phi_\theta(0.0)$ is the "zeroth derivative" of the neural network with respect to its input $x$. Hence, we can grab the Node representing the value $\Phi_\theta(0.0)$ by

```julia  	        
Φ = ff_rec.derivatives[1] 
```

Next, we use an **AD-Recorder** to record the operations performed by a reverse-mode AD algorithm that operates on the DAG corresponding to the expression $\Phi_\theta(0.0)$ and that uses the Node `Φ`, which holds the output of the neural network, as the AD seed. Note that instead of passing the seed Node itself, we tell the AD-Recorder where to find it on the input DAG by means of its tape position.

```julia
ad_rec_1 = record_ad(ff_rec, Φ.tpos.position) 
```

As we want to obtain the DAG corresponding to $\Phi_\theta''(0.0)$, we prepare another AD-Record. This time, the automatic differentiator will run on the DAG corresponding to the expression $\Phi_\theta'(0.0)$, whose output Node `Φx` serves as the AD seed.  

```julia  	        
Φx       = ad_rec_1.derivatives[1]
ad_rec_2 = record_ad(ad_rec_1, getpos(Φx))
```

We remark that if we had used a neural network with, say, three inputs $x,y$ and $z$, we could grab, for instance, the Node representing the derivative $\partial_z\Phi_\theta(x,y,z)$ with this line of code:

```julia  	        
Φz = ad_rec_1.derivatives[3]
```

The parameter gradient $\nabla_\theta\Phi_\theta''(0.0)$ can now be obtained from 

```julia
include("autodiff.jl")
Φxx = ad_rec_2.derivatives[1]
autodiff!(ad_rec_2, Φxx.tpos.position)
```

Here, the `autodiff!` method writes the gradients of the seed Node `Φxx` with respect to every other Node's value in the DAG into the corresponding Node's `tgradval` attribute. For instance, to get the partial derivative 

$$ \frac{\partial}{\partial w^2_{1,2}}\Phi_\theta''(0.0)$$

with respect to the weight $w^3_{1,2}$ that connects neuron #1 of layer 3 with neuron #2 of the input layer, we can type

```julia
θ.weights[2][1,2].tgradval
```

Now that we have constructed the DAG, we bundle the complete set of record data into a `RecordCollection` to be conveniently accessible for the construction of the empirical risk gradients. 

```julia
records = RecordCollection(ff_rec, activations, [ad_rec_1, ad_rec_2])
```

### 2.2. Defining methods for the computation of the empirical risk gradients.

The gradients for both the residual and the boundary component of the empirical risk function is a sum of "per sample" gradients over the training data. For instance

$$ \hat{\mathcal{R}}_\text{res} = \sum_{i=1}^{N_\text{res}}{\nabla_\theta \frac{1}{2}(\Phi_\theta''(x_i) - f(x_i))^2} = \sum_{i=1}^{N_\text{res}}{(\Phi_\theta''(x_i) - f(x_i))\nabla_\theta}\Phi_\theta''(x_i)$$

According to what has already been written, this sum can be computed by looping over the training data, obtaining the DAGs $\Phi_\theta''(x_i)$ and feeding them into the `autodiff` method. The latter can be scaled in order to incorporate outer derivatives. The following code implements this for our minimal working example. It is important to note that instead of creating new DAGs each time, we overwrite existing ones to recycle memory (and hence avoid memory allocation).

```julia
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
	        autodiff!(records.adrecords[2], getpos(Φxx), scale=(1/N)*(ΔΦ - f))
	end
	# Report residual loss for current set of parameters.
	return loss
end
```

### 2.3. Fitting the model

We will feed the gradient data into the popular Adam optimizer[^1] to update the parameters. After the gradients have been fully accumulated in terms of the two methods `addgrad_resloss!` and `addgrad_resloss!`, the Adam algorithm performs the following update of the parameter values:

[^1]:  https://doi.org/10.48550/arXiv.1412.6980

```julia
function adam_update!(node::Node, t::Int64)
	# Hyperparameters as suggested by the creators of Adam.
	α   = 0.01
	β_1 = 0.99
	β_2 = 0.999
	ε   = 10^(-9)
	# Following the instructions given in the original Adam paper ...
	# ... update moments
	node.mt = β_1*nd.mt + (1-β_1)*node.tgradvalue	     
	node.vt = β_2*nd.vt + (1-β_2)*abs2(node.tgradvalue)  
	# ... bias correction
	node.mhat = nd.mt / (1-β_1^t) 
	node.vhat = nd.vt / (1-β_2^t)
	# ... update parameters
	node.value -= α*node.m̂hat/(sqrt(node.v̂hat) + ε) 
	# Resetting gradients. They will be freshly accumulated  
	# for the next descent step.
	node.tgradvalue = 0.0 
end
```

Finally, we can sample the training data and fit our model $\Phi_\theta$.

```julia
res_samples = collect(0.01:0.01:0.99)
bdr_samples = [0.0, 1.0]
fit!(records, res_samples, bdr_samples, 10000)
```

![losses](https://user-images.githubusercontent.com/124056760/217243091-9435be29-7c5d-4c62-80ae-76ec9262607d.png)


The latter works by repeatedly computing the gradients and using them by applying the `adam_update!` to each and every parameter.  

```julia
function fit!(records::RecordCollection, 
             res_samples::Vector{Float64},
	     bdr_samples::Vector{Float64},
             nbr_epochs::Int64
             )  
	res_losses = Float64[]
	bdr_losses = Float64[]	
	@inbounds for epoch = 1:nbr_epochs
		res_loss = addgrad_resloss!(records, res_samples)
		bdr_loss = addgrad_bdrloss!(records, bdr_samples)
		push!(res_losses, res_loss)
		push!(bdr_losses, bdr_loss)
		adamstep!(records.ffrecord.θ, epoch)
	end
	return res_losses, bdr_losses
end

function adamstep!(θ::ParameterNodes, t::Int64)
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
```

### 2.4. Model validation

![validation](https://user-images.githubusercontent.com/124056760/217243111-8ce5e1ed-cd58-4062-a13b-2b8ad7cfd35f.png)

## 3. Details about DAG construction 

### 3.1. Basic DAG construction

The core idea is shared with popular ML libraries such as Tensorflow or PyTorch and relies on recording the series of performed computations $f_1,\dots,f_n$ that result in a particular DAG onto a "tape" $\boldsymbol{T}$, where the relevant data for each intermediate result produced by a step $f_i$ will be stored in its own *Node* object $\boldsymbol{N}$. We will represent Nodes as (mutable) Julia structs. The most basic information hold by an instance `N` of the Node struct is the Node's position on the tape `N.tpos` and the numerical value `N.value` obtained from the computational step that led to the instanciation of the Node `N`.

```julia
mutable struct Node

	tpos::Int64                  # tape position                 
	value::Float64               # intermediate result of computational step	
	pdata::Vector{Int64}         # parent information

	function Node(;
		      tpos   = (0,0,0,0),
		      value  = 0.0
		      pdata  = Int64[]              
	             )
		new(tpos, value, pdata)
	end
end
```

The general scheme for DAG construction is then the following

1. Initialize a tape `t` that holds the root Nodes representing the computation inputs.
2. For each computational step $f_i$, execute a corresponding **Record-step method** that appends a Node, which holds the computational data related to $f_i$, to `t`. 

### Example.

For every $(x,y) \in \mathbb{R}^2$, the computation $f(x,y) = (x+y)^2$ can be executed stepwise by first computing $f_1(x,y) = x+y$ and then $f_2(f_1) = f_1^2$. We will use the scheme above to construct the DAG corresponding to the computation $f$ with inputs $x=1.0$ and $y=2.0$.

We first create the two input Nodes `x` and `y` with the desired values and initialize a tape `t` holding these Nodes.

```julia
x = Node(tpos=1, value=1.0)
y = Node(tpos=2, value=2.0)
t = [x, y]
```

The computational steps $f_1$ and $f_2$ will be recorded by means of corresponding Record-step methods.

```julia
function add!(node_1::Node, node_2::Node, tape::Vector{Node})
	push!(tape, Node(tpos  = length(tape)+1,
	                 value = node_1.value+node_2.value,
			 pdata = [node_1.tpos, node_2.tpos]		 
	                 )
             )
end

function square!(node::Node, tape::Vector{Node})
	push!(tape, Node(tpos  = length(tape)+1,
	                 value = abs2(node.value),
			 pdata = [node.tpos]			 
	                 )
             )
end
```

In order to record the computation $f(1.0,2.0)$, we consecutively pass the tape `t` through the Record-step methods. Afterwards, the state of the variable `t` will represent the DAG that underlies the function evaluation $f(1.0,2.0)$.

```julia
add!(x, y, t)
square!(last(t), t)
```

The more convenient approach is to define a method that bundles the record of these computations in advance.

```julia
function record_f(root_1::Node, root2::Node)
	tape = [root_1, root_2]
	add!(root_1, root_2, tape)
	square!(last(tape), tape)
	return tape
end	
```

Then we can record the computation $f(1.0,2.0)$ by means of 

```julia
x = Node(tpos=1, value=1.0)
y = Node(tpos=2, value=2.0)
t = record_f(x, y)
```

In the file `node.jl` we provide Record-methods to track the feedforward swipe of a multilayer perceptron $\Phi_\theta:\mathbb{R}^d\to\mathbb{R}$.


### 3.2. Reverse-mode AD

A reverse-mode automatic differentiator works by traversing the DAG in reversed topologically sorted order, backpropagating derivative data from the child Nodes to their parents. We use generic "textbook code" to implement this as

```julia
function autodiff!(tape::Vector{Node},		# DAG 
                  n_s::Int64			# index of 'seed Node'
		  ) 
	tape[n_s].tgrad = 1.0        		# set total gradient of the seed
	# traverse the DAG in reverse order
	@inbounds for i = n_s:-1:1              
		node = tape[i]			                 							
		backprop!(node)                 # backpropagate derivative data to the Node's parents
	end
end
```
The `backprop!` method reads the parent data from the current Node selected by the AD algorithm and sends the quantities

$$ \frac{d v_s}{d v_{\text{child}}}\frac{\partial v_{\text{child}}}{\partial v_{\text{parent}}}$$

*from* the child nodes *to* the parents.

```julia
function backprop!(node::Node)
  	@inbounds for i in eachindex(node.pdata)
			parent         = node.pdata.tpos[i]
			parent.tgrad  += node.tgrad * node.pdata.lgrads[i] 
	end
end
```
The code requires some straight forward modifications of the `Node` struct. First, we endow the Nodes with a **total gradient** attribute `tgrad` which corresponds to the quantity $d v_s/d v_i$. This attribute will be initialized with zero and is to be updated by the automatic differentiator. Second, the backpropagation requires the **local derivatives** $\partial v_i/\partial v_{\text{parent}}$ of a Node's value with respect to its parent Nodes values. We adapt our Node struct by changing the type of the Node attribute `pdata` from `Vector{Int64}` to `ParentData`, which is specified as

```julia
struct ParentData
	tpos::Vector{ID}		  # tape positions of parents
	lgrads::Vector{Float64}           # local gradients of parents (used by 'autodiff')
end
```

Here, the `tpos` attribute, which holds the tape positions $[p_{1},\dots,p_{r}]$ of the parents for some Node $\boldsymbol{N}$, is accompanied by the `lgrads` attribute, that lists the local gradients 

$$ [ \frac{\partial f}{\partial v_{p_1}}(v_{p_1},\dots,v_{p_r}),\dots, \frac{\partial f}{\partial v_{p_r}}(v_{p_1},\dots,v_{p_r})], $$

where $f$ is the computational step recorded by means of the Node $\boldsymbol{N}$. Note an entry `node.pdata.lgrads[j]` contains exactly the partial derivative with respect to the parent Node with tape position `node.pdata.tpos[j]`. We can convenienty create the local gradient data by means of the Record-step methods. For instance, for the bivariat multiplication we can write

```julia
function multiply!(node_1::Node, node_2::Node, tape::Vector{Node})
	push!(tape, Node(tpos  = length(tape)+1,
	                 value = node_1.value*node_2.value,
			 pdata = ParentData([node_1.tpos, node_2.tpos],
			                    [node_2.value, node_1.value]
			                   )		 
	                 )
             )
end
```

In case you have never heard of reverse-mode AD, we recall that this works due to the chain rule. To be more precise, let $\boldsymbol{N}_s$ be the selected output Node and $\boldsymbol{N}_i,\ i < s,$ a preceeding Node on the DAG. Then, according to the chain rule,

$$ \frac{d v_s}{d v_i} = S_i \overset{\text{def}}{=} \sum_{j=1}^{p} { \frac{d v_s}{d v_{n_j}} \frac{\partial v_{n_j}}{\partial v_i}  }$$

where $n_{1},\dots,n_{p}$ are the tape positions of the *children* of the Node $i$. Note that we ommit arguments for simplicity. By traversing the DAG, a reverse-mode automatic differentiator accumulates the sums $S_i$ term by term. It can be shown by induction that whenever the algorithm picks a Node $i$ in the loop over the DAG, then the total gradient of this Node $d v_s/d v_i$ must have already been fully accumulated, making the contributions to the sums $S_{p_1},\dots,S_{p_r}$ by means of backpropagation valid. For the induction's base case $(i=s)$ note that $d v_s/d v_s = 1$, which corresponds to the line `tape[n_s].tgrad = 1.0` in the code for the `autodiff!` method.


### 3.3. AD Recording

To obtain the DAG that corresponds to an expression of the form $D^{\boldsymbol{\alpha}}f(\boldsymbol{x})$, where $\boldsymbol{\alpha} \in \mathbb{N}^p$ is some multiindex, our idea is to repeatedly record the operations performed by the automatic differentiator itself. For instance, given some $\boldsymbol{x}\in\mathbb{R}^2$, say we want to construct the DAG that corresponds to the expression 

$$ D^{(2,1)}f(\boldsymbol{x}) = \partial_{xxy}f(\boldsymbol{x})$$

for some function $f:\mathbb{R}^2\to\mathbb{R}$. Then we can succeed as follows:

1. Construct the DAG of $f(\boldsymbol{x})$ in the form of a tape `t`.
2. Feed the tape `t` into a reverse-mode AD algorithm and record the operations it performs. This yields a tape `t_p` containing in particular the Nodes `n_x` and `n_y`, which represent the quantities $\partial_{x}f(\boldsymbol{x})$ and, respectively, $\partial_{y}f(\boldsymbol{x})$.
3. Feed the tape `t_p` into an automatic differentator with starting position `n_x.tpos` and again record its operations, which results in a tape `t_pp`. The tape `t_pp` will hold Nodes `n_xx` and `n_xy`, where `n_xx.value` and, respectively, `n_xy.value` equal $\partial_{xx}f(\boldsymbol{x})$ and $\partial_{xy}f(\boldsymbol{x})$, respectively.
6. Track the run of the automatic differentiator with input tape `t_pp` and starting position `n_xx.tpos` to obtain a tape `t_ppp`.
7. On the tape `t_ppp`, locate the Node whose value matches $D^{(2,1)}f(\boldsymbol{x})$.

In this section, we will discuss the implementation of this scheme.

Let $\boldsymbol{T}$ contain the record data from some computation $f(\boldsymbol{x})$ and let $\boldsymbol{T}'$ denote the tape that will hold the data generated by the AD-Recorder. For each derivative data transmission from a Node $\boldsymbol{N}$ to one of its parents $\boldsymbol{N}_{p_j}$ that is done by the automatic differentator, three operations have to be taped onto $\boldsymbol{T}'$ by the AD-Recorder:

1. Evaluating the expression $\partial f/\partial v_{p_j}(v_{p_1},\dots,v_{p_r})$ 
2. Multiplying the result of Step 1 with the total gradient value hold by the Node $\boldsymbol{N}$.
3. Adding the result of Step 2 to the total gradient value hold by the parent Node $\boldsymbol{N}_{p_j}$.

Note that although the expressions $\partial f/\partial v_{p_j}(v_{p_1},\dots,v_{p_r})$ are already evaluated when we perform the Record-step methods for DAG construction, these computations are *not* part of the computation represented by that DAG. The idea is to add function labels to the parent data, where these functions labels refer to a particular Record-Step method. When it sees such a label, the AD-Recorder can lookup, for instance, a dictionary to pick the corresponding function in order to perform the corresponding computational step itself, taping it to $\boldsymbol{T}'$. For our toy example, we will add an attribute `gradfun` which is of type `Vector{String}` to the `ParentData` struct, i.e.


```julia
struct ParentData
	tpos::Vector{Int64}		  # tape positions of parents
	lgrads::Vector{Float64}           # local gradient values of parents (used by 'autodiff')
	gradfuns::Vector{String}          # local gradient functions (executed by 'adrecord')
end
```

The `String` labels refer to and are set by Record-step methods. For example, the `square!` method, which records the computational step $f(x) = x^2$, will point to the Record-step method `times2!`, corresponding to $f$'s first derivative $f'(x) = 2x$. The latter will point to another method `const2`, implementing the record of $f''(x) = 2$. This chain ends when either a derivative will be zero or the required order of the derivative will be exceeded by the next function. We will indicate this with the label `"zero"`, which will tell the AD-Recorder to skip the function evaluation. (The latter is okay, since vanishing local gradients do not contribute to the sums $S_i$ and hence can be neglected for the DAG construction.)

```julia
# Record step method for the function f(x) = x^2
function square!(node::Node, tape::Vector{Node})
	push!(tape, Node(tpos  = length(tape)+1,
	                 value = abs2(node.value),
			 pdata = ParentData([node_1.tpos],
			                    [2*node.value],
					    ["times2"]
			                   )		 
	                 )
             )
end
# Record step method for the function f'(x) = 2x
function times2!(node::Node, tape::Vector{Node})
	push!(tape, Node(tpos  = length(tape)+1,
	                 value = 2.0*node.value,
			 pdata = ParentData([node.tpos],
			                    [2.0],
					    ["const2"]
			                   )		 
	                 )
             )
end
# Record step method for the function f''(x) = 2
function const2!(node::Node, tape::Vector{Node})
	push!(tape, Node(tpos  = length(tape)+1,
	                 value = 2.0,
			 pdata = ParentData([node.tpos],
			                    [0.0],
					    ["zero"]
			                   )		 
	                 )
             )
end
```

The AD-Recorder will use similar chains of Record-step methods to track the total gradient updates (Steps 2 and 3) performed by the AD algorithm. When a summand is added to a sum $S(p_j)$, three Nodes are involved. 

1. The Node $\dot{\boldsymbol{N}_i}$ that resulted from completing the sum $S_i$ and hence represents the total gradient of $\boldsymbol{N}_i$.  
2. The "local gradient Node" that is created when the AD-Recorder calls the function `n.gradfuns[j]`. This Node will be involved in the multiplication that yields the contribution to the sum $S(p_j)$. 
3. The Node $\dot{\boldsymbol{N}}_{p_j}$ that resulted from the most recent update of the sum $S(p_j)$. This Node will be passed to the `add!` operation together with the Node representing the total gradient contribution. 

```julia
function record_backprop!(node::Node, record::Record, outtape::Vector{Node})
	tg_node = outtape[node.tgradpos]
	# loop over parents of 'node'
	for j in eachindex(node.pdata.tpos)
		# tape position of j-th parent of 'node'
		p_j = node.pdata.tpos[j]
		# label of 'node.value' partial derivative w.r.t. j-th parent value
		g_j = node.pdata.gradfuns[j]    
		if g_j != "zero"
			# evaluate derivative function
			lg_node = record_lg_node!(g_j, node.pdata.tpos, tape, tape_p) 
			# j-th parent of 'node'
			p_node  = findnode(n_i, tape)
			# record computation of contribution and track parent's total gradient update
			contribute!(tg_node, lg_node, p_node, tape_p)          
		end
	end	
end
```

To keep track of where the Nodes that represent the most current states of the sums $S(i)$ sit on the tape $\boldsymbol{T}'$, we will add an attribute `tgradpos` to our `Node` struct. Whenever the `tgradval` (we formerly called this just `tgrad`) attribute of a Node $i$ is touched (and hence the sum $S(i)$ is updated), the attribute `tgradpos` will be set to the current length of the tape $\boldsymbol{T}'$.

```julia
function record_ad(tape:Vector{Node} n_s::Int64)
	resetpos!(tape)
	seed_tg_node  = Node(value=1.0, tpos=n_s)   
	tape_p        = vcat(intape[1:n_s-1], [ṡ])
	invalid       = Int64[] 
	seed          = tape[n_s]
	seed.tgradpos = length(tape_p)
	# 'playback' of the intape
	for i = n_s:-1:1	
		node = tape[i]
		# skip 'off path' Nodes
		if node.tgradpos != 0			
		        backward!(node)
		end
	end
	return tape_p	
end
```

### 3.4. Example

```julia
# define root Nodes representing the computation's inputs
x = Node(tpos=1, value=1.0)
y = Node(tpos=2, value=2.0)
# record the computation 
t = record_f([x, y])
# record the automatic differentiation operating on the tape 't'
t_p = record_ad(t, last(t))
# choose seed node for second AD record 
n_x = t_p[x.tgradpos]
# record the automatic differentiation operating on the tape 't_p' with seed 'n_x'
t_pp = record_ad(t_p, n_x)
# run automatic differentiator on second order tape `t_pp` to obtain third order derivatives
autodiff!(t_pp)
# read total gradient value from root Node
f_xxy = y.tgradvalue
```
