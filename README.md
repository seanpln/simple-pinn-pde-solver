# simple-pinn-pde-solver

## The PINN method

A PINN solver works by training a surrogate model $\Psi$ for the state $u:\overline{\Omega} \subset \mathbb{R}^d \to \mathbb{R}$ of a system

$$
\begin{cases}
\mathcal{F}[u] = f & \text{on } \Omega\\
\mathcal{B}[u] = g & \text{on } \partial{\Omega},\\
\end{cases}
$$

whose right hand side data $(f,g)$ is assumed to be sufficiently regular such that pointwise evaluation is well-defined. The model is of the form

$$\Psi_{\theta}(\boldsymbol{x}) = A(\boldsymbol{x}) + B(\boldsymbol{x})\Phi_{\theta}(\boldsymbol{x})$$

and parametrized by a set of weights and biases $\theta$ of a feedforwad neural network $\Phi_{\theta}$ (the "PINN"). One can either try to find fitting functions $A,B:\overline{\Omega} \to \mathbb{R}$ that achieve $\mathcal{B}[\Psi_\theta] = g$ and hence lead to a model that matches the boundary conditions exactly, or incorporate a boundary term such as in the loss function that determines the model's empirical risk. (In the latter case one sets $A \equiv 0$ and $B \equiv 1$.) 

The model training is usually gradient based, which requires the expressions $\nabla_\theta \hat{ \mathcal{R} }$ in order to update parameters. These gradients can be computed with a sufficient efficiency by feeding the computational graph (DAG) that underlies $\hat{ \mathcal{R} }$ into a reverse-mode automatic differentiator. 

### Example. 

Consider the Poisson problem

$$
\begin{cases}
\mathcal{F}[u] = f & \text{on } \Omega\\
\mathcal{B}[u] = g & \text{on } \partial{\Omega},\\
\end{cases}
$$

blablabla



Thus, when the AD algortihm reaches a Node, it has to know this Nodes parents, as well as the local gradient data.

## Basic DAG construction

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
	                 value = abs2(node.tpos),
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

In the file `node.jl` we provide Record-methods to track the feedforward swipe of a multilayer perceptron $\Phi_\theta:\mathbb{R}^d\to\mathbb{R}$.

## AD

A reverse-mode automatic differentiator works by traversing the DAG in reversed topologically sorted order and backpropagating derivative data from the child Nodes to their parents.

```julia
function autodiff!(tape::Vector{Node},		# DAG 
                  n_s::Int64			# index of 'seed Node'
		  ) 
	tape[n_s].tgrad = 1.0        		# set total gradient of the seed
	# traverse the DAG
	@inbounds for i = n_s:-1:1              
		node = tape[i]			# grab Node from tape                  							
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
The code requires some straight forward modifications of the `Node` struct. First, we endow the Nodes with a **total gradient** attribute `tgrad` which corresponds to the quantity $\frac{d v_s}{d v_i}$. This attribute will be initialized with zero and will be updated by the automatic differentiator. Second, the backpropagation requires the **local derivatives** $\frac{\partial v_{\text{child}}}{\partial v_{\text{parent}}}$ of a Node's value with respect to its parent Nodes values. We will bundle this with the parents tape positions in a struct

```julia
struct ParentData
	tpos::Vector{ID}		  # tape positions of parents
	lgrads::Vector{Float64}           # local gradients of parents (used by 'autodiff')
end
```
Note that for some Node $N_i$ that results from some computational step $f$ and that has parents $N_{p_1},\dots,N_{p_r}$, the local gradients are given by the partial derivatives

$$ \frac{\partial f}{\partial v_{p_j}}(v_{p_1},\dots,v_{p_r}) $$

that can be hardcoded into the Record-step methods. For instance, we can track a bivariat multiplication with the Record-step method

```julia
function multiply!(node_1::Node, node_2::Node, tape::Vector{Node})
	push!(tape, Node(tpos  = length(tape)+1,
	                 value = node_1.value+node_2.value,
			 pdata = ParentData([node_1.tpos, node_2.tpos],
			                    [node_2.value, node_1.value]
			                   )		 
	                 )
             )
end
```


Note that if the Node $\boldsymbol{N}{i}$ has parents results from the computational step $f$ and has parents $\boldsymbol{N}_{p_1},dots,\boldsymbol{N}_{p_r}$

The **local gradients** $\frac{\partial v_{\text{child}}}{\partial v_{\text{parent}}}$ are the partial derivatives of the computational step
$f_i$ that


To be more precise, let $\boldsymbol{N}_s$ be the selected output Node and $\boldsymbol{N}_i,\ i < s,$ a preceeding Node on the DAG $\boldsymbol{T}$. Then, according to the chain rule,

$$ \frac{d v_s}{d v_i} = S_i \overset{\text{def}}{=} \sum_{j=1}^{p} { \frac{d v_s}{d v_{n_j}} \frac{\partial v_{n_j}}{\partial v_i}  }$$

where $n_{1},\dots,n_{p}$ are the tape positions of the children of the Node $\boldsymbol{N}_i$ and $v_m$ denotes the value hold by the Node with tape position $m$. Note that we ommit arguments for simplicity. An automatic differentiator works by traversing a DAG in reversed topologically sorted order and "sending" the quantities

$$ \frac{d v_s}{d v_{\text{child}}}\frac{\partial v_{\text{child}}}{\partial v_{\text{parent}}}$$

*from* the child nodes *to* the parents. This way, the sums $S_i$ are accumulated term by term.

```julia
function autodiff!(record::Record, n_s::Int64; scale = 1.0) 
	record.outtape[n_s].v̇ = scale
	@inbounds for i = n_s:-1:1
		node = record.outtape[i]
		backprop!(node, record)
		node.v̇ = 0.0
	end
end

function backprop!(node::Node, record::Record)

  	@inbounds for i in eachindex(node.p.n)
  	
  		localgrad = node.p.x[i]
	
		if localgrad != 0.0
			contribution = node.v̇ * localgrad
			parent       = findnode(node.p.n[i], record)
			parent.v̇    += contribution  
		end
	end
end
```




By construction, a tape is a topologically sorted list of the DAG vertices. Hence the DAG can be traversed by a reverse-mode AD algorithm simply by looping over $\boldsymbol{T} = [\boldsymbol{N}_1,\dots,\boldsymbol{N}_M]$ in reversed order. Such an algorithm gets passed a Node $\boldsymbol{N}_s\ (s \leq M)$ and outputs all the (total) derivatives $\dot{v}_i \overset{\text{def}}{=} dv_s/dv_i,\ (i = 1,\dots,M)$, where $v_i$ is the value hold by Node $\boldsymbol{N}_i$. To achieve this, the algorithm exploits the chain rule to accumulate new total gradients from already computed ones by passing the quantities 

$$ \dot{v}_i\frac{\partial f_i}{\partial v_{n_r}}(v_{n_1},\dots,v_{n_r})$$

to the $r$ parent Nodes $\boldsymbol{N}_{n_1},\dots$ of $\boldsymbol{N}_i$. To allow the AD algorithm the graph traversation, we bundle parent information about a Node in the struct

```julia
struct ParentData
	tpos::Vector{ID}		  # tape positions of parents
	lgrads::Vector{Float64}           # local gradients of parents (used by 'autodiff')
end
```

All together, this describes the basic layout of a Node object:

```julia
mutable struct Node

	tpos::Int64                  # tape position                 
	value::Float64               # intermediate result of computational step
        tgrad::Float64               # total gradient (overwritten by 'autodiff')	
	pdata::ParentData            # parent information

	function Node(;
		      tpos   = (0,0,0,0),
		      value  = 0.0
		      tgrad  = 0.0
		      pdata  = ParentData(Int64[], Float64[])              
	             )
		new(tpos, value, tgrad, pdata)
	end
end
```



We can then pass our tape `t` to a reverse-mode automatic differentiator in order to obtain the derivatives $\partial_xf(1,2)$ and $\partial_yf(1,2)$.

***

For the "Record-step" functions that are required for our PINN solver, see the file `node.jl`. Note that as soons as the automatic differentiator reaches one of the root-Nodes, there is nothing left to do for the algorithm. Thus, we can split a tape into $\boldsymbol{T}_0$, that holds only the root-Nodes, and $\boldsymbol{T}_c$ which holds all the Nodes that resulted from computations. You will find that the `Node struct` in the file `node.jl` makes use of a four-tuple to identify Nodes. This is because in the proper code we will split the tape even further apart. Namely, $\boldsymbol{T}_0$ will be split into input Nodes 

## AD Recording

To obtain the DAG that corresponds to an expression of the form $D^{\boldsymbol{\alpha}}f(\boldsymbol{x})$, where $\boldsymbol{\alpha} \in \mathbb{N}^p$ is some multiindex, our idea is to repeatedly record the operations performed by the automatic differentiator itself. For instance, given some $\boldsymbol{x}\in\mathbb{R}^2$, say we want to construct the DAG that corresponds to the expression 

$$ D^{(2,1)}f(\boldsymbol{x}) = \partial_{xxy}f(\boldsymbol{x}).$$

for some function $f:\mathbb{R}\to\mathbb{R}$. Then we can succeed as follows:

1. Construct the DAG of $f(\boldsymbol{x})$ in the form of a tape `t`.
2. Feed the tape `t` into a reverse-mode AD algorithm and record the operations performed by this algorithm. This yields a tape `t_p` that contains in particular the Nodes `n_x` and `n_y` representing the quantities $\partial_{x}f(\boldsymbol{x})$ and, respectively $\partial_{y}f(\boldsymbol{x})$
3. Feed the tape `t_p` into an automatic differentator with starting position `n_x.tpos` and again record its operations, yielding a tape `t_pp` which holds Nodes `n_xx` and `n_xy` that hold the derivative values $\partial_{xx}f(\boldsymbol{x})$ and, respectively $\partial_{xy}f(\boldsymbol{x})$.
4. Finally, run the automatic differentiator with input tape `t_pp` and starting position `n_xx.tpos` which yields a tape `t_ppp`.
5. Grab the Node `t_ppp[n_xxy.tpos]` and read its value. 




