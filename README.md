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


## AD Recording

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











