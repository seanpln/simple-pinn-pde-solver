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

## DAG construction

The core idea is shared with popular ML libraries such as Tensorflow or PyTorch and relies on recording the performed computations that result in a particular DAG onto a "tape".

### Node struct

A reverse-mode automatic differentiator works by traversing a given DAG in reverse topologically sorted order $[\boldsymbol{N}_1,\dots,\boldsymbol{N}_M]$. The algorithm starts at one of the out-Nodes $\boldsymbol{N}_s\ (s \leq M)$ and stops until it reaches a root-Node. A reverse-mode automatic differentiator accumulates derivative data from Nodes that directly depend on each other (so called *local gradient data*) into *total gradients* of the form $\partial v_s/\partial v_i$. 

In order to "send" the terms ... out to the current Node's parents, the AD algortihm needs to locate these Nodes. We therefor assign a label 

```julia
ID = Tuple{Int8,Int8,Int64,Int64} 
```

to each Node, which is specified as follows:

* blabla
* blabla
* blabla
* blabla

Among plain adjacency info, we will also endow each Node with local gradient data which can be processed by the AD algorithm into the products ... These products will be add as summands into the total gradient attributes of the parents listed in the `ParentData.n` attribute  

```julia
struct ParentData
	n::Vector{ID}		     # tape position
	x::Vector{Float64}           # local gradient (used by 'autodiff')
	g::Vector{Int8}              # gradient function (used by 'adrecord')
end
```

A `Node` object $\boldsymbol{N}$ contains information about the intermediate results that where used by the operation that created $\boldsymbol{N}$



### Node 

```julia
ID      = Tuple{Int8,Int8,Int64,Int64} 

struct ParentData
	n::Vector{ID}		     # tape position
	x::Vector{Float64}           # local gradient (used by 'autodiff')
	g::Vector{Int8}              # gradient function (used by 'adrecord')
end

mutable struct Node

	n::ID                        # tape position
	ṅ::Int64                     # total gradient's position  (tracked by 'adrecord')                  
	v::Float64                   # v
        v̇::Float64                   # total gradient v (set by 'autodiff')	
	p::ParentData                # parent data
	# only for parameter nodes:
	mt::Union{Float64,Nothing}   # adam: first moment
	vt::Union{Float64,Nothing}   # adam: second moment
	m̂::Union{Float64,Nothing}    # adam: first moment: bias correction
	v̂::Union{Float64,Nothing}    # adam: second moment: bias correction
	#

	function Node(v;
		      n  = (0,0,0,0),
		      ṅ  = 0,
		      v̇  = 0.0,
		      p  = ParentData(ID[], Float64[], Int8[]),      
		      mt = nothing,
		      vt = nothing,
	              m̂  = nothing,
	              v̂  = nothing            
	             )
		new(n, ṅ, v, v̇, p, mt, vt, m̂ , v̂)
	end
end

```

The DAG construction boils down to the problem of constructing the DAGs for expressions of the form $D^{\boldsymbol{\alpha}}\Phi_{\theta}(\boldsymbol{x})$, where $\boldsymbol{\alpha} \in \mathbb{N}^p$ is some multiindex and $\boldsymbol{x}$ a training sample. Our idea for constructing these DAGs is to repeatedly record the operations performed by a reverse-mode AD algorithm.
