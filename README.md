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

The core idea is shared with popular ML libraries such as Tensorflow or PyTorch and relies on recording the series of performed computations $f_1,\dots,f_n$ that result in a particular DAG onto a "tape" $\boldsymbol{T}$, where the relevant data for each intermediate result produced by a step $f_i$ will be stored in its own *Node* object $\boldsymbol{N}$. The tape will be initialized with the *root-Nodes* representing the DAG inputs and a new Node is appended to the end of the tape whenever a step $f_i$ is performed. We will represent Nodes as (mutable) Julia structs. The most basic information hold by an instance `N` of the Node struct is the Node's position on the tape `N.tpos` and the numerical value `N.value` obtained from the computational step that led to the instanciation of the Node `N`.

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

### Example.

The computation $f(x,y) = (x+y)^2$ can be performed within the two steps $f_1(x,y) = x+y$ and, respectively, $f_2(f_1) = f_1^2$. We show how to construct the DAG corresponding to $f$ with inputs $x=1.0$ and $y=2.0$. To start with, we initialize a tape `t` with two root-Nodes, i.e.

```julia
x = Node(value=1.0, tpos=1)
y = Node(value=2.0, tpos=2)
t = [x, y]
```

Next, we append the Node `f_1`, representing the intermediate result from the first computational step, to the tape `t`.

```julia
f_1 = Node(tpos  = length(t)+1
	   pdata = ParentData([1, 2],
			      [1.0, 1.0]
	                     )
          )
push!(t, f_1)
```

We can proceed similiarly with the second computational step. However, we automatize this in terms of a "Record-step" method.

```julia
function square!(node::Node, tape::Vector{Node})
	value = node.value
	pdata = ParentData([node.tpos],
	                   [2*node.value]
	                  )
	push!(tape, Node(value=value,
			 pdata=pdata			 
	                 )
             )
end

square!(last(t), t)			  
```

We can then pass our tape `t` to a reverse-mode automatic differentiator in order to obtain the derivatives $\partial_xf(1,2)$ and $\partial_yf(1,2)$. Note that as soons as the automatic differentiator reaches one of the root-Nodes, there is nothing left to do for the algorithm. Thus, we can split a tape into $\boldsymbol{T}_0$, that holds only the root-Nodes, and $\boldsymbol{T}_c$ which holds all the Nodes that resulted from computations.

## AD Recording

To obtain the DAG that corresponds to an expression of the form $D^{\boldsymbol{\alpha}}f(\boldsymbol{x})$, where $\boldsymbol{\alpha} \in \mathbb{N}^p$ is some multiindex, our idea is to repeatedly record the operations performed by the automatic differentiator itself. For instance, given some $\boldsymbol{x}\in\mathbb{R}^2$, say we want to construct the DAG that corresponds to the expression 

$$ D^{(2,1)}f(\boldsymbol{x}) = \partial_{xxy}f(\boldsymbol{x}).$$

for some function $f:\mathbb{R}\to\mathbb{R}$. Then we can succeed as follows:

1. Construct the DAG of $f(\boldsymbol{x})$ in the form of a tape `t`.
2. Feed the tape `t` into a reverse-mode AD algorithm and record the operations performed by this algorithm. This yields a tape `t_p` that contains in particular the Nodes `n_x` and `n_y`.
3. Feed the tape `t_p` into an automatic differentator with start position `n_x.tpos` and again record its operations, yielding a tape `tape_pp` 




