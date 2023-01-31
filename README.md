# simple-pinn-pde-solver

We train a PINN (physics informed neural network) based machine learning model to predict the state $u:\overline{\Omega} \subset \mathbb{R}^d \to \mathbb{R}$ of a system

$$
\begin{cases}
\mathcal{F}[u] = f & \text{on } \Omega\\
\mathcal{B}[u] = g & \text{on } \partial{\Omega},\\
\end{cases}
$$

whose right hand side data $(f,g)$ is assumed to be sufficiently regular such that pointwise evaluation is well-defined. The model is of the form

$$\Psi_{\theta}(\boldsymbol{x}) = A(\boldsymbol{x}) + B(\boldsymbol{x})\Phi_{\theta}(\boldsymbol{x})$$

and parametrized by a set of weights and biases $\theta$ of a feedforwad neural network $\Phi_{\theta}$. One can either try to find fitting functions $A,B:\overline{\Omega} \to \mathbb{R}$ that achieve $\mathcal{B}[\Psi_\theta] = g$ and hence lead to a model that matches the boundary conditions exactly, or incorporate a boundary term such as $\frac{1}{2}(\mathcal{B}[\Phi_\theta] - g)^2$ in the loss function that determines the model's empirical risk. (In the latter case one sets $A \equiv 0$ and $B \equiv 1$.)  

The training will be performed by means of the popular Adam optimizer, which requires the gradients $\nabla_\theta \hat{ \mathcal{R} }$ of the model's empirical risk with respect to the network parameters $\theta$. The latter can be conveniently obtained by feeding the computational graph (DAG) that underlies $\hat{ \mathcal{R} }$ into a reverse-mode automatic differentiator. However, constructing the DAG of $\hat{\mathcal{R}}$



While first order reverse-mode automatic differentiation (AD) can be used for computing the gradient $\nabla_\theta \hat{ \mathcal{R} })$

Reverse-mode automatic differentiation enables a convenient computation of 


