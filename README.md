## Automatic Differentiation
AD is simply an alternative formulation of the chain rule: Given a computation graph, the partial derivative of a node $y: \mathbb{R}^k \rightarrow \mathbb{R}^l$ with respect to any parameter $\theta$ used in the definition of another node $x: \mathbb{R}^m \rightarrow \mathbb{R}^n$ can be computed as

$$ \frac{\partial y}{\partial \theta} = \sum_{p\ \in\ paths(y \rightarrow x)} \left[\left( \prod^{\curvearrowright}_{(u, v)\ \in\ p} \frac{\partial u}{\partial v} \right) \cdot \frac{\partial x}{\partial \theta} \right] $$ 

i.e. by accumulating Jacobian matrices

$$ \left(\frac{\partial u}{\partial v}\right)_{ij} = \frac{\partial u_i}{\partial v_j} $$
