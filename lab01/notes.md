Radiotherapy – a model of hypoxia and the radiation effect. Baseline model:

$$
    \frac{\partial u(x, t)}{\partial t} = D \nabla^2 u + \rho u(1 - \frac{u}{K}) -
\beta R(x, t) H(x) u,
$$

where:

- $R(x, t)$ - radiation dose distribution.
- $H(x)$ - hypoxia function.
- $\beta$ - radiosensitivity coefficient (sensitivity to radiation).
- $u$ - tumor cell density function, normalized to $K=1$.

Parts of the equation represent:

- $D \nabla^2$ - motility of cancer cells (ability to move from one location to another)
- $\rho u(1-\frac{u}{K})$ - poliferation of cells (the increase in cell number through cell growth and division).
- $\beta R(x,t) H(x) u$ - the rate of killing cancer cells, weighted by hypoxia.
