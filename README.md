# Complex Systems

Radiotherapy – a model of hypoxia and the radiation effect. Baseline model:

$$
    \frac{\partial u(x, t)}{\partial t} = D \nabla^2 u + \rho u(1 - \frac{u}{K}) -
\beta R(x, t) H(x) u,
$$

where:

- $R(x, t)$ - radiation dose distribution
- $H(x)$ - hypoxia function
- $\beta$ - radiosensitivity coefficient (sensitivity to radiation)
