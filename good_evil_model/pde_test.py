# Simulate three chaotic systems (Lorenz, Rössler, Chen) with RK4 and plot in 3D.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def rk4(f, x0, t0, dt, n):
    """Runge-Kutta 4 integrator.
    f: function f(x, t) -> dx/dt
    x0: initial state (array-like)
    t0: initial time
    dt: timestep
    n: number of steps
    """
    x = np.zeros((n, len(x0)), dtype=float)
    t = np.zeros(n, dtype=float)
    x[0] = np.array(x0, dtype=float)
    t[0] = t0
    for i in range(1, n):
        xi = x[i-1]
        ti = t[i-1]
        k1 = f(xi, ti)
        k2 = f(xi + 0.5*dt*k1, ti + 0.5*dt)
        k3 = f(xi + 0.5*dt*k2, ti + 0.5*dt)
        k4 = f(xi + dt*k3, ti + dt)
        x[i] = xi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t[i] = ti + dt
    return t, x

# 1) Lorenz system
def lorenz(params):
    sigma, rho, beta = params
    def f(state, t):
        x, y, z = state
        dx = sigma*(y - x)
        dy = x*(rho - z) - y
        dz = x*y - beta*z
        return np.array([dx, dy, dz], dtype=float)
    return f

# 2) Rössler system
def rossler(params):
    a, b, c = params
    def f(state, t):
        x, y, z = state
        dx = -y - z
        dy = x + a*y
        dz = b + z*(x - c)
        return np.array([dx, dy, dz], dtype=float)
    return f

# 3) Chen system
def chen(params):
    a, b, c = params
    def f(state, t):
        x, y, z = state
        dx = a*(y - x)
        dy = (c - a)*x - x*z + c*y
        dz = x*y - b*z
        return np.array([dx, dy, dz], dtype=float)
    return f

# Simulation settings
dt = 0.01
steps = 20000  # 200 seconds

# Initial conditions
x0_lorenz = [1.0, 1.0, 1.0]
x0_rossler = [0.1, 0.0, 0.0]
x0_chen = [-10.0, 0.0, 37.0]

# Parameters
params_lorenz = (10.0, 28.0, 8.0/3.0)
params_rossler = (0.2, 0.2, 5.7)
params_chen = (35.0, 3.0, 28.0)

# Integrate
t_l, X_l = rk4(lorenz(params_lorenz), x0_lorenz, 0.0, dt, steps)
t_r, X_r = rk4(rossler(params_rossler), x0_rossler, 0.0, dt, steps)
t_c, X_c = rk4(chen(params_chen), x0_chen, 0.0, dt, steps)

# Discard a short transient for clearer plots
trim = 1000
X_l_plot = X_l[trim:]
X_r_plot = X_r[trim:]
X_c_plot = X_c[trim:]

# Plot Lorenz
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(X_l_plot[:,0], X_l_plot[:,1], X_l_plot[:,2])
ax1.set_title("Lorenz Attractor (σ=10, ρ=28, β=8/3)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
plt.show()

# Plot Rössler
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(X_r_plot[:,0], X_r_plot[:,1], X_r_plot[:,2])
ax2.set_title("Rössler Attractor (a=0.2, b=0.2, c=5.7)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
plt.show()

# Plot Chen
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot(X_c_plot[:,0], X_c_plot[:,1], X_c_plot[:,2])
ax3.set_title("Chen Attractor (a=35, b=3, c=28)")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")
plt.show()
