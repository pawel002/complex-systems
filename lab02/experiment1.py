import numpy as np
import matplotlib.pyplot as plt

# ---- Parameters for the PDE and surrogate ODE ----
L = 1.0
N = 101
dx = L / (N - 1)
x = np.linspace(0, L, N)

D = 0.001      # diffusion coefficient
rho = 0.3     # logistic growth rate
beta = 1.0    # radiosensitivity
K = 1.0

# Hypoxia/oxygenation profile H(x): better oxygenation near x ~ 0.3
H_vec = 0.4 + 0.6 * np.exp(-((x - 0.5 * L)/0.5) ** 2)
H_eff = np.mean(H_vec)


def dose_rate(t):
    dose_amp = 4.0
    fraction_duration = 0.2
    course_starts = [5.0, 15.0, 25.0]

    for cs in course_starts:
        for n in range(5):
            t_start = cs + n * 1.0
            t_end = t_start + fraction_duration
            if t_start <= t <= t_end:
                return dose_amp
    return 0.0


T_end = 35.0
dt = 0.01
nsteps = int(T_end / dt)

# Initial conditions
u = 0.2 * np.ones(N)  # PDE: uniform initial tumor cell density
U_ode = 0.2           # ODE: same initial mean density

times = np.zeros(nsteps + 1)
U_avg_hist = np.zeros(nsteps + 1)
U_ode_hist = np.zeros(nsteps + 1)
dose_hist = np.zeros(nsteps + 1)

times[0] = 0.0
U_avg_hist[0] = np.mean(u)
U_ode_hist[0] = U_ode
dose_hist[0] = dose_rate(0.0)

# ---- Time integration ----
for n in range(1, nsteps + 1):
    t = n * dt

    # PDE step (explicit Euler, Neumann BCs)
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    u_xx[0] = 2 * (u[1] - u[0]) / dx**2
    u_xx[-1] = 2 * (u[-2] - u[-1]) / dx**2

    R_t = dose_rate(t)
    du_dt = D * u_xx + rho * u * (1 - u / K) - beta * R_t * H_vec * u
    u = u + dt * du_dt
    u = np.clip(u, 0.0, None)

    # ODE surrogate for the spatial mean
    dU_dt = rho * U_ode * (1 - U_ode / K) - beta * H_eff * R_t * U_ode
    U_ode = U_ode + dt * dU_dt
    U_ode = max(U_ode, 0.0)

    times[n] = t
    U_avg_hist[n] = np.mean(u)
    U_ode_hist[n] = U_ode
    dose_hist[n] = R_t

# ---- Plot: spatial mean of PDE vs ODE surrogate ----
plt.figure()
plt.plot(times, U_avg_hist, label="PDE spatial mean")
plt.plot(times, U_ode_hist, linestyle="--", label="ODE surrogate")
plt.xlabel("time")
plt.ylabel("mean tumor cell density")
plt.legend()
plt.title("PDE spatial mean vs ODE surrogate (3x5 fractions)")
plt.tight_layout()
plt.savefig("figures/f1.png")

# ---- Plot: dose schedule (for reference) ----
plt.figure()
plt.plot(times, dose_hist, label="dose_rate(t)")
plt.xlabel("time")
plt.ylabel("dose rate")
plt.title("Radiation dose schedule: 3 courses x 5 fractions")
plt.tight_layout()
plt.savefig("figures/f2.png")
