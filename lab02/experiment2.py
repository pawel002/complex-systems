import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Global spatial / temporal setup
# ============================================================

L = 1.0           # domain length
N = 101           # number of spatial points
x = np.linspace(0.0, L, N)
dx = x[1] - x[0]

T_END = 35.0      # final time
dt = 0.01         # time step
NSTEPS = int(T_END / dt)

# Base biophysical parameters (will be slightly varied per scenario)
D_base = 0.001    # diffusion coefficient
rho_base = 0.25   # logistic growth rate
beta_base = 0.6   # radiosensitivity
K = 1.0           # carrying capacity

# ------------------------------------------------------------
# Hypoxia profile H(x) (fixed for all scenarios)
# ------------------------------------------------------------
# Better oxygenation near x ~ 0.3
H_vec = 0.4 + 0.6 * np.exp(-((x - 0.5 * L) / 0.5) ** 2)

# ============================================================
# 2. Temporal dose schedule r(t) (same for all scenarios)
# ============================================================

def dose_rate_time(t):
    """
    Fractionated radiotherapy schedule:
    - 3 courses
    - each course has 5 daily fractions
    - each fraction lasts 0.2 time units
    - courses start at t = 5, 15, 25
    """
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

# ============================================================
# 3. Spatial patterns: initial conditions and dose profiles
# ============================================================

def initial_condition(kind, x):
    """
    Return non-uniform initial tumor density u0(x) in [0,1].
    """
    if kind == "left_peak":
        u0 = 0.1 + 0.5 * np.exp(-((x - 0.2) / 0.1) ** 2)
    elif kind == "center_peak":
        u0 = 0.1 + 0.5 * np.exp(-((x - 0.5) / 0.1) ** 2)
    elif kind == "right_peak":
        u0 = 0.1 + 0.5 * np.exp(-((x - 0.8) / 0.1) ** 2)
    elif kind == "double_peak":
        u0 = (
            0.05
            + 0.35 * np.exp(-((x - 0.25) / 0.07) ** 2)
            + 0.35 * np.exp(-((x - 0.75) / 0.07) ** 2)
        )
    else:
        # fallback: uniform
        u0 = 0.2 * np.ones_like(x)

    u = np.clip(u0, 0.0, 1.0)
    return u / np.sum(u) * 20


def dose_profile(kind, x):
    """
    Return spatial weight W(x) for the dose R(x,t) = r(t)*W(x).
    W(x) ∈ [0,1] describes how much of the beam hits each location.
    """
    if kind == "uniform":
        W = np.ones_like(x)
    elif kind == "left_focus":
        W = 0.2 + 0.8 * np.exp(-((x - 0.2) / 0.1) ** 2)
    elif kind == "center_focus":
        W = 0.2 + 0.8 * np.exp(-((x - 0.5) / 0.1) ** 2)
    elif kind == "right_focus":
        W = 0.2 + 0.8 * np.exp(-((x - 0.8) / 0.1) ** 2)
    else:
        W = np.ones_like(x)

    # Normalize so max = 1
    W = W / np.sum(W)
    return W * 200

# ============================================================
# 4. Core simulator: PDE + surrogate ODE for one parameter set
# ============================================================

def run_simulation(D, rho, beta, u0, W):
    """
    Run one simulation of the PDE and its ODE surrogate.

    PDE:  du/dt = D u_xx + rho u(1-u/K) - beta R(x,t) H(x) u
    ODE:  dU/dt = rho U(1-U/K) - beta r(t) <W*H> U

    where R(x,t) = r(t) * W(x) and < · > denotes spatial average.

    Returns
    -------
    times      : (NSTEPS+1,) array
    mass_pde   : (NSTEPS+1,) array, integral of u(x,t) over space
    mass_ode   : (NSTEPS+1,) array, same quantity from ODE surrogate
    """

    # Precompute constants
    WH_mean = np.mean(W * H_vec)  # <W*H>

    # Initialize PDE state
    u = u0.copy()

    # Initialize ODE state (spatial mean of u0)
    U = np.mean(u0)

    # Storage for outputs
    times = np.zeros(NSTEPS + 1)
    mass_pde = np.zeros(NSTEPS + 1)
    mass_ode = np.zeros(NSTEPS + 1)

    # Initial masses
    times[0] = 0.0
    mass_pde[0] = np.trapz(u, x)
    mass_ode[0] = U * L

    # Time stepping
    for n in range(1, NSTEPS + 1):
        t = n * dt

        # --- PDE step (explicit Euler, Neumann BCs) ---
        u_xx = np.zeros_like(u)
        # interior
        u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        # Neumann at boundaries: du/dx = 0
        u_xx[0] = 2 * (u[1] - u[0]) / dx**2
        u_xx[-1] = 2 * (u[-2] - u[-1]) / dx**2

        r_t = dose_rate_time(t)          # temporal dose
        R_xt = r_t * W                   # full R(x,t) = r(t)*W(x)
        kill_term = beta * R_xt * H_vec * u

        du_dt = D * u_xx + rho * u * (1.0 - u / K) - kill_term
        u = u + dt * du_dt
        u = np.clip(u, 0.0, 1.0)

        # --- ODE surrogate step ---
        dU_dt = rho * U * (1.0 - U / K) - beta * r_t * WH_mean * U
        U = U + dt * dU_dt
        U = max(U, 0.0)

        # --- Store masses ---
        times[n] = t
        mass_pde[n] = np.trapz(u, x)   # integral over space
        mass_ode[n] = U * L            # same quantity for the ODE

    return times, mass_pde, mass_ode

# ============================================================
# 5. Scenario definitions (4×4 grid)
# ============================================================

# 4 different non-uniform initial conditions
ic_kinds = ["left_peak", "center_peak", "right_peak", "double_peak"]

# 4 different spatial dose patterns
dose_kinds = ["uniform", "left_focus", "center_focus", "right_focus"]

# Factors to slightly vary rho (by row) and beta (by column)
rho_factors = [0.7, 0.9, 1.1, 1.3]
beta_factors = [0.6, 0.8, 1.0, 1.2]

# ============================================================
# 6. NEW: Visualise initial conditions u0(x) in 4×1 plots
# ============================================================

fig_ic, axes_ic = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for i, ic_name in enumerate(ic_kinds):
    u0_demo = initial_condition(ic_name, x)
    ax = axes_ic[i]
    ax.plot(x, u0_demo)
    ax.set_ylabel("u0(x)")
    ax.set_title(f"{ic_name}")
    ax.set_xlabel("spatial x")

axes_ic[0].set_ylabel("value")
fig_ic.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
fig_ic.suptitle("Non-uniform initial conditions u0(x)", fontsize=14)
plt.savefig("figures/u.png")

# ============================================================
# 7. NEW: Visualise spatial masks W(x) for R(x,t) in 4×1 plots
# ============================================================

fig_mask, axes_mask = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for i, dose_name in enumerate(dose_kinds):
    W_demo = dose_profile(dose_name, x)
    ax = axes_mask[i]
    ax.plot(x, W_demo)
    ax.set_title(f"{dose_name}")
    ax.set_xlabel("spatial x")

axes_mask[0].set_ylabel("value")
fig_mask.suptitle("Spatial masks W(x) for R(x,t) = r(t)·W(x)", fontsize=14)
fig_mask.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
plt.savefig("figures/r.png")
# ============================================================
# 8. Run 16 simulations and plot cumulative tumor mass in 4×4 grid
# ============================================================

fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)

scenario_idx = 0
for i, ic_name in enumerate(ic_kinds):
    for j, dose_name in enumerate(dose_kinds):
        scenario_idx += 1

        # Scenario-specific parameters
        D = D_base
        rho = rho_base * rho_factors[i]
        beta = beta_base * beta_factors[j]

        # Build spatial profiles
        u0 = initial_condition(ic_name, x)
        W = dose_profile(dose_name, x)

        # Run simulation (PDE + ODE)
        times, mass_pde, mass_ode = run_simulation(D, rho, beta, u0, W)

        ax = axes[i, j]
        ax.plot(times, mass_pde, label="PDE mass")
        ax.plot(times, mass_ode, "--", label="ODE mass")

        ax.set_title(
            f"{scenario_idx}: IC={ic_name}, R={dose_name}\n"
            f"p={rho:.2f}, β={beta:.2f}"
        )

        if i == 3:
            ax.set_xlabel("time")
        if j == 0:
            ax.set_ylabel("tumor mass (∫u dx)")

# One global legend and title
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.suptitle(
    "Radiotherapy model - cumulative tumor mass (PDE vs ODE)\n"
    "16 scenarios with varying parameters, u₀(x), and R(x,t)",
    fontsize=16
)

fig.tight_layout(rect=[0.03, 0.03, 0.9, 0.93])
plt.savefig("figures/f3.png")
