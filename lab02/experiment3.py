import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. Global spatial / temporal / biophysical setup
# ============================================================

# Spatial domain
L = 1.0           # domain length
N = 101           # number of spatial grid points
x = np.linspace(0.0, L, N)
dx = x[1] - x[0]

# Time domain
T_END = 35.0      # final simulation time
dt = 0.01         # time step
NSTEPS = int(T_END / dt)

# Base parameters (we will vary them around these values)
D_base = 0.001    # diffusion coefficient
rho_base = 0.25   # logistic growth rate
beta_base = 0.6   # radiosensitivity
K = 1.0           # carrying capacity

# Hypoxia profile H(x): better oxygenation near x ~ 0.3
H_vec = 0.4 + 0.6 * np.exp(-((x - 0.3 * L) / 0.15) ** 2)

# Initial condition and dose mask kinds (for variety)
ic_kinds = ["left_peak", "center_peak", "right_peak", "double_peak"]
dose_kinds = ["uniform", "left_focus", "center_focus", "right_focus"]

# For reproducibility (optional)
np.random.seed(0)
torch.manual_seed(0)

# ============================================================
# 2. Temporal dose schedule r(t)
# ============================================================

def dose_rate_time(t: float) -> float:
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

def initial_condition(kind: str, x: np.ndarray) -> np.ndarray:
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

    return np.clip(u0, 0.0, 1.0)


def dose_profile(kind: str, x: np.ndarray) -> np.ndarray:
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
    W = W / np.max(W)
    return W

# ============================================================
# 4. PDE + ODE simulator for mean tumor density U(t)
# ============================================================

def run_simulation(D: float,
                   rho: float,
                   beta: float,
                   u0: np.ndarray,
                   W: np.ndarray,
                   T_end: float = T_END,
                   dt_local: float = dt):
    """
    Run one simulation of the PDE and its ODE surrogate.

    PDE:  du/dt = D u_xx + rho u(1-u/K) - beta R(x,t) H(x) u
    with R(x,t) = r(t)*W(x).

    We track the spatial mean of u:
        U_pde(t) = (1/L) ∫ u(x,t) dx

    ODE surrogate:
        dU/dt = rho U(1-U/K) - beta r(t) H_eff U
    where H_eff = <W*H> is an effective radiosensitivity factor.

    Returns
    -------
    times     : (NSTEPS+1,) array
    U_pde     : (NSTEPS+1,) array, spatial mean from PDE
    U_ode     : (NSTEPS+1,) array, mean from ODE surrogate
    """
    nsteps = int(T_end / dt_local)
    times = np.linspace(0.0, T_end, nsteps + 1)

    # PDE state
    u = u0.copy()

    # ODE state (start from same mean)
    U_ode = np.mean(u0)

    # Effective radiosensitivity factor
    H_eff = np.mean(W * H_vec)

    U_pde_hist = np.zeros(nsteps + 1)
    U_ode_hist = np.zeros(nsteps + 1)
    U_pde_hist[0] = np.mean(u0)
    U_ode_hist[0] = U_ode

    for n in range(1, nsteps + 1):
        t = times[n]

        # ---------- PDE step (explicit Euler, Neumann BCs) ----------
        u_xx = np.zeros_like(u)
        # interior
        u_xx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
        # Neumann boundaries: du/dx = 0 => second derivative approx
        u_xx[0] = 2.0 * (u[1] - u[0]) / dx**2
        u_xx[-1] = 2.0 * (u[-2] - u[-1]) / dx**2

        r_t = dose_rate_time(t)
        R_xt = r_t * W
        kill_term = beta * R_xt * H_vec * u

        du_dt = D * u_xx + rho * u * (1.0 - u / K) - kill_term
        u = u + dt_local * du_dt
        u = np.clip(u, 0.0, 1.5)  # small upper bound to avoid blow-up

        U_pde_hist[n] = np.mean(u)

        # ---------- ODE surrogate step ----------
        dU_dt = rho * U_ode * (1.0 - U_ode / K) - beta * H_eff * r_t * U_ode
        U_ode = U_ode + dt_local * dU_dt
        U_ode = max(0.0, U_ode)

        U_ode_hist[n] = U_ode

    return times, U_pde_hist, U_ode_hist

# ============================================================
# 5. MLP surrogate: one-step time integrator for U
# ============================================================

class TumorTimeStepperMLP(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x) 

# ============================================================
# 6. Dataset generation: build one-step pairs from PDE data
# ============================================================

def generate_dataset(num_sims: int,
                     T_train: float,
                     T_end: float = T_END,
                     dt_local: float = dt):
    X_list = []
    y_list = []

    nsteps_total = int(T_end / dt_local)
    nsteps_train = int(T_train / dt_local)

    for sim_idx in range(num_sims):
        # Sample parameters around base values
        rho = rho_base * np.random.uniform(0.7, 1.3)
        beta = beta_base * np.random.uniform(0.6, 1.2)
        D = D_base * np.random.uniform(0.5, 1.5)

        # Randomly pick initial condition and dose mask shape
        ic_name = np.random.choice(ic_kinds)
        dose_name = np.random.choice(dose_kinds)

        u0 = initial_condition(ic_name, x)
        W = dose_profile(dose_name, x)

        # Run PDE + ODE for this scenario
        times, U_pde, U_ode = run_simulation(D, rho, beta, u0, W,
                                             T_end=T_end,
                                             dt_local=dt_local)

        # Build one-step pairs from PDE mean (training interval only)
        for n in range(nsteps_train):
            t_n = times[n]
            r_n = dose_rate_time(t_n)

            U_n = U_pde[n]
            U_next = U_pde[n + 1]

            X_list.append([U_n, r_n, rho, beta, D])
            y_list.append([U_next])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y

# ============================================================
# 7. Training loop
# ============================================================

def train_model(model,
                X_train,
                y_train,
                X_val,
                y_val,
                num_epochs=50,
                batch_size=1024,
                lr=1e-3,
                device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_dev = X_val.to(device)
            y_val_dev = y_val.to(device)
            y_val_pred = model(X_val_dev)
            val_loss = loss_fn(y_val_pred, y_val_dev).item()

        print(f"Epoch {epoch:3d} | train MSE: {epoch_loss:.4e} | "
              f"val MSE: {val_loss:.4e}")

# ============================================================
# 8. Rollout: generate trajectory with trained MLP surrogate
# ============================================================

def rollout_nn(model,
               U0: float,
               rho: float,
               beta: float,
               D: float,
               T_end: float = T_END,
               dt_local: float = dt,
               device: str = "cpu"):
    nsteps = int(T_end / dt_local)
    times = np.linspace(0.0, T_end, nsteps + 1)
    U_nn = np.zeros(nsteps + 1)
    U_nn[0] = U0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for n in range(nsteps):
            t_n = times[n]
            r_n = dose_rate_time(t_n)

            x_in = torch.tensor([[U_nn[n], r_n, rho, beta, D]],
                                dtype=torch.float32, device=device)
            U_next = model(x_in).item()
            # Enforce non-negativity and reasonable cap
            U_nn[n + 1] = max(0.0, min(1.5, U_next))

    return times, U_nn

# ============================================================
# 9. Example main: train the model and compare trajectories
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------------------
    # 9.1 Generate training data
    # ----------------------------
    T_train = 20.0          # training interval (0..T_train)
    num_sims = 200           # number of PDE simulations to generate data from

    print("Generating dataset...")
    X, y = generate_dataset(num_sims=num_sims,
                            T_train=T_train,
                            T_end=T_END,
                            dt_local=dt)
    print("Dataset shape:", X.shape, y.shape)

    # Train/validation split
    n_samples = X.shape[0]
    idx = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # ----------------------------
    # 9.2 Define and train the MLP
    # ----------------------------
    model = TumorTimeStepperMLP(hidden_dim=32)
    print("Training MLP time-stepper...")
    train_model(model, X_train, y_train, X_val, y_val,
                num_epochs=200,
                batch_size=2048,
                lr=1e-4,
                device=device)

    # ----------------------------
    # 9.3 Choose a test scenario
    # ----------------------------
    # Pick some parameters and profiles (not used to build X,y if you like)
    rho_test = rho_base * 1.1
    beta_test = beta_base * 0.9
    D_test = D_base

    ic_name_test = "center_peak"
    dose_name_test = "center_focus"

    u0_test = initial_condition(ic_name_test, x)
    W_test = dose_profile(dose_name_test, x)

    # Run PDE + ODE
    times, U_pde, U_ode = run_simulation(D_test, rho_test, beta_test,
                                         u0_test, W_test,
                                         T_end=T_END,
                                         dt_local=dt)

    # NN rollout (starting from same initial mean)
    U0_mean = np.mean(u0_test)
    times_nn, U_nn = rollout_nn(model, U0_mean, rho_test, beta_test,
                                D_test, T_end=T_END, dt_local=dt,
                                device=device)

    # Convert means to cumulative tumor mass (∫ u dx ≈ L * U)
    mass_pde = L * U_pde
    mass_ode = L * U_ode
    mass_nn = L * U_nn

    # ----------------------------
    # 9.4 Plot comparison
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(times, mass_pde, label="PDE (mass)", linewidth=2)
    plt.plot(times, mass_ode, "--", label="ODE surrogate (mass)", linewidth=2)
    plt.plot(times_nn, mass_nn, ":", label="NN surrogate (mass)", linewidth=2)

    # Highlight the training interval
    plt.axvspan(0.0, T_train, color="gray", alpha=0.1,
                label="training interval")

    plt.xlabel("time")
    plt.ylabel("cumulative tumor mass ≈ ∫ u(x,t) dx")
    plt.title("PDE vs ODE vs Neural-network surrogate")
    plt.legend()
    plt.tight_layout()
    plt.show()
