import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from title import params_to_mathtext

def get_delayed_value(t_hist, y_hist, t_query):
    if t_query <= t_hist[0]:
        return y_hist[0]
    if t_query >= t_hist[-1]:
        return y_hist[-1]
    
    idx = np.searchsorted(t_hist, t_query)
    t1, t2 = t_hist[idx-1], t_hist[idx]
    y1, y2 = y_hist[idx-1], y_hist[idx]
    w = (t_query - t1) / (t2 - t1)
    return (1 - w) * y1 + w * y2

def simulate(params, T=120.0, dt=0.01, B0=0.25, P0=0.05):
    sigma = params["sigma"]
    rho   = params["rho"]
    alpha = params["alpha"]
    delta = params["delta"]
    tau   = params["tau"]
    kappa = params["kappa"]
    n     = params["n"]
    tau_d = params["tau_d"]
    
    steps = int(T/dt) + 1
    t = np.linspace(0, T, steps)
    B = np.empty(steps); P = np.empty(steps); G = np.empty(steps)
    B[0] = B0; P[0] = P0; G[0] = 1 - B0
    
    t_hist = [t[0]]
    B_hist = [B0]
    
    for k in range(1, steps):
        t_prev = t[k-1]
        B_delay = get_delayed_value(np.array(t_hist), np.array(B_hist), t_prev - tau_d)
        
        dB = sigma * (1 - B[k-1]) * B[k-1] - rho * P[k-1] * B[k-1]
        dP = (alpha * (B_delay ** n) - delta) * P[k-1] + tau * (1 - B[k-1]) - kappa * (P[k-1] ** 2)
        
        Bk = B[k-1] + dt * dB
        Pk = P[k-1] + dt * dP
        
        Bk = min(max(Bk, 0.0), 1.0)
        Pk = max(Pk, 0.0)
        
        B[k] = Bk
        P[k] = Pk
        G[k] = 1 - Bk
        
        t_hist.append(t[k])
        B_hist.append(Bk)
    
    return t, G, B, P

experiments = [
    ("E1 — Mild nonlinearity, short delay",
     {"sigma": 0.9, "rho": 0.6, "alpha": 1.4, "delta": 0.35, "tau": 0.08, "kappa": 0.20, "n": 1.5, "tau_d": 0.2},
     {"B0": 0.25, "P0": 0.08, "T": 100.0, "dt": 0.01}),
    ("E2 — Superlinear response, moderate delay",
     {"sigma": 0.9, "rho": 0.6, "alpha": 2.2, "delta": 0.35, "tau": 0.05, "kappa": 0.18, "n": 2.0, "tau_d": 0.6},
     {"B0": 0.22, "P0": 0.06, "T": 120.0, "dt": 0.01}),
    ("E3 — Strong response, small baseline funding",
     {"sigma": 0.85, "rho": 0.65, "alpha": 2.8, "delta": 0.35, "tau": 0.01, "kappa": 0.18, "n": 2.5, "tau_d": 0.8},
     {"B0": 0.20, "P0": 0.05, "T": 120.0, "dt": 0.01}),
    ("E4 — High temptation vs enforcement",
     {"sigma": 1.2, "rho": 0.55, "alpha": 2.4, "delta": 0.35, "tau": 0.03, "kappa": 0.16, "n": 2.2, "tau_d": 0.7},
     {"B0": 0.25, "P0": 0.04, "T": 120.0, "dt": 0.01}),
    ("E5 — Capacity cap relaxed (more boom-bust)",
     {"sigma": 0.95, "rho": 0.6, "alpha": 2.6, "delta": 0.3, "tau": 0.04, "kappa": 0.05, "n": 2.3, "tau_d": 0.9},
     {"B0": 0.24, "P0": 0.05, "T": 120.0, "dt": 0.01}),
    ("E6 — Heavier cap (damped oscillations)",
     {"sigma": 0.95, "rho": 0.6, "alpha": 2.6, "delta": 0.3, "tau": 0.04, "kappa": 0.50, "n": 2.3, "tau_d": 0.9},
     {"B0": 0.24, "P0": 0.05, "T": 120.0, "dt": 0.01}),
    ("E7 — Longer delay",
     {"sigma": 0.9, "rho": 0.6, "alpha": 2.0, "delta": 0.32, "tau": 0.06, "kappa": 0.20, "n": 2.0, "tau_d": 1.3},
     {"B0": 0.22, "P0": 0.06, "T": 140.0, "dt": 0.01}),
    ("E8 — Very strong nonlinearity",
     {"sigma": 0.9, "rho": 0.6, "alpha": 3.0, "delta": 0.33, "tau": 0.02, "kappa": 0.15, "n": 3.0, "tau_d": 0.6},
     {"B0": 0.20, "P0": 0.06, "T": 120.0, "dt": 0.01}),
    ("E9 — Lower enforcement effectiveness",
     {"sigma": 1.0, "rho": 0.45, "alpha": 2.4, "delta": 0.35, "tau": 0.05, "kappa": 0.18, "n": 2.2, "tau_d": 1.0},
     {"B0": 0.28, "P0": 0.05, "T": 140.0, "dt": 0.01}),
    ("E10 — Noise-free oscillatory window (tuned)",
     {"sigma": 0.88, "rho": 0.62, "alpha": 2.35, "delta": 0.34, "tau": 0.75, "kappa": 0.12, "n": 2.4, "tau_d": 0.95},
     {"B0": 0.23, "P0": 0.05, "T": 140.0, "dt": 0.01}),
]

for i, (label, params, simcfg) in enumerate(experiments):
    print(f"Running experiment {i}...")
    t, G, B, P = simulate(params, **simcfg)
    plt.figure()
    plt.plot(t, G, label="Good (G)")
    plt.plot(t, B, label="Bad (B)")
    plt.plot(t, P, label="Guards (P)")
    title = params_to_mathtext(params, 
        order=["sigma","rho","alpha","delta","tau","kappa","n","tau_d"],
        precision=2,
        wrap=True
    )
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("population (normalized)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"data/experiment_0{i}.png")