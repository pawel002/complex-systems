import numpy as np

# ----------------------
# Utilities
# ----------------------
def laplacian_neumann(u, dx):
    # 2-D 5-point Laplacian with zero-flux via edge mirroring
    up = np.pad(u, 1, mode='edge')
    L = (up[2:,1:-1] + up[:-2,1:-1] + up[1:-1,2:] + up[1:-1,:-2] - 4*up[1:-1,1:-1]) / (dx*dx)
    return L

def make_radial_po2_map(nx, ny, p_center=1.0, p_edge=30.0):
    yy, xx = np.meshgrid(np.linspace(-1,1,ny), np.linspace(-1,1,nx), indexing='ij')
    r = np.sqrt(xx**2 + yy**2)
    r = np.clip(r, 0, 1)
    return p_center + (p_edge - p_center) * r  # low O2 at center, higher at rim

def OMF_from_pO2(pO2, OER_max=3.0, K_m=3.0):
    # Alper–Howard–Flanders-inspired OMF in [~1/OER_max, 1]
    return (1.0 + (OER_max - 1.0) * (pO2 / (pO2 + K_m))) / OER_max

def weekday_fraction_times(start_day, n_fractions):
    # 5 fractions per week (Mon–Fri), no explicit calendar: two-day gap after every 5
    times = []
    day = start_day
    for i in range(n_fractions):
        times.append(day)
        day += 1.0
        if (i+1) % 5 == 0:
            day += 2.0  # weekend gap
    return np.array(times)

# ----------------------
# Core simulator
# ----------------------
class RTParams:
    def __init__(self, alpha=0.06, beta=0.006, OER_max=3.0, K_m=3.0):
        self.alpha = alpha
        self.beta = beta
        self.OER_max = OER_max
        self.K_m = K_m

def simulate_pde(
    T_days=120.0, dt=0.05,
    L_mm=50.0, N=100,
    D_mm2_per_day=0.137, rho_per_day=0.0274, K=1.0,
    use_LQ_pulses=False,
    # RT schedule
    start_day=10.0, n_fractions=30, d_per_frac=2.0, frac_duration_day=0.004,  # ~6 min
    # Hypoxia + beam
    p_center=1.0, p_edge=30.0, beam_uniform=True, beam_sigma_mm=10.0,
    rt=RTParams()
):
    dx = L_mm / N
    nx = ny = N
    x = np.linspace(0, L_mm, nx)
    y = np.linspace(0, L_mm, ny)

    # Initial tumor: small Gaussian near center
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = 0.1 * np.exp(-((X - L_mm/2)**2 + (Y - L_mm/2)**2) / (2*(5.0**2)))

    # Hypoxia map -> OMF -> H(x)
    pO2 = make_radial_po2_map(nx, ny, p_center, p_edge)
    H = OMF_from_pO2(pO2, rt.OER_max, rt.K_m)  # in [~1/3, 1]

    # Beam profile (1=uniform or centered Gaussian field)
    if beam_uniform:
        beam = np.ones_like(u)
    else:
        r2 = (X - L_mm/2)**2 + (Y - L_mm/2)**2
        beam = np.exp(-r2 / (2*(beam_sigma_mm**2)))
        beam /= beam.max()

    # Precompute fraction times
    fr_times = weekday_fraction_times(start_day, n_fractions)
    fr_end = fr_times + frac_duration_day
    # For continuous-kill PDE we need dose rate during delivery:
    dose_rate = d_per_frac / frac_duration_day  # Gy/day during the short window

    # Storage for quick diagnostics
    times = np.arange(0.0, T_days + 1e-12, dt)
    total_cells = np.zeros_like(times)

    # For LQ pulses: precompute hypoxia-weighted alpha/beta maps
    alpha_eff = rt.alpha * H
    beta_eff  = rt.beta * (H**2)

    # Helper to know if we are inside any fraction window
    def in_delivery_window(t):
        return np.any((t >= fr_times) & (t < fr_end))

    # Main loop
    cumdose = np.zeros_like(u)  # cumulative dose if you want LQ-continuous
    for k, t in enumerate(times):
        # Diffusion + logistic growth
        Lu = laplacian_neumann(u, dx)
        growth = rho_per_day * u * (1.0 - u / K)
        du = D_mm2_per_day * Lu + growth

        if use_LQ_pulses:
            # No continuous killing; apply pulses exactly at fraction *end* times
            # (closest grid time step that just passes fr_end)
            # Integrate tumor forward without RT for this dt
            u = np.clip(u + dt * du, 0.0, K)
            # If we just crossed a fraction end, apply survival
            crossed = np.isclose(t % 1.0, 0.0, atol=1e-9)  # cheap; optional
            for t_end in fr_end:
                if (t - dt) < t_end <= t:
                    local_dose = d_per_frac * beam  # Gy map delivered this fraction
                    S = np.exp(-(alpha_eff * local_dose + beta_eff * local_dose**2))
                    u *= S
                    break
        else:
            # Continuous-dose PDE kill during brief delivery windows
            if in_delivery_window(t):
                R_xt = dose_rate * beam  # Gy/day during delivery
                # Baseline: linear kill only (β here = alpha in LQ), hypoxia-weighted
                kill = (rt.alpha * R_xt) * H * u
                du -= kill
                # Track cumulative dose if you later want LQ-consistent continuous kill
                cumdose += (R_xt * dt)
            u = np.clip(u + dt * du, 0.0, K)

        total_cells[k] = u.sum()

    return {
        "x": x, "y": y, "u": u, "times": times, "total_cells": total_cells,
        "pO2": pO2, "H": H, "beam": beam
    }

if __name__ == "__main__":
    # 1) No RT baseline growth
    out0 = simulate_pde(T_days=40, n_fractions=0)
    print("Baseline (no RT) final tumor mass:", out0["total_cells"][-1])

    # 2) Standard 60 Gy with uniform beam, continuous PDE kill
    out1 = simulate_pde(T_days=120, use_LQ_pulses=False)
    print("With RT (continuous kill) final mass:", out1["total_cells"][-1])

    # 3) Same schedule but exact LQ pulses (usually a bit stronger kill than linear-only PDE kill)
    out2 = simulate_pde(T_days=120, use_LQ_pulses=True)
    print("With RT (LQ pulses) final mass:", out2["total_cells"][-1])

    # 4) Make hypoxia more severe and see worse response
    out3 = simulate_pde(T_days=120, p_center=0.5, p_edge=10.0, use_LQ_pulses=True)
    print("With RT (LQ) & severe hypoxia final mass:", out3["total_cells"][-1])
