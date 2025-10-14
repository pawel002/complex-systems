from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple
from .params import SimConfig
from .numerics import laplacian_neumann, init_tumor_gaussian, beam_profile
from .hypoxia import radial_pO2, omf_from_pO2
from .schedule import weekday_fractions

@dataclass
class SimOutput:
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    uT: np.ndarray
    pO2: np.ndarray
    H: np.ndarray
    beam: np.ndarray
    total_mass: np.ndarray
    dose_rate_series: np.ndarray

def simulate(cfg: SimConfig, save_hook=None) -> SimOutput:
    g = cfg.grid
    kin = cfg.tumor
    rb = cfg.rtd
    sch = cfg.schedule
    b = cfg.beam

    x, y, L = g.x, g.y, g.L_mm
    dt, T = cfg.dt_day, cfg.T_days
    N = g.N

    if not cfg.stability_ok():
        raise RuntimeError(f"dt={dt:.4g} too large for explicit diffusion; try dt ≤ {cfg.recommended_dt():.4g} day")

    # fields
    u = init_tumor_gaussian(x, y, L)
    pO2 = radial_pO2(N)
    H = omf_from_pO2(pO2, rb.OER_max, rb.K_m_mmHg)
    beam = beam_profile(x, y, L, b.profile, b.sigma_mm)

    times = np.arange(0.0, T + 1e-12, dt)
    total_mass = np.zeros_like(times)
    dose_rate_series = np.zeros_like(times)

    fr_t = weekday_fractions(sch.start_day, sch.n_fractions)
    fr_end = fr_t + sch.frac_duration_day
    dose_rate = sch.d_per_frac_Gy / sch.frac_duration_day  # Gy/day during window

    alpha_eff = rb.alpha * H
    beta_eff  = rb.beta * (H**2)
    cumdose = np.zeros_like(u)

    def in_window(t: float) -> bool:
        return np.any((t >= fr_t) & (t < fr_end))

    for k, t in enumerate(times):
        Lu = laplacian_neumann(u, g.dx)
        growth = kin.rho_per_day * u * (1.0 - u/kin.K)
        du = kin.D_mm2_per_day * Lu + growth

        if rb.mode == "pde_kill":
            if in_window(t):
                Rxt = dose_rate * beam  # Gy/day
                kill = rb.beta_rt_per_Gy * Rxt * H * u
                du -= kill
                cumdose += Rxt * dt
                dose_rate_series[k] = Rxt.max()
        else:  # LQ pulses
            if in_window(t):
                dose_rate_series[k] = dose_rate

        u = np.clip(u + dt*du, 0.0, kin.K)

        if rb.mode == "lq_pulses":
            # apply survival at the end of a fraction
            for te in fr_end:
                if (t - dt) < te <= t + 1e-12:
                    d_map = sch.d_per_frac_Gy * beam
                    S = np.exp(-(alpha_eff*d_map + beta_eff*d_map**2))
                    u *= S
                    break

        total_mass[k] = u.sum()

        if save_hook is not None:
            save_hook(k, t, u, times[:k+1], total_mass[:k+1], dose_rate_series[:k+1], beam)

    return SimOutput(x=x, y=y, times=times, uT=u, pO2=pO2, H=H, beam=beam,
                     total_mass=total_mass, dose_rate_series=dose_rate_series)
