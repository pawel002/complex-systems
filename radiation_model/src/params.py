from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class Grid:
    L_mm: float = 50.0
    N: int = 100
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)

    def __post_init__(self) -> None:
        self.dx = self.L_mm / self.N
        self.x = np.linspace(0.0, self.L_mm, self.N)
        self.y = np.linspace(0.0, self.L_mm, self.N)

@dataclass
class TumorKinetics:
    D_mm2_per_day: float = 0.137     # ~50 mm^2/yr, within Rockne/Swanson range
    rho_per_day: float   = 0.0274    # ~10 /yr  (doubling ~25 d)
    K: float = 1.0                   # normalized carrying capacity

@dataclass
class Radiobiology:
    mode: Literal["pde_kill", "lq_pulses"] = "lq_pulses"
    # PDE-kill radiosensitivity coefficient (baseline α-like in Gy^-1)
    beta_rt_per_Gy: float = 0.06
    # LQ parameters (used if mode == "lq_pulses")
    alpha: float = 0.06              # Gy^-1  (Qi 2006)
    beta: float  = 0.006             # Gy^-2  (α/β ≈ 10 Gy)

    # Hypoxia model parameters (OMF)
    OER_max: float = 3.0
    K_m_mmHg: float = 3.0

@dataclass
class DoseSchedule:
    start_day: float = 10.0
    n_fractions: int = 30
    d_per_frac_Gy: float = 2.0
    frac_duration_day: float = 0.004  # ~6 min

@dataclass
class Beam:
    profile: Literal["uniform", "gaussian"] = "uniform"
    sigma_mm: float = 10.0

@dataclass
class VizConfig:
    frame_every: int = 5             # save a frame every N steps
    cmap: str = "viridis"

@dataclass
class SimConfig:
    T_days: float = 120.0
    dt_day: float = 0.05
    grid: Grid = field(default_factory=Grid)
    tumor: TumorKinetics = field(default_factory=TumorKinetics)
    rtd: Radiobiology = field(default_factory=Radiobiology)
    schedule: DoseSchedule = field(default_factory=DoseSchedule)
    beam: Beam = field(default_factory=Beam)
    viz: VizConfig = field(default_factory=VizConfig)

    def stability_ok(self) -> bool:
        dx = self.grid.dx
        D = self.tumor.D_mm2_per_day
        return self.dt_day <= (dx*dx) / (4.0*D)

    def recommended_dt(self) -> float:
        dx = self.grid.dx
        D = self.tumor.D_mm2_per_day
        return 0.5 * (dx*dx) / (4.0*D)
