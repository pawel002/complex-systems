from __future__ import annotations
import numpy as np
from typing import Tuple

Array = np.ndarray

def radial_pO2(N: int, p_center: float = 1.0, p_edge: float = 30.0) -> np.ndarray:
    """Simple phantom: hypoxic core → better oxygenated rim."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
    r = np.clip(np.sqrt(xx*xx + yy*yy), 0.0, 1.0)
    return p_center + (p_edge - p_center) * r

def omf_from_pO2(pO2: np.ndarray, OER_max: float = 3.0, K_m_mmHg: float = 3.0) -> np.ndarray:
    """
    Oxygen Modification Factor (≈ relative radiosensitivity) in [~1/OER_max, 1].
    Alper-Howard-Flanders-inspired monotone mapping.
    """
    return (1.0 + (OER_max - 1.0) * (pO2 / (pO2 + K_m_mmHg))) / OER_max
