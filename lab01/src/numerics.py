from __future__ import annotations
import numpy as np
from typing import Tuple

def laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    up = np.pad(u, 1, mode="edge")
    return (up[2:,1:-1] + up[:-2,1:-1] + up[1:-1,2:] + up[1:-1,:-2] - 4*up[1:-1,1:-1]) / (dx*dx)

def init_tumor_gaussian(x: np.ndarray, y: np.ndarray, L_mm: float, peak: float = 0.1, sigma_mm: float = 5.0) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing="ij")
    return peak * np.exp(-((X - L_mm/2)**2 + (Y - L_mm/2)**2) / (2*sigma_mm**2))

def beam_profile(x: np.ndarray, y: np.ndarray, L_mm: float, kind: str, sigma_mm: float) -> np.ndarray:
    if kind == "uniform":
        return np.ones((x.size, y.size))
    X, Y = np.meshgrid(x, y, indexing="ij")
    r2 = (X - L_mm/2)**2 + (Y - L_mm/2)**2
    g = np.exp(-r2 / (2*sigma_mm**2))
    return g / g.max()