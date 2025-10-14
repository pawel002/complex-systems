from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

Array = np.ndarray

def make_frame(
    out_dir: str,
    k: int,
    t_day: float,
    u: Array,
    times: Array,
    mass: Array,
    dose_series: Array,
    cmap: str = "viridis"
) -> str:
    """Create a figure with 3 panels: density image, dose-rate(t), cumulative mass(t)."""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 4.5), dpi=140)

    # Left: density
    ax0 = fig.add_subplot(1, 3, 1)
    im = ax0.imshow(u.T, origin="lower", interpolation="nearest", cmap=cmap)
    ax0.set_title(f"Cell density u(x,y)\nDay {t_day:.2f}")
    ax0.set_xticks([]); ax0.set_yticks([])
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("u (normalized)")

    # Top-right: dose rate over time
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(times, dose_series)
    ax1.set_title("Dose rate over time")
    ax1.set_xlabel("Day"); ax1.set_ylabel("Dose rate (Gy/day)")

    # Bottom-right: cumulative tumor mass
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(times, mass)
    ax2.set_title("Cumulative tumor mass")
    ax2.set_xlabel("Day"); ax2.set_ylabel("Sum(u)")

    fig.tight_layout()
    fname = os.path.join(out_dir, f"frame_{k:05d}.png")
    fig.savefig(fname)
    plt.close(fig)
    return fname
