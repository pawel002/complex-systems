from __future__ import annotations
import numpy as np

def weekday_fractions(start_day: float, n: int) -> np.ndarray:
    """Mon-Fri fractions (no calendar), two-day gap after each 5 fx."""
    times = []
    d = start_day
    for i in range(n):
        times.append(d)
        d += 1.0
        if (i+1) % 5 == 0:
            d += 2.0
    return np.array(times)
