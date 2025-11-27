import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy import sparse
from typing import List, Literal

from tqdm import trange

def simulate(
    adj: List[List[int]],
    coords: np.ndarray,
    steps: int,
    m: float,
    plot_every: int = 5,
    values_init: Literal["uniform", "random"] = "uniform",
    title: str = "Title",
    output_folder: str = "frames",
    linearize: float = 0,
    aggregate: int = 0,
    verbose: bool = False
) -> List[float]:
    
    if plot_every > 0: 
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        if verbose:
            print(f"Output folder '{output_folder}' created/cleared.")
    n = len(adj)
    
    data = []
    rows = []
    cols = []
    degrees = np.zeros(n)
    
    for u, neighbors in enumerate(adj):
        k = len(neighbors)
        degrees[u] = k if k > 0 else 1.0
        for v in neighbors:
            rows.append(u)
            cols.append(v)
            data.append(1)
            
    adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    if values_init == "uniform":
        x = np.random.uniform(0.01, 0.99, n)
    elif values_init == "random":
        x = np.random.rand(n)
    
    total_duration = steps + aggregate
    if verbose:
        print(f"Simulation started for N={n} nodes, T={total_duration} steps (Burn-in: {steps}, Agg: {aggregate}), m={m}")
    
    history_avg = []
    history_t = []
    
    aggregated_results = []

    iterator = trange(total_duration) if verbose else range(total_duration)
    for t in iterator:
        
        neighbor_sums = adj_matrix.dot(x)
        
        A = neighbor_sums / degrees
        x_next = m * A * (1 - A)
        
        if linearize < 0.0001:
            x = np.clip(x_next, 0, 1)
        else:
            v = x_next - x
            x = x + (1 - linearize) * v
            x = np.clip(x, 0, 1)
        
        current_mean = np.mean(x)
        history_avg.append(current_mean)
        history_t.append(t)
        
        if aggregate > 0 and t >= steps:
            aggregated_results.append(current_mean)
        
        if plot_every > 0 and t % plot_every == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(title, fontsize=16)
            
            sc = ax1.scatter(coords[:, 0], coords[:, 1], c=x, cmap='coolwarm', 
                             s=15, alpha=0.8, vmin=0, vmax=1)
            ax1.set_title(f"Graph State (t={t})")
            ax1.axis('off')
            plt.colorbar(sc, ax=ax1, label="Value $x_i$")

            mean_val = current_mean 
            std_val = np.std(x)
            
            ax2.hist(x, bins=50, range=(0, 1), color='gray', alpha=0.7, density=True)
            ax2.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax2.axvline(mean_val - std_val, color='orange', linestyle='--', label=f'Std: {std_val:.3f}')
            ax2.axvline(mean_val + std_val, color='orange', linestyle='--')
            
            ax2.set_title("Distribution of States $x_i$")
            ax2.set_xlabel("Value")
            ax2.set_ylabel("Density")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 10)
            ax2.legend(loc='upper right')

            ax3.scatter(history_t, history_avg, c='blue', s=5, alpha=0.6)
            
            if aggregate > 0:
                ax3.axvline(steps, color='green', linestyle='--', alpha=0.5, label='Agg Start')
                if t > steps:
                    ax3.axvspan(steps, total_duration, color='green', alpha=0.1)

            ax3.set_title("Evolution of Average Value")
            ax3.set_xlabel("Time Step (t)")
            ax3.set_ylabel("Mean $x_i$")
            ax3.set_xlim(0, total_duration)
            ax3.set_ylim(0, 1)     
            ax3.grid(True, linestyle=':', alpha=0.6)

            filename = os.path.join(output_folder, f"frame_{t:04d}.png")
            plt.tight_layout()
            plt.savefig(filename, dpi=100)
            plt.close(fig)

    if plot_every > 0 and verbose:   
        print(f"Simulation complete. Frames saved to /{output_folder}")
    
    return aggregated_results