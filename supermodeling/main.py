import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. HASTINGS-POWELL MODEL (ECOLOGY)
# ==========================================
def ecosystem_deriv(state, t, a1, b1, a2, b2, d1, d2):
    x, y, z = state
    # Ecological constraints: populations cannot be negative
    x = max(0, x); y = max(0, y); z = max(0, z)
    
    dx = x * (1 - x) - (a1 * x * y) / (1 + b1 * x)
    dy = (a1 * x * y) / (1 + b1 * x) - (a2 * y * z) / (1 + b2 * y) - d1 * y
    dz = (a2 * y * z) / (1 + b2 * y) - d2 * z
    return [dx, dy, dz]

# Ground Truth Parameters
P_TRUE = {'a1': 5.0, 'b1': 3.0, 'a2': 0.1, 'b2': 2.0, 'd1': 0.4, 'd2': 0.01}
x0 = [0.8, 0.2, 0.1]

# Time
dt = 0.5
t_train = np.arange(0, 400, dt)
t_test  = np.arange(0, 250, dt) # Prediction window

# Generate Ground Truth
truth_train = odeint(ecosystem_deriv, x0, t_train, args=tuple(P_TRUE.values()))
truth_test = odeint(ecosystem_deriv, truth_train[-1], t_test, args=tuple(P_TRUE.values()))

# ==========================================
# 2. EXPERIMENT ENGINE
# ==========================================
def run_scenario(quality_scale):
    """
    Runs the supermodel pipeline.
    quality_scale: Higher = worse standalone models (shorter training).
    """
    
    # 1. Create 3 Surrogate Models (Imperfect parameters)
    surrogates = []
    perturbations = [-1.0, 0.5, 1.0] # Different biases for each model
    
    for i in range(3):
        p = P_TRUE.copy()
        # Perturb 'a1' (predation) and 'd1' (mortality)
        p['a1'] += perturbations[i] * (4.0 * quality_scale)
        p['d1'] -= perturbations[i] * (0.2 * quality_scale)
        surrogates.append(p)

    # 2. Supermodel Dynamics Wrapper
    def sm_dynamics(states_flat, t, C):
        s1, s2, s3 = states_flat[0:3], states_flat[3:6], states_flat[6:9]
        
        # Derivatives
        d1 = np.array(ecosystem_deriv(s1, t, **surrogates[0]))
        d2 = np.array(ecosystem_deriv(s2, t, **surrogates[1]))
        d3 = np.array(ecosystem_deriv(s3, t, **surrogates[2]))
        
        # Coupling
        avg = (s1 + s2 + s3) / 3.0
        return np.concatenate([
            d1 + C * (avg - s1),
            d2 + C * (avg - s2),
            d3 + C * (avg - s3)
        ])

    # 3. Training Phase (Find C)
    def cost(params):
        C_val = params[0]
        init = np.concatenate([x0, x0, x0])
        traj = odeint(sm_dynamics, init, t_train, args=(C_val,))
        sm_avg = (traj[:, 0:3] + traj[:, 3:6] + traj[:, 6:9]) / 3.0
        return np.mean((sm_avg - truth_train)**2)

    res = minimize(cost, [1.0], bounds=[(0.0, 20.0)], method='L-BFGS-B')
    C_opt = res.x[0]

    # 4. Prediction Phase (Supermodel)
    start_coupled = np.concatenate([truth_train[-1]]*3)
    sm_traj_full = odeint(sm_dynamics, start_coupled, t_test, args=(C_opt,))
    sm_pred = (sm_traj_full[:, 0:3] + sm_traj_full[:, 3:6] + sm_traj_full[:, 6:9]) / 3.0
    
    # 5. Prediction Phase (Standalone Models - Uncoupled)
    # We run them individually to see where they would go without help
    m1_pred = odeint(ecosystem_deriv, truth_train[-1], t_test, args=tuple(surrogates[0].values()))
    m2_pred = odeint(ecosystem_deriv, truth_train[-1], t_test, args=tuple(surrogates[1].values()))
    m3_pred = odeint(ecosystem_deriv, truth_train[-1], t_test, args=tuple(surrogates[2].values()))

    # Calculate RMSE over time for the supermodel
    error_traj = np.sqrt(np.sum((sm_pred - truth_test)**2, axis=1))
    
    return {
        'sm': sm_pred, 
        'm1': m1_pred, 
        'm2': m2_pred, 
        'm3': m3_pred, 
        'error': error_traj, 
        'C': C_opt
    }

# ==========================================
# 3. RUN SIMULATION
# ==========================================
print("Running simulation (Medium Pre-training scenario)...")
# We use a medium scale (0.08) to make the standalone divergence visible but not instant
results_med = run_scenario(quality_scale=0.08)

# We also run a "Short" training scenario just to get the error curve for comparison
results_short = run_scenario(quality_scale=0.25)


# ==========================================
# 4. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(18, 8))

# --- PLOT 1: 3D Trajectories (The Path) ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# A. Ground Truth (Black)
ax1.plot(truth_test[:,0], truth_test[:,1], truth_test[:,2], 
         color='black', lw=2.5, label='Ground Truth')

# B. Standalone Models (Faint Dashed)
# These show the behavior WITHOUT supermodeling
ax1.plot(results_med['m1'][:,0], results_med['m1'][:,1], results_med['m1'][:,2], 
         color='red', ls='--', alpha=0.3, lw=1, label='Model 1 (Alone)')
ax1.plot(results_med['m2'][:,0], results_med['m2'][:,1], results_med['m2'][:,2], 
         color='green', ls='--', alpha=0.3, lw=1, label='Model 2 (Alone)')
ax1.plot(results_med['m3'][:,0], results_med['m3'][:,1], results_med['m3'][:,2], 
         color='blue', ls='--', alpha=0.3, lw=1, label='Model 3 (Alone)')

# C. Supermodel (Magenta)
ax1.plot(results_med['sm'][:,0], results_med['sm'][:,1], results_med['sm'][:,2], 
         color='magenta', lw=2.5, alpha=0.9, label=f'Supermodel (C={results_med["C"]:.1f})')

ax1.set_title("Prediction Trajectories: Supermodel vs Standalone", fontsize=14)
ax1.set_xlabel("Vegetation")
ax1.set_ylabel("Herbivore")
ax1.set_zlabel("Predator")
ax1.legend(loc='upper right', fontsize=9)
ax1.view_init(elev=30, azim=120) 

# --- PLOT 2: Quality Comparison (Error vs Time) ---
ax2 = fig.add_subplot(1, 2, 2)

# Compare the Error of the Short Training vs Medium Training Supermodels
ax2.plot(t_test, results_short['error'], 'r-', lw=2, label='Short Pre-training (Worse Models)')
ax2.plot(t_test, results_med['error'], 'g-', lw=2, label='Medium Pre-training (Better Models)')

# Add divergence threshold area
ax2.fill_between(t_test, results_short['error'], results_med['error'], color='green', alpha=0.1)

ax2.set_title("Impact of Pre-training Duration on Quality", fontsize=14)
ax2.set_xlabel("Prediction Time Horizon")
ax2.set_ylabel("RMSE (Distance from Truth)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()