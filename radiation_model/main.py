from __future__ import annotations
import argparse, os
from src.params import SimConfig, Radiobiology, TumorKinetics, DoseSchedule, Beam, Grid, VizConfig
from src.simulation import simulate
from src.viz import make_frame
from src.giftools import frames_to_gif

def main() -> None:
    ap = argparse.ArgumentParser(description="RT + hypoxia RD simulator (2-D)")
    ap.add_argument("--mode", choices=["pde_kill", "lq_pulses"], default="lq_pulses")
    ap.add_argument("--out", default="out1")
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--L", type=float, default=50.0)
    ap.add_argument("--D", type=float, default=0.137)   # mm^2/day
    ap.add_argument("--rho", type=float, default=0.0274)# 1/day
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--beta", type=float, default=0.006)
    ap.add_argument("--beta_rt", type=float, default=0.06)
    ap.add_argument("--start", type=float, default=10.0)
    ap.add_argument("--fx", type=int, default=30)
    ap.add_argument("--dpf", type=float, default=2.0)
    ap.add_argument("--dur", type=float, default=0.004)
    ap.add_argument("--beam", choices=["uniform","gaussian"], default="uniform")
    ap.add_argument("--beam_sigma", type=float, default=10.0)
    ap.add_argument("--frame_every", type=int, default=5)
    ap.add_argument("--gif_fps", type=int, default=10)
    args = ap.parse_args()

    out_dir = args.out
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cfg = SimConfig(
        T_days=args.T,
        dt_day=args.dt,
        grid=Grid(L_mm=args.L, N=args.N),
        tumor=TumorKinetics(D_mm2_per_day=args.D, rho_per_day=args.rho, K=1.0),
        rtd=Radiobiology(mode=args.mode, beta_rt_per_Gy=args.beta_rt, alpha=args.alpha, beta=args.beta),
        schedule=DoseSchedule(start_day=args.start, n_fractions=args.fx, d_per_frac_Gy=args.dpf, frac_duration_day=args.dur),
        beam=Beam(profile=args.beam, sigma_mm=args.beam_sigma),
        viz=VizConfig(frame_every=args.frame_every),
    )

    def hook(k, t, u, times, mass, dr, beam, max_u):
        if k % cfg.viz.frame_every == 0:
            make_frame(frames_dir, k, t, u, times, mass, dr, max_u, cmap=cfg.viz.cmap)

    print("Running simulation...")
    simulate(cfg, save_hook=hook)
    print("Simulation done. Making GIF...")
    gif_path = os.path.join(out_dir, "simulation.gif")
    frames_to_gif(frames_dir, gif_path, fps=args.gif_fps)
    print(f"GIF saved to: {gif_path}")

if __name__ == "__main__":
    main()
