from __future__ import annotations
import os, glob
import imageio.v2 as imageio
from typing import List

def frames_to_gif(frames_dir: str, outfile: str, fps: int = 10) -> str:
    imgs = []
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    for f in files:
        imgs.append(imageio.imread(f))
    imageio.mimsave(outfile, imgs, fps=fps)
    return outfile
