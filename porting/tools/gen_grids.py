"""Convert Spinach's spherical averaging grids (kernel/grids/*.mat) to NumPy archives.

    python porting/tools/gen_grids.py /spinach/kernel/grids src/spinpy/grids

Each grid becomes <name>.npz with float64 arrays alphas, betas, gammas (rad)
and weights (summing to one). The grids are Spinach's (MIT licence); the
files are generated, do not edit them by hand.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

KEYS = ("alphas", "betas", "gammas", "weights")


def load(path: Path) -> dict:
    """Read a v7 .mat with SciPy or a v7.3 (HDF5) .mat with h5py."""
    try:
        grid = sio.loadmat(path)
        return {k: np.asarray(grid[k], dtype=float).reshape(-1) for k in KEYS}
    except NotImplementedError:
        with h5py.File(path, "r") as h5:
            return {k: np.asarray(h5[k][()], dtype=float).reshape(-1) for k in KEYS}


src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for path in sorted(src.glob("*.mat")):
    arrays = load(path)
    assert len({v.size for v in arrays.values()}) == 1, path
    assert abs(arrays["weights"].sum() - 1) < 1e-10, (path, arrays["weights"].sum())
    np.savez_compressed(dst / (path.stem + ".npz"), **arrays)
    print(path.stem, arrays["weights"].size)
