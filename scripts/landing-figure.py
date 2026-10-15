"""Render the hero figure of the landing page (index.html).

The figure is the sparsity pattern of the Liouvillian respinach builds for
examples/nmr_liquids/noesy_strychnine.py: the Hamiltonian commutation
superoperator plus i times the Redfield relaxation superoperator of the 22
protons of strychnine in the IK-2 basis. Nonzeros are binned into pixels;
the pixel alpha grows with the logarithm of the count. Only the alpha
channel matters: index.html uses the PNG as a CSS mask filled with the
page's accent colour, so the figure follows the light and dark themes.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python scripts/landing-figure.py

writes assets/liouvillian-strychnine.png and prints the matrix statistics
quoted in the figure caption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import respinach  # noqa: E402
from examples.nmr_liquids.noesy_strychnine import strychnine  # noqa: E402
from respinach.assume import assume  # noqa: E402
from respinach.hamiltonian import hamiltonian  # noqa: E402

PIXELS = 1200     # image side; rendered at about 500 CSS pixels, so this is sharp on high-density displays
THICKEN = 1      # grow each occupied pixel by this many pixels so single-element lines stay visible


def liouvillian() -> sps.csr_array:
    """The generator of the strychnine NOESY example (same setup as the example, no simulation)."""
    sys_, inter = strychnine(["1H"])
    sys_["magnet"] = 5.9
    sys_["tols"] = {"prox_cutoff": 4.0}
    inter["relaxation"] = ["redfield"]
    inter["equilibrium"] = "IME"
    inter["temperature"] = 298
    inter["rlx_keep"] = "kite"
    inter["tau_c"] = [[200e-12]]
    bas = {"formalism": "sphten-liouv", "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 3}
    spin_system = respinach.basis(respinach.create(sys_, inter), bas)
    H = hamiltonian(assume(spin_system, "nmr"), "comm")
    R = respinach.relaxation(spin_system)
    L = sps.csr_array(H + 1j * R)
    L.eliminate_zeros()
    return L


def render(matrix: sps.csr_array, pixels: int = PIXELS, thicken: int = THICKEN) -> Image.Image:
    n = matrix.shape[0]
    rows, cols = matrix.nonzero()
    counts = np.zeros((pixels, pixels))
    np.add.at(counts, (rows * pixels // n, cols * pixels // n), 1)
    weight = np.log1p(counts) / np.log1p(counts.max())
    alpha = np.where(counts > 0, 0.45 + 0.55 * weight, 0.0)
    for shift in range(1, thicken + 1):
        for axis in (0, 1):
            for direction in (shift, -shift):
                alpha = np.maximum(alpha, 0.55 * np.roll(alpha, direction, axis=axis))
    mask = (255 * np.clip(alpha, 0, 1)).astype(np.uint8)
    rgba = np.zeros((pixels, pixels, 4), np.uint8)
    rgba[..., :3] = (0x0F, 0x6E, 0x63)   # the light-theme accent, for viewers that ignore the mask
    rgba[..., 3] = mask
    return Image.fromarray(rgba, "RGBA")


def main() -> None:
    L = liouvillian()
    n = L.shape[0]
    out = ROOT / "assets" / "liouvillian-strychnine.png"
    render(L).save(out, optimize=True)
    print(f"{out}: {n} states, {L.nnz} nonzeros, density {100 * L.nnz / n**2:.3f}%")


if __name__ == "__main__":
    main()
