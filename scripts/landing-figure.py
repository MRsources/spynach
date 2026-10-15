"""Render the hero figure of the landing page (index.html).

The figure is the sparsity pattern of the Liouvillian respinach builds for
examples/nmr_liquids/noesy_strychnine.py: the Hamiltonian commutation
superoperator plus i times the Redfield relaxation superoperator of the 22
protons of strychnine in the IK-2 basis. Nonzeros are binned into pixels;
the pixel brightness grows with the logarithm of the count. The greyscale
PNG is a luminance mask: index.html embeds it as a data URI inside an
inline SVG <mask> over a rectangle filled with the page's accent colour,
so the figure follows the light and dark themes and needs no file fetch
(CSS mask images are CORS-fetched and fail for pages opened from file://).

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python scripts/landing-figure.py

writes assets/liouvillian-strychnine.png, replaces the data URI in
index.html, and prints the matrix statistics quoted in the figure caption.
It also writes alternative renderings to assets/ for later selection:
``*-third.png`` (the first third of the states, top-left block, same
mask style), ``*-tenth-<start>.png`` (a diagonal window of a tenth of the
states beginning at state ``<start>``; by default the first tenth,
``--tenth-start`` chooses other windows, for example 1100 or 3213 where
Redfield blocks sit along the diagonal) and ``*-density.png`` variants of each (opaque
colour-mapped density images, ``--cmap`` selects the Matplotlib colormap,
``--background`` the colour of empty bins).
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from matplotlib import colormaps
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


def bin_counts(matrix: sps.csr_array, pixels: int = PIXELS) -> np.ndarray:
    """Number of nonzeros falling into each of ``pixels x pixels`` bins."""
    n = matrix.shape[0]
    rows, cols = matrix.nonzero()
    counts = np.zeros((pixels, pixels))
    np.add.at(counts, (rows * pixels // n, cols * pixels // n), 1)
    return counts


def render(matrix: sps.csr_array, pixels: int = PIXELS, thicken: int = THICKEN) -> Image.Image:
    """Greyscale luminance mask: brightness grows with the log of the bin count."""
    counts = bin_counts(matrix, pixels)
    weight = np.log1p(counts) / np.log1p(counts.max())
    alpha = np.where(counts > 0, 0.45 + 0.55 * weight, 0.0)
    for shift in range(1, thicken + 1):
        for axis in (0, 1):
            for direction in (shift, -shift):
                alpha = np.maximum(alpha, 0.55 * np.roll(alpha, direction, axis=axis))
    return Image.fromarray((255 * np.clip(alpha, 0, 1)).astype(np.uint8), "L")


def render_density(matrix: sps.csr_array, cmap: str, background: str, pixels: int = PIXELS,
                   thicken: int = THICKEN) -> Image.Image:
    """Opaque colour-mapped density: log of the bin count through a Matplotlib colormap, empty bins in ``background``."""
    counts = bin_counts(matrix, pixels)
    for shift in range(1, thicken + 1):
        for axis in (0, 1):
            for direction in (shift, -shift):
                counts = np.maximum(counts, 0.6 * np.roll(counts, direction, axis=axis))
    weight = np.log1p(counts) / np.log1p(counts.max())
    rgb = (255 * colormaps[cmap](0.12 + 0.88 * weight)[..., :3]).astype(np.uint8)
    bg = np.array([int(background.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)
    rgb[counts == 0] = bg
    return Image.fromarray(rgb, "RGB")


def first_third(matrix: sps.csr_array) -> sps.csr_array:
    """The top-left block covering the first third of the states."""
    k = matrix.shape[0] // 3
    return sps.csr_array(matrix[:k, :k])


def tenth(matrix: sps.csr_array, start: int) -> sps.csr_array:
    """A diagonal block covering a tenth of the states, beginning at ``start``."""
    k = matrix.shape[0] // 10
    return sps.csr_array(matrix[start:start + k, start:start + k])


def embed(image: Image.Image, page: Path) -> int:
    """Replace the data URI of the hero mask image in index.html; returns the encoded size in bytes."""
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    html = page.read_text()
    pattern = re.compile(r'(<image id="hero-mask-image" [^>]*href=")[^"]*(")')
    if not pattern.search(html):
        raise SystemExit(f"{page}: hero mask image element not found")
    page.write_text(pattern.sub(lambda m: m.group(1) + uri + m.group(2), html, count=1))
    return len(uri)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap of the density images (default viridis)")
    parser.add_argument("--background", default="#15191c", help="colour of empty bins in the density images")
    parser.add_argument("--tenth-start", type=int, nargs="*", default=[0],
                        help="first state of each tenth-of-the-states diagonal window (default 0, the first tenth)")
    parser.add_argument("--no-embed", action="store_true", help="do not rewrite the data URI in index.html")
    args = parser.parse_args(argv)
    assets = ROOT / "assets"
    L = liouvillian()
    n = L.shape[0]
    image = render(L)
    image.save(assets / "liouvillian-strychnine.png", optimize=True)
    print(f"liouvillian-strychnine.png: {n} states, {L.nnz} nonzeros, density {100 * L.nnz / n**2:.3f}%")
    if not args.no_embed:
        size = embed(image, ROOT / "index.html")
        print(f"embedded in index.html as a {size / 1024:.0f} KB data URI")
    third = first_third(L)
    render(third).save(assets / "liouvillian-strychnine-third.png", optimize=True)
    print(f"liouvillian-strychnine-third.png: first {third.shape[0]} states, {third.nnz} nonzeros")
    blocks = {"third": third}
    for start in args.tenth_start:
        blocks[f"tenth-{start}"] = tenth(L, start)
        render(blocks[f"tenth-{start}"]).save(assets / f"liouvillian-strychnine-tenth-{start}.png", optimize=True)
        print(f"liouvillian-strychnine-tenth-{start}.png: states {start} to {start + L.shape[0] // 10 - 1}, "
              f"{blocks[f'tenth-{start}'].nnz} nonzeros")
    for suffix, matrix in [("", L)] + [(f"-{k}", m) for k, m in blocks.items()]:
        name = f"liouvillian-strychnine{suffix}-density.png"
        render_density(matrix, args.cmap, args.background).save(assets / name, optimize=True)
        print(f"{name}: {args.cmap} on {args.background}")


if __name__ == "__main__":
    main()
