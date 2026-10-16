"""Reader for oracle fixtures written by ``oracle/matlab/lib/oracle_write.m``.

A fixture case is a directory holding ``data.h5`` and ``meta.json``. Every
variable in ``data.h5`` is an HDF5 group with a ``kind`` attribute:

``dense``
    ``re`` and, if the ``complex`` attribute is 1, ``im`` datasets whose
    C-order shape equals the MATLAB shape stored in the ``shape`` attribute.
``sparse``
    0-based ``row``/``col`` int64 vectors and ``re``/``im`` value vectors;
    returned as ``scipy.sparse.csr_array``.
``cell``
    member variables named by the 0-based column-major linear index of each
    element; returned as a numpy object array with the MATLAB shape.

Datasets are absent for empty arrays and for sparse matrices with no stored
elements. MATLAB shapes are kept as they are (a column vector stays ``(n, 1)``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sps

DEFAULT_FIXTURE_ROOT = Path("/fixtures")


def fixture_root() -> Path:
    """Root of the fixture tree: ``$SPYNACH_FIXTURES`` or ``/fixtures``."""
    return Path(os.environ.get("SPYNACH_FIXTURES", DEFAULT_FIXTURE_ROOT))


def find_case(case: str, tag: str | None = None) -> Path | None:
    """Locate ``<root>/<tag>/<case>``; with no tag, the case must exist under one tag only.

    Returns None when the case is not present.
    """
    root = fixture_root()
    if tag is not None:
        path = root / tag / case
        return path if path.is_dir() else None
    hits = sorted(p for p in root.glob(f"*/{case}") if p.is_dir())
    if len(hits) > 1:
        raise LookupError(f"case {case!r} exists under several tags, pass tag=: {hits}")
    return hits[0] if hits else None


@dataclass(frozen=True)
class Fixture:
    """One fixture case. Index with HDF5 variable paths, e.g. ``fx["/ham/H"]``."""

    path: Path
    meta: dict[str, Any]

    @property
    def data_file(self) -> Path:
        return self.path / "data.h5"

    def __getitem__(self, name: str) -> Any:
        with h5py.File(self.data_file, "r") as h5:
            if name not in h5:
                raise KeyError(f"{name!r} not in {self.data_file}")
            return _read_variable(h5[name])

    def __contains__(self, name: str) -> bool:
        with h5py.File(self.data_file, "r") as h5:
            return name in h5 and "kind" in h5[name].attrs

    def keys(self) -> list[str]:
        """Paths of all top-level variables (cell members are not listed)."""
        with h5py.File(self.data_file, "r") as h5:
            return list(_variable_paths(h5))


def load_fixture(path: str | os.PathLike[str]) -> Fixture:
    path = Path(path)
    with open(path / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    if not (path / "data.h5").is_file():
        raise FileNotFoundError(path / "data.h5")
    return Fixture(path=path, meta=meta)


def _attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return _attr_str(value.item())
    return str(value)


def _attr_int(value: Any) -> int:
    return int(np.asarray(value).item())


def _variable_paths(group: h5py.Group) -> Iterator[str]:
    for member in group.values():
        if not isinstance(member, h5py.Group):
            continue
        if "kind" in member.attrs:
            yield member.name
        else:
            yield from _variable_paths(member)


def _read_variable(group: h5py.Group) -> Any:
    kind = _attr_str(group.attrs["kind"])
    shape = tuple(int(n) for n in np.atleast_1d(group.attrs["shape"]))
    if kind == "cell":
        out = np.empty(shape, dtype=object)
        for k in range(out.size):
            out[np.unravel_index(k, shape, order="F")] = _read_variable(group[str(k)])
        return out
    is_complex = bool(_attr_int(group.attrs["complex"]))
    dtype = np.complex128 if is_complex else np.float64
    if kind == "sparse":
        if _attr_int(group.attrs["nnz"]) == 0:
            return sps.csr_array(shape, dtype=dtype)
        vals = np.asarray(group["re"][()], dtype=np.float64).ravel()
        if is_complex:
            vals = vals + 1j * np.asarray(group["im"][()], dtype=np.float64).ravel()
        rows = np.asarray(group["row"][()], dtype=np.int64).ravel()
        cols = np.asarray(group["col"][()], dtype=np.int64).ravel()
        return sps.csr_array((vals.astype(dtype), (rows, cols)), shape=shape)
    if kind == "dense":
        if "re" not in group:
            return np.zeros(shape, dtype=dtype)
        data = np.asarray(group["re"][()], dtype=np.float64)
        if is_complex:
            data = data + 1j * np.asarray(group["im"][()], dtype=np.float64)
        if data.shape != shape:
            raise ValueError(f"{group.name}: stored shape {data.shape} != recorded shape {shape}")
        return data.astype(dtype, copy=False)
    raise ValueError(f"{group.name}: unknown kind {kind!r}")
