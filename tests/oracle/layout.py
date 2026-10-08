"""Python writer that mirrors oracle_write.m, used to test the reader without MATLAB.

It reproduces the on-disk result of oracle_write.m, not its code path: MATLAB
permutes dense arrays before h5write so that the HDF5 C-order shape equals the
MATLAB shape, which in Python means writing the array as it is.
"""

import json

import h5py
import numpy as np
import scipy.sparse as sps


def write(h5: h5py.File, name: str, value) -> None:
    group = h5.create_group(name)
    if isinstance(value, np.ndarray) and value.dtype == object:
        group.attrs["shape"] = np.array(value.shape, dtype=np.int64)
        group.attrs["kind"] = "cell"
        for k in range(value.size):
            write(h5, f"{name}/{k}", value[np.unravel_index(k, value.shape, order="F")])
        return
    is_complex = np.iscomplexobj(value.data if sps.issparse(value) else value)
    group.attrs["shape"] = np.array(value.shape, dtype=np.int64)
    group.attrs["complex"] = np.int64(is_complex)
    if sps.issparse(value):
        coo = sps.coo_array(value)
        order = np.lexsort((coo.row, coo.col))  # MATLAB find() is column-major
        rows, cols, vals = coo.row[order], coo.col[order], coo.data[order]
        group.attrs["kind"] = "sparse"
        group.attrs["nnz"] = np.int64(vals.size)
        if vals.size:
            group["row"] = rows.astype(np.int64)[None, :]
            group["col"] = cols.astype(np.int64)[None, :]
            group["re"] = vals.real[None, :]
            if is_complex:
                group["im"] = vals.imag[None, :]
        return
    value = np.asarray(value)
    group.attrs["kind"] = "dense"
    if value.size:
        group["re"] = value.real.astype(np.float64)
        if is_complex:
            group["im"] = value.imag.astype(np.float64)


def write_case(case_dir, variables: dict, meta: dict) -> None:
    case_dir.mkdir(parents=True)
    with h5py.File(case_dir / "data.h5", "w") as h5:
        for name, value in variables.items():
            write(h5, name, value)
    (case_dir / "meta.json").write_text(json.dumps(meta))
