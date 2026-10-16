"""Parity of the pulse sequence library against Spinach fixtures (FR-007).

Each stage of oracle/matlab/dumps/dump_seq.m is recomputed with respinach and
compared with the stored MATLAB result; the case definitions below must
stay identical to the dump script's. Tests skip when the fixtures are not
available. 2-D FIDs are (npoints[1], npoints[0]) arrays in both.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import respinach
from respinach.oracle import assert_matches, find_case, load_fixture

LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}
FID_TOL = dict(rtol=1e-9, atol=0)
EXACT = dict(rtol=1e-14, atol=0)


def dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def column(x):
    return dense(x).reshape(-1)


def fixture(name):
    path = find_case(name)
    if path is None:
        pytest.skip(f"fixture {name} not available")
    return load_fixture(path)


def test_cosy_derome_and_processing():
    fx = fixture("seq_cosy_derome")
    ss = respinach.basis(
        respinach.create(
            {"isotopes": ["1H"] * 3, "magnet": 16.1},
            {"zeeman": {"scalar": [3.70, 3.92, 4.50]}, "coupling": {"scalar": [[0, 10, 4], [0, 0, 12], [0, 0, 0]]}},
        ),
        LIOUV,
    )
    pars = {
        "angle": np.pi / 2,
        "offset": 2800,
        "sweep": 700,
        "npoints": [64, 64],
        "zerofill": [128, 128],
        "spins": ["1H"],
    }
    fid = respinach.liquid(ss, respinach.cosy, pars, "nmr")
    assert_matches(fid, dense(fx["/fid/cosy"]), **FID_TOL, name="cosy")
    apod = respinach.apodisation(fid, [("sqcos",), ("sqcos",)])
    assert_matches(apod, dense(fx["/fid/cosy_apodised"]), **FID_TOL, name="cosy apodised")
    assert_matches(
        np.fft.fftshift(np.fft.fft2(apod, (128, 128))), dense(fx["/spectrum/cosy"]), **FID_TOL, name="cosy spectrum"
    )
    v = np.arange(1, 17) + 1j * np.arange(16, 0, -1)
    A = np.arange(1, 49).reshape(8, 6).T + 1j  # MATLAB reshape(1:48,[6 8]) is column-major
    assert_matches(respinach.apodisation(v, [("exp", 6)]), column(fx["/apod/exp6"]), **EXACT)
    assert_matches(respinach.apodisation(v, [("gauss", 4)], fp_half=False), column(fx["/apod/gauss4_nohalf"]), **EXACT)
    assert_matches(respinach.apodisation(v, [("crisp",)]), column(fx["/apod/crisp"]), **EXACT)
    assert_matches(respinach.apodisation(v, [("kaiser", 3)]), column(fx["/apod/kaiser3"]), rtol=1e-12, atol=0)
    assert_matches(respinach.apodisation(A, [("sqcos",), ("sin",)]), dense(fx["/apod/matrix_sqcos_sin"]), **EXACT)
    assert_matches(respinach.apodisation(A, [("none",), ("cos",)]), dense(fx["/apod/matrix_none_cos"]), **EXACT)
    nmr = respinach.assume(ss, "nmr")
    H = respinach.frqoffset(ss, respinach.hamiltonian(nmr), pars)
    rho = respinach.state(ss, "L+", "1H")
    stack = respinach.evolution(ss, H, None, rho, 1e-3, 7, "trajectory")
    assert_matches(stack, dense(fx["/evo/trajectory"]), **FID_TOL, name="trajectory")
    assert_matches(
        respinach.evolution(ss, H, None, stack, 1e-3, 0, "refocus"),
        dense(fx["/evo/refocus"]),
        **FID_TOL,
        name="refocus",
    )
    apars = {"spins": ["1H"], "offset": 2800, "sweep": 700, "npoints": 32, "dead_time": 3e-3, "rho0": rho, "coil": rho}
    assert_matches(
        respinach.liquid(ss, respinach.acquire, apars, "nmr"), column(fx["/fid/acquire_dead_time"]), **FID_TOL
    )
    assert_matches(respinach.homospoil(ss, stack.T, "destroy").T, dense(fx["/states/homospoil_destroy"]), **EXACT)
    assert_matches(respinach.homospoil(ss, stack.T, "keep").T, dense(fx["/states/homospoil_keep"]), **EXACT)


def test_noesy_pair():
    fx = fixture("seq_noesy_pair")
    ss = respinach.basis(
        respinach.create(
            {"isotopes": ["1H", "1H"], "magnet": 14.1},
            {
                "zeeman": {"scalar": [1.0, 3.0]},
                "coordinates": [[0, 0, 0], [0, 0, 2.5]],
                "relaxation": ["redfield"],
                "equilibrium": "IME",
                "rlx_keep": "kite",
                "temperature": 298,
                "tau_c": [[2e-9]],
            },
        ),
        LIOUV,
    )
    pars = {
        "tmix": 0.3,
        "offset": 1200,
        "sweep": [2000, 2000],
        "npoints": [48, 48],
        "zerofill": [128, 128],
        "spins": ["1H"],
        "needs": ["rho_eq"],
    }
    assert_matches(respinach.equilibrium(ss), column(fx["/states/rho_eq"]), **FID_TOL, name="rho_eq")
    fid = respinach.liquid(ss, respinach.noesy, pars, "nmr")
    assert_matches(fid["cos"], dense(fx["/fid/noesy_cos"]), **FID_TOL, name="noesy cos")
    assert_matches(fid["sin"], dense(fx["/fid/noesy_sin"]), **FID_TOL, name="noesy sin")
    old = respinach.liquid(ss, respinach.noesy, {**pars, "oldschool": True}, "nmr")
    assert_matches(old["cos"], dense(fx["/fid/noesy_oldschool_cos"]), **FID_TOL, name="noesy oldschool")


def test_hsqc_with_dilution():
    fx = fixture("seq_hsqc_dilute")
    ss = respinach.create(
        {"isotopes": ["1H", "13C", "13C", "1H"], "magnet": 5.9},
        {
            "zeeman": {"scalar": [3.0, 40.0, 50.0, 1.0]},
            "coupling": {"scalar": [[0, 140, 0, 7], [0, 0, 35, 0], [0, 0, 0, 150], [0, 0, 0, 0]]},
        },
    )
    pars = {
        "J": 140,
        "sweep": [3000, 2000],
        "offset": [1000, 500],
        "npoints": [32, 32],
        "zerofill": [128, 128],
        "spins": ["13C", "1H"],
        "decouple_f1": ["1H"],
        "decouple_f2": ["13C"],
    }
    subsystems = respinach.dilute(ss, "13C")
    assert len(subsystems) == fx.meta["inputs"]["nsub"]
    spectrum = np.zeros((128, 128), dtype=complex)
    for n, sub in enumerate(subsystems, start=1):
        sub = respinach.basis(sub, LIOUV)
        np.testing.assert_array_equal(sub.basis.basis, dense(fx[f"/sub/basis_{n}"]))
        fid = respinach.liquid(sub, respinach.hsqc, pars, "nmr")
        assert_matches(fid["pos"], dense(fx[f"/fid/hsqc_pos_{n}"]), **FID_TOL, name=f"hsqc pos {n}")
        assert_matches(fid["neg"], dense(fx[f"/fid/hsqc_neg_{n}"]), **FID_TOL, name=f"hsqc neg {n}")
        pos = np.fft.fftshift(
            np.fft.fft(respinach.apodisation(fid["pos"], [("sqcos",), ("sqcos",)]), 128, axis=0), axes=0
        )
        neg = np.fft.fftshift(
            np.fft.fft(respinach.apodisation(fid["neg"], [("sqcos",), ("sqcos",)]), 128, axis=0), axes=0
        )
        spectrum += np.fft.fftshift(np.fft.fft(pos + np.conj(neg), 128, axis=1), axes=1)
    assert_matches(spectrum, dense(fx["/spectrum/hsqc"]), **FID_TOL, name="hsqc spectrum")
    full = respinach.basis(ss, LIOUV)
    H = respinach.hamiltonian(respinach.assume(full, "nmr"))
    rho = respinach.state(full, "L+", "1H") + respinach.state(full, ["L+", "Lz"], [0, 1])
    assert_matches(H, fx["/decouple/H"], rtol=1e-12, atol=0, name="H")
    H_dec, rho_dec = respinach.decouple(full, H, rho, ["13C"])
    assert_matches(H_dec, fx["/decouple/H_dec"], rtol=1e-12, atol=0, name="H decoupled")
    assert_matches(rho_dec, column(fx["/decouple/rho_dec"]), **EXACT, name="rho decoupled")
