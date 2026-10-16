"""Fixture-free tests of the pulse sequence library: processing, decoupling, 2D sequences, isotopomers."""

import importlib.util

import numpy as np
import pytest

import respinach
from respinach.oracle import assert_matches

LIOUV = {"formalism": "sphten-liouv", "approximation": "none"}


def system(isotopes, bas=LIOUV, magnet=14.1, **inter):
    return respinach.basis(respinach.create({"isotopes": isotopes, "magnet": magnet}, inter), bas)


def spectrum_2d(fid, zerofill):
    return np.fft.fftshift(np.fft.fft2(fid, zerofill))


def peak_position(spectrum, axes):
    """(F1, F2) coordinates of the largest |spectrum| element; spectrum is (nF2, nF1)."""
    i2, i1 = np.unravel_index(np.argmax(np.abs(spectrum)), spectrum.shape)
    return axes[0][i1], axes[1][i2]


def test_apodisation():
    fid = np.ones(8, dtype=complex)
    out = respinach.apodisation(fid, [("exp", 6)])
    assert out[0] == 0.5 and np.allclose(out[1:], np.exp(-6 * np.linspace(0, 1, 8))[1:])
    assert respinach.apodisation(fid, [("none",)], fp_half=False)[0] == 1
    two = respinach.apodisation(np.ones((4, 6), dtype=complex), [("sqcos",), ("sin",)])
    expected = np.outer(np.cos(np.linspace(0, np.pi / 2, 4)) ** 2, np.sin(np.linspace(0, np.pi, 6)))
    expected[0, :] /= 2
    expected[:, 0] /= 2
    np.testing.assert_allclose(two, expected)
    assert np.allclose(
        respinach.apodisation(np.ones((4, 6), dtype=complex), [None, ("crisp",)])[:, 1], np.cos(np.pi / 2 / 5) ** 8
    )
    with pytest.raises(ValueError, match="one element per"):
        respinach.apodisation(fid, [("exp", 6), ("exp", 6)])
    with pytest.raises(ValueError, match="takes one parameter"):
        respinach.apodisation(fid, [("exp",)])


def test_decouple_and_homospoil():
    spin_system = system(["1H", "13C"], coupling={"scalar": [[0, 140.0], [0, 0]]})
    nmr = respinach.assume(spin_system, "nmr")
    H = respinach.hamiltonian(nmr)
    rho = respinach.state(spin_system, "L+", "all") + respinach.state(spin_system, ["L+", "Lz"], [0, 1])
    H_dec, rho_dec = respinach.decouple(spin_system, H, rho, ["13C"])
    carbon = spin_system.basis.basis[:, 1] > 0
    assert np.allclose(rho_dec[carbon], 0) and np.allclose(rho_dec[~carbon], rho[~carbon])
    Hd = np.asarray(H_dec)
    assert np.allclose(Hd[carbon], 0) and np.allclose(Hd[:, carbon], 0)
    # Decoupled acquisition: the proton doublet collapses to a singlet at the proton frequency
    pars = {
        "spins": ["1H"],
        "offset": 1200.0,
        "sweep": 2000.0,
        "npoints": 128,
        "rho0": respinach.state(spin_system, "L+", "1H"),
        "coil": respinach.state(spin_system, "L+", "1H"),
    }
    coupled = respinach.liquid(spin_system, respinach.acquire, pars, "nmr")
    decoupled = respinach.liquid(spin_system, respinach.acquire, {**pars, "decouple": ["13C"]}, "nmr")
    t = np.arange(128) / 2000.0
    omega = spin_system.basefrqs[0] * 0 + 2 * np.pi * 1200.0
    assert np.allclose(decoupled, decoupled[0] * np.exp(-1j * omega * t), rtol=1e-9)
    assert not np.allclose(coupled, decoupled)
    assert_matches(respinach.decouple(spin_system, H, rho, [1])[1], rho_dec, rtol=0, atol=0)
    same_L, same_rho = respinach.decouple(spin_system, H, rho, [])
    assert same_L is H and same_rho is rho
    kept = respinach.homospoil(spin_system, rho, "destroy")
    _, projs = respinach.operators.lin2lm(spin_system.basis.basis)
    assert np.allclose(kept[np.abs(projs).sum(axis=1) > 0], 0)
    # 'keep' retains states with zero Zeeman frequency: a heteronuclear L+L- is not one, a homonuclear one is
    assert (
        np.count_nonzero(respinach.homospoil(spin_system, respinach.state(spin_system, ["L+", "L-"], [0, 1]), "keep"))
        == 0
    )
    hh = system(["1H", "1H"])
    assert (
        np.count_nonzero(
            respinach.homospoil(
                hh, respinach.state(hh, ["L+", "L-"], [0, 1]) + respinach.state(hh, "L+", "all"), "keep"
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="zqc_flag"):
        respinach.homospoil(spin_system, rho, "whatever")


def test_refocus_and_dead_time():
    spin_system = system(["1H", "13C"], coupling={"scalar": [[0, 140.0], [0, 0]]})
    L = respinach.hamiltonian(respinach.assume(spin_system, "nmr"))
    dt = 1e-3
    stack = respinach.evolution(spin_system, L, respinach.state(spin_system, "L+", "1H"), dt, 5, "trajectory")
    refocused = respinach.evolution(spin_system, L, stack, dt, 0, "refocus")
    P = respinach.propagator(spin_system, L, dt)
    for k in range(6):
        expected = np.linalg.matrix_power(np.asarray(P), k) @ stack[:, k]
        assert_matches(refocused[:, k], expected, rtol=1e-10, atol=1e-14)
    pars = {
        "spins": ["1H"],
        "sweep": 2000.0,
        "npoints": 16,
        "rho0": respinach.state(spin_system, "L+", "1H"),
        "coil": respinach.state(spin_system, "L+", "1H"),
    }
    plain = respinach.liquid(spin_system, respinach.acquire, {**pars, "npoints": 20}, "nmr")
    delayed = respinach.liquid(spin_system, respinach.acquire, {**pars, "dead_time": 4 / 2000.0}, "nmr")
    assert_matches(delayed, plain[4:], rtol=1e-9, atol=0)


def test_cosy_cross_peaks_need_a_coupling():
    shifts, J = [1.0, 3.0], 12.0
    pars = {
        "angle": np.pi / 2,
        "offset": 1200.0,
        "sweep": 2000.0,
        "npoints": [64, 64],
        "spins": ["1H"],
        "zerofill": [256, 256],
    }
    coupled = system(["1H", "1H"], zeeman={"scalar": shifts}, coupling={"scalar": [[0, J], [0, 0]]})
    uncoupled = system(["1H", "1H"], zeeman={"scalar": shifts})
    spectra = {}
    for name, spin_system in (("coupled", coupled), ("uncoupled", uncoupled)):
        fid = respinach.liquid(spin_system, respinach.cosy, pars, "nmr")
        assert fid.shape == (64, 64)
        spectra[name] = spectrum_2d(respinach.apodisation(fid, [("sqcos",), ("sqcos",)]), (256, 256))
    from respinach.plotting import axis_2d

    f1, f2, _, _ = axis_2d(coupled, spectra["coupled"], {**pars, "axis_units": "ppm"})
    # Cross peak region: F1 at spin 1, F2 at spin 2

    def near(axis, value):
        return np.abs(axis - value) < 0.15

    cross = np.abs(spectra["coupled"])[np.ix_(near(f2, 3.0), near(f1, 1.0))].max()
    diagonal = np.abs(spectra["coupled"])[np.ix_(near(f2, 1.0), near(f1, 1.0))].max()
    assert cross > 0.1 * diagonal
    # Without a coupling only truncation sidelobes of the diagonal peaks (about 1 %) reach the cross-peak region
    assert np.abs(spectra["uncoupled"])[np.ix_(near(f2, 3.0), near(f1, 1.0))].max() < 0.02 * diagonal
    with pytest.raises(ValueError, match="angle"):
        respinach.liquid(coupled, respinach.cosy, {k: v for k, v in pars.items() if k != "angle"}, "nmr")


def test_noesy_cross_peaks_from_cross_relaxation():
    spin_system = system(
        ["1H", "1H"],
        zeeman={"scalar": [1.0, 3.0]},
        coordinates=[[0, 0, 0], [0, 0, 2.5]],
        relaxation=["redfield"],
        equilibrium="IME",
        rlx_keep="kite",
        temperature=298,
        tau_c=[[2e-9]],
    )
    pars = {
        "tmix": 0.3,
        "offset": 1200.0,
        "sweep": [2000.0, 2000.0],
        "npoints": [48, 48],
        "spins": ["1H"],
        "needs": ["rho_eq"],
        "zerofill": [256, 256],
        "axis_units": "ppm",
    }
    fid = respinach.liquid(spin_system, respinach.noesy, pars, "nmr")
    assert set(fid) == {"cos", "sin"} and fid["cos"].shape == (48, 48)
    cos = respinach.apodisation(fid["cos"], [("sqcos",), ("sqcos",)])
    sin = respinach.apodisation(fid["sin"], [("sqcos",), ("sqcos",)])
    f1_cos = np.real(np.fft.fftshift(np.fft.fft(cos, 256, axis=0), axes=0))
    f1_sin = np.real(np.fft.fftshift(np.fft.fft(sin, 256, axis=0), axes=0))
    spectrum = np.fft.fftshift(np.fft.fft(f1_cos - 1j * f1_sin, 256, axis=1), axes=1)
    from respinach.plotting import axis_2d

    f1, f2, _, _ = axis_2d(spin_system, spectrum, pars)

    def near(axis, value):
        return np.abs(axis - value) < 0.15

    diagonal = spectrum.real[np.ix_(near(f2, 1.0), near(f1, 1.0))]
    cross = spectrum.real[np.ix_(near(f2, 3.0), near(f1, 1.0))]
    assert np.abs(cross).max() > 0.05 * np.abs(diagonal).max()
    # Slow tumbling (omega tau_c >> 1): positive NOE cross peaks have the same sign as the diagonal
    assert np.sign(cross.flat[np.argmax(np.abs(cross))]) == np.sign(diagonal.flat[np.argmax(np.abs(diagonal))])
    with pytest.raises(ValueError, match="conflicts"):
        respinach.liquid(
            spin_system, respinach.noesy, {**pars, "rho0": respinach.state(spin_system, "Lz", "1H")}, "nmr"
        )


def test_hsqc_peak_at_the_pair_shifts():
    spin_system = system(
        ["1H", "13C", "1H"],
        magnet=5.9,
        zeeman={"scalar": [3.0, 40.0, 1.0]},
        coupling={"scalar": [[0, 140.0, 7.0], [0, 0, 0], [0, 0, 0]]},
    )
    pars = {
        "J": 140.0,
        "sweep": [3000.0, 2000.0],
        "offset": [2500.0, 500.0],
        "npoints": [48, 48],
        "zerofill": [256, 256],
        "spins": ["13C", "1H"],
        "decouple_f1": ["1H"],
        "decouple_f2": ["13C"],
        "axis_units": "ppm",
    }
    fid = respinach.liquid(spin_system, respinach.hsqc, pars, "nmr")
    assert set(fid) == {"pos", "neg"}
    pos = np.fft.fftshift(np.fft.fft(respinach.apodisation(fid["pos"], [("sqcos",), ("sqcos",)]), 256, axis=0), axes=0)
    neg = np.fft.fftshift(np.fft.fft(respinach.apodisation(fid["neg"], [("sqcos",), ("sqcos",)]), 256, axis=0), axes=0)
    spectrum = np.fft.fftshift(np.fft.fft(pos + np.conj(neg), 256, axis=1), axes=1)
    from respinach.plotting import axis_2d

    f1, f2, _, _ = axis_2d(spin_system, spectrum, pars)
    F1, F2 = peak_position(spectrum, (f1, f2))
    assert abs(F1 - 40.0) < 0.3 and abs(F2 - 3.0) < 0.05  # 13C shift in F1, bound proton in F2
    with pytest.raises(ValueError, match="two different isotopes"):
        respinach.liquid(spin_system, respinach.hsqc, {**pars, "spins": ["1H", "1H"], "offset": [500.0, 500.0]}, "nmr")


def test_kill_spin_and_dilute():
    spin_system = respinach.create(
        {"isotopes": ["1H", "13C", "13C"], "magnet": 5.9},
        {
            "zeeman": {"scalar": [3.0, 40.0, 50.0]},
            "coupling": {"scalar": [[0, 140.0, 0], [0, 0, 35.0], [0, 0, 0]]},
            "coordinates": [[0, 0, 0], [0, 0, 1.09], [0, 0, 2.6]],
        },
    )
    sub = respinach.dilute(spin_system, "13C")
    assert [s.isotopes for s in sub] == [("1H", "13C"), ("1H", "13C")]
    assert np.isclose(sub[1].zeeman_matrix[1][0, 0] / sub[1].basefrqs[1], 1 + 50e-6)
    assert np.isclose(np.trace(sub[0].coupling_matrix[0][1]) / 3, 2 * np.pi * 140.0)  # one-bond J kept
    assert np.isclose(np.trace(sub[1].coupling_matrix[0][1]), 0)  # only the dipolar part remains
    assert sub[0].basis is None and sub[0].chem.parts == ((0, 1),)
    pairs = respinach.dilute(spin_system, "13C", 2)
    assert len(pairs) == 1 and pairs[0].isotopes == ("1H", "13C", "13C")
    with pytest.raises(ValueError, match="not enough"):
        respinach.dilute(spin_system, "13C", 3)
    with pytest.raises(ValueError, match="bounds"):
        respinach.kill_spin(spin_system, [5])


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="plotly not installed")
def test_plot_2d_and_contours():
    from respinach.plotting import contspacing, plot_2d

    levels, pos, neg = contspacing(1.0, -0.5, (0.1, 0.5, 0.1, 0.5), 2, "both", 5)
    assert np.allclose(pos, 0.1 + 0.4 * np.linspace(0, 1, 5) ** 2) and np.allclose(
        neg, -0.05 - 0.2 * np.linspace(0, 1, 5) ** 2
    )
    assert np.all(np.diff(levels) > 0)
    assert contspacing(1.0, -0.5, (0.1, 0.5, 0.1, 0.5), 2, "positive", 5)[2].size == 0
    spin_system = system(["1H", "1H"], zeeman={"scalar": [1.0, 3.0]})
    spectrum = np.zeros((64, 32))
    spectrum[20, 10] = 1.0
    spectrum[40, 20] = -0.5
    fig = plot_2d(spin_system, spectrum, {"spins": ["1H"], "offset": 1200.0, "sweep": 2000.0, "axis_units": "ppm"})
    assert (
        len(fig.data) == 2 and fig.data[0].contours.end == 19 and fig.data[1].contours.end == 19
    )  # one trace per sign, 20 levels each
    assert fig.layout.xaxis.range[0] > fig.layout.xaxis.range[1]  # reversed axes
