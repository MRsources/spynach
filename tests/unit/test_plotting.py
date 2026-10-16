"""Frequency axes and plots (spynach.plotting)."""

import numpy as np
import pytest

import spynach
from spynach.isotopes import spin
from spynach.plotting import axis_1d, ft_axis, plot_1d, plot_fid


@pytest.fixture(scope="module")
def spin_system():
    return spynach.create({"isotopes": ["1H", "13C"], "magnet": 14.1}, {})


@pytest.mark.parametrize("npoints", [8, 9, 1024, 1025])
def test_ft_axis_is_the_fftshifted_frequency_grid(npoints):
    expected = 250 + np.fft.fftshift(np.fft.fftfreq(npoints, 1 / 500))
    np.testing.assert_allclose(ft_axis(250, 500, npoints), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("args", [(0, -1, 8), (0, 10, 2), (0, 10, 8.5), (1j, 10, 8)])
def test_ft_axis_rejects_bad_input(args):
    with pytest.raises(ValueError):
        ft_axis(*args)


def test_axis_1d_units(spin_system):
    base = {"spins": ["13C"], "offset": 3000, "sweep": 2000, "zerofill": 64}
    hz, label = axis_1d(spin_system, {**base, "axis_units": "Hz"})
    np.testing.assert_allclose(hz, ft_axis(3000, 2000, 64))
    assert label == "13C offset frequency / Hz"

    khz, _ = axis_1d(spin_system, {**base, "axis_units": "kHz"})
    np.testing.assert_allclose(khz, 1e-3 * hz)

    ppm, label = axis_1d(spin_system, {**base, "axis_units": "ppm"})
    carrier = spin("13C")[0] * 14.1 / (2 * np.pi)
    np.testing.assert_allclose(ppm, 1e6 * hz / carrier)
    assert label == "13C chemical shift / ppm"

    points, _ = axis_1d(spin_system, {**base, "axis_units": "points"})
    np.testing.assert_array_equal(points, np.arange(1, 65))


def test_axis_1d_two_element_sweep(spin_system):
    ax, _ = axis_1d(spin_system, {"spins": ["1H"], "sweep": [-100, 100], "zerofill": 5, "axis_units": "Hz"})
    np.testing.assert_allclose(ax, [-100, -50, 0, 50, 100])
    with pytest.raises(ValueError):
        axis_1d(spin_system, {"spins": ["1H"], "sweep": [-100, 100], "offset": 0, "zerofill": 5, "axis_units": "Hz"})


def test_axis_1d_rejects_unknown_units(spin_system):
    with pytest.raises(ValueError):
        axis_1d(spin_system, {"spins": ["1H"], "offset": 0, "sweep": 100, "zerofill": 8, "axis_units": "furlongs"})


def test_plot_1d_follows_spinach_defaults(spin_system):
    pytest.importorskip("plotly")
    spectrum = np.random.default_rng(0).normal(size=256)
    fig = plot_1d(spin_system, spectrum, {"spins": ["1H"], "offset": 600, "sweep": 1000})
    (trace,) = fig.data
    ppm, _ = axis_1d(spin_system, {"spins": ["1H"], "offset": 600, "sweep": 1000, "zerofill": 256, "axis_units": "ppm"})
    np.testing.assert_allclose(trace.x, ppm)
    np.testing.assert_array_equal(trace.y, spectrum)
    assert fig.layout.xaxis.range == (ppm.max(), ppm.min())  # invert_axis=1
    assert "<sup>1</sup>H chemical shift / ppm" == fig.layout.xaxis.title.text
    assert not fig.layout.showlegend


def test_plot_1d_overlays_and_splits_complex_data(spin_system):
    pytest.importorskip("plotly")
    pars = {"spins": ["1H"], "offset": 0, "sweep": 1000, "axis_units": "Hz", "invert_axis": 0}
    spectra = np.ones((128, 2)) * [1, 2]
    fig = plot_1d(spin_system, spectra, pars, names=["a", "b"])
    assert [t.name for t in fig.data] == ["a", "b"]
    assert fig.layout.showlegend
    assert fig.layout.xaxis.range[0] < fig.layout.xaxis.range[1]

    fig = plot_1d(spin_system, np.ones(128) * (1 + 2j), pars)
    assert [t.name for t in fig.data] == ["real", "imag"]
    np.testing.assert_array_equal(fig.data[1].y, 2)


def test_plot_1d_rejects_mismatch_and_unported_options(spin_system):
    pytest.importorskip("plotly")
    pars = {"spins": ["1H"], "offset": 0, "sweep": 1000}
    with pytest.raises(ValueError, match="zerofill"):
        plot_1d(spin_system, np.ones(128), {**pars, "zerofill": 256})
    with pytest.raises(NotImplementedError):
        plot_1d(spin_system, np.ones(128), {**pars, "derivative": 1})
    with pytest.raises(ValueError):
        plot_1d(spin_system, [np.ones(128), np.ones(64)], pars)


def test_plot_fid_time_axis():
    pytest.importorskip("plotly")
    fig = plot_fid(np.exp(-np.arange(100) / 20) * (1 + 0j), {"sweep": 500})
    assert [t.name for t in fig.data] == ["real", "imag"]
    np.testing.assert_allclose(fig.data[0].x, 1e3 * np.arange(100) / 500)  # ms
    assert fig.layout.xaxis.title.text == "time / ms"


def test_plot_2d_levels_and_binning():
    from spynach.plotting import _bin_extremes, _level_index, contspacing, plot_2d

    # the level map sends every contour level to its index, in between it is monotone
    levels = np.array([0.1, 0.4, 1.0])
    np.testing.assert_allclose(_level_index(levels, levels), [0, 1, 2])
    np.testing.assert_allclose(_level_index(np.array([0.25, 0.7]), levels), [0.5, 1.5])
    assert _level_index(np.array([-5.0]), levels) <= -1 and _level_index(np.array([50.0]), levels) >= 3
    neg = contspacing(1.0, -1.0, (0.1, 0.5, 0.1, 0.5), 2, "both", 4)[2]  # descending values
    np.testing.assert_allclose(_level_index(neg, neg), np.arange(4))
    np.testing.assert_allclose(_level_index(np.array([0.5]), np.array([0.5])), [0])

    # binning keeps the extreme of each bin (sign included) and averages the axis
    spectrum = np.zeros((10, 3))
    spectrum[3, 0] = -2.0
    spectrum[2, 0] = 1.5
    spectrum[9, 2] = 0.7
    reduced, axis = _bin_extremes(spectrum, np.arange(10.0), 4, axis=0)
    assert reduced.shape == (4, 3) and reduced[1, 0] == -2.0 and reduced[3, 2] == 0.7
    np.testing.assert_allclose(axis, [1, 4, 7, 9])
    same, same_axis = _bin_extremes(spectrum, np.arange(10.0), 10, axis=0)
    assert same is spectrum and same_axis is not None

    pytest.importorskip("plotly")
    spin_system = spynach.create({"isotopes": ["1H", "1H"], "magnet": 14.1}, {"zeeman": {"scalar": [1.0, 3.0]}})
    pars = {"spins": ["1H"], "offset": 1200.0, "sweep": [2000.0, 2000.0], "npoints": [8, 8], "zerofill": [64, 64]}
    z = np.zeros((2000, 1500))
    z[500, 300] = 1.0
    z[1200, 900] = -0.5
    fig = plot_2d(spin_system, z, pars, ncont=20, delta=(0.05, 0.5, 0.05, 0.5), k=2, ncol=256, m=6, signs="both")
    assert len(fig.data) == 2  # one trace per sign
    assert (
        fig.data[0].z.shape == (500, 500) and fig.data[0].x.size == 500 and fig.data[0].y.size == 500
    )  # 2000/4, 1500/3
    assert fig.data[0].contours.end == 19 and fig.data[1].contours.end == 19
    fig_full = plot_2d(
        spin_system,
        z[450:550, 250:350],
        pars,
        ncont=20,
        delta=(0.05, 0.5, 0.05, 0.5),
        k=2,
        ncol=256,
        m=6,
        signs="positive",
        max_points=None,
    )
    assert len(fig_full.data) == 1 and fig_full.data[0].z.shape == (100, 100)


def test_time_series_and_trajan_plots():
    pytest.importorskip("plotly")
    from spynach.plotting import plot_time_series, plot_trajan

    t = np.linspace(0, 1, 5)
    fig = plot_time_series(t, [np.ones(5), np.arange(5.0)], names=["a", "b"], xlabel="time (s)", ylabel="y")
    assert len(fig.data) == 2 and fig.layout.xaxis.title.text == "time (s)"
    many = plot_time_series(t, np.ones((12, 5)))
    assert len(many.data) == 12 and len({tr.line.color for tr in many.data}) == 12  # beyond the fixed palette
    with pytest.raises(ValueError, match="as long as"):
        plot_time_series(t, np.ones(4))
    spin_system_liouv = spynach.basis(
        spynach.create({"isotopes": ["1H", "13C"], "magnet": 14.1}, {}),
        {"formalism": "sphten-liouv", "approximation": "none"},
    )
    traj = np.stack(
        [spynach.state(spin_system_liouv, ["Lz"], [0]), spynach.state(spin_system_liouv, ["L+"], [1])], axis=1
    )
    fig = plot_trajan(spin_system_liouv, traj, "coherence_order", np.array([0.0, 0.5]))
    assert len(fig.data) == 5 and fig.layout.title.text == "coherence orders"
    assert plot_trajan(spin_system_liouv, traj, "total_each_spin").layout.xaxis.title.text == "trajectory point"
