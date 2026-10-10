"""Frequency axes and plots (spinpy.plotting)."""

import numpy as np
import pytest

import spinpy
from spinpy.isotopes import spin
from spinpy.plotting import axis_1d, ft_axis, plot_1d, plot_fid


@pytest.fixture(scope="module")
def ss():
    return spinpy.create({"isotopes": ["1H", "13C"], "magnet": 14.1}, {})


@pytest.mark.parametrize("npoints", [8, 9, 1024, 1025])
def test_ft_axis_is_the_fftshifted_frequency_grid(npoints):
    expected = 250 + np.fft.fftshift(np.fft.fftfreq(npoints, 1 / 500))
    np.testing.assert_allclose(ft_axis(250, 500, npoints), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("args", [(0, -1, 8), (0, 10, 2), (0, 10, 8.5), (1j, 10, 8)])
def test_ft_axis_rejects_bad_input(args):
    with pytest.raises(ValueError):
        ft_axis(*args)


def test_axis_1d_units(ss):
    base = {"spins": ["13C"], "offset": 3000, "sweep": 2000, "zerofill": 64}
    hz, label = axis_1d(ss, {**base, "axis_units": "Hz"})
    np.testing.assert_allclose(hz, ft_axis(3000, 2000, 64))
    assert label == "13C offset frequency / Hz"

    khz, _ = axis_1d(ss, {**base, "axis_units": "kHz"})
    np.testing.assert_allclose(khz, 1e-3 * hz)

    ppm, label = axis_1d(ss, {**base, "axis_units": "ppm"})
    carrier = spin("13C")[0] * 14.1 / (2 * np.pi)
    np.testing.assert_allclose(ppm, 1e6 * hz / carrier)
    assert label == "13C chemical shift / ppm"

    points, _ = axis_1d(ss, {**base, "axis_units": "points"})
    np.testing.assert_array_equal(points, np.arange(1, 65))


def test_axis_1d_two_element_sweep(ss):
    ax, _ = axis_1d(ss, {"spins": ["1H"], "sweep": [-100, 100], "zerofill": 5, "axis_units": "Hz"})
    np.testing.assert_allclose(ax, [-100, -50, 0, 50, 100])
    with pytest.raises(ValueError):
        axis_1d(ss, {"spins": ["1H"], "sweep": [-100, 100], "offset": 0, "zerofill": 5, "axis_units": "Hz"})


def test_axis_1d_rejects_unknown_units(ss):
    with pytest.raises(ValueError):
        axis_1d(ss, {"spins": ["1H"], "offset": 0, "sweep": 100, "zerofill": 8, "axis_units": "furlongs"})


def test_plot_1d_follows_spinach_defaults(ss):
    pytest.importorskip("plotly")
    spectrum = np.random.default_rng(0).normal(size=256)
    fig = plot_1d(ss, spectrum, {"spins": ["1H"], "offset": 600, "sweep": 1000})
    (trace,) = fig.data
    ppm, _ = axis_1d(ss, {"spins": ["1H"], "offset": 600, "sweep": 1000, "zerofill": 256, "axis_units": "ppm"})
    np.testing.assert_allclose(trace.x, ppm)
    np.testing.assert_array_equal(trace.y, spectrum)
    assert fig.layout.xaxis.range == (ppm.max(), ppm.min())   # invert_axis=1
    assert "<sup>1</sup>H chemical shift / ppm" == fig.layout.xaxis.title.text
    assert not fig.layout.showlegend


def test_plot_1d_overlays_and_splits_complex_data(ss):
    pytest.importorskip("plotly")
    pars = {"spins": ["1H"], "offset": 0, "sweep": 1000, "axis_units": "Hz", "invert_axis": 0}
    spectra = np.ones((128, 2)) * [1, 2]
    fig = plot_1d(ss, spectra, pars, names=["a", "b"])
    assert [t.name for t in fig.data] == ["a", "b"]
    assert fig.layout.showlegend
    assert fig.layout.xaxis.range[0] < fig.layout.xaxis.range[1]

    fig = plot_1d(ss, np.ones(128) * (1 + 2j), pars)
    assert [t.name for t in fig.data] == ["real", "imag"]
    np.testing.assert_array_equal(fig.data[1].y, 2)


def test_plot_1d_rejects_mismatch_and_unported_options(ss):
    pytest.importorskip("plotly")
    pars = {"spins": ["1H"], "offset": 0, "sweep": 1000}
    with pytest.raises(ValueError, match="zerofill"):
        plot_1d(ss, np.ones(128), {**pars, "zerofill": 256})
    with pytest.raises(NotImplementedError):
        plot_1d(ss, np.ones(128), {**pars, "derivative": 1})
    with pytest.raises(ValueError):
        plot_1d(ss, [np.ones(128), np.ones(64)], pars)


def test_plot_fid_time_axis():
    pytest.importorskip("plotly")
    fig = plot_fid(np.exp(-np.arange(100) / 20) * (1 + 0j), {"sweep": 500})
    assert [t.name for t in fig.data] == ["real", "imag"]
    np.testing.assert_allclose(fig.data[0].x, 1e3 * np.arange(100) / 500)   # ms
    assert fig.layout.xaxis.title.text == "time / ms"
