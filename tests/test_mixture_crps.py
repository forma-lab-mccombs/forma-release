"""Closed-form mixture CRPS: pin the Laplace kernels against independent checks.

The Laplace-mixture CRPS is easy to get subtly wrong (b = sigma/sqrt(2) vs
sigma; the difference-of-Laplaces cross term; the a == b_l singularity), and a
wrong answer is a plausible-looking number, not a crash — so it is checked here
three ways: numerical quadrature of the CRPS integral, Monte Carlo, and
reduction to evaluate.py's per-seed closed form at K=1.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mixture_calibration import (  # noqa: E402
    _lap_E_abs,
    _lap_E_abs_diff,
    _mixture_crps,
    _mixture_pit,
)
from forma.scoring.evaluate import _crps_per_entry  # noqa: E402

SQRT2 = np.sqrt(2.0)


def _crps_quadrature(y, mus, sigmas, family):
    """CRPS = int (F(x) - 1{x >= y})^2 dx for the equal-weight mixture CDF F."""
    from scipy.special import ndtr

    def F(x):
        if family == 'gaussian':
            return float(np.mean(ndtr((x - mus) / sigmas)))
        b = sigmas / SQRT2
        z = (x - mus) / b
        cdf = np.where(z <= 0, 0.5 * np.exp(np.minimum(z, 0.0)),
                       1.0 - 0.5 * np.exp(-np.maximum(z, 0.0)))
        return float(np.mean(cdf))

    lo = float(np.min(mus) - 80 * np.max(sigmas))
    hi = float(np.max(mus) + 80 * np.max(sigmas))
    pts = sorted(set(list(np.asarray(mus, float)) + [float(y)]))
    left, _ = integrate.quad(lambda x: F(x) ** 2, lo, y,
                             points=[p for p in pts if lo < p < y] or None, limit=500)
    right, _ = integrate.quad(lambda x: (F(x) - 1.0) ** 2, y, hi,
                              points=[p for p in pts if y < p < hi] or None, limit=500)
    return left + right


def _call(y, mus, sigmas, family):
    """_mixture_crps takes res = mu - y and sigma, stacked (K, n)."""
    res = (np.asarray(mus, float) - y).reshape(-1, 1)
    sig = np.asarray(sigmas, float).reshape(-1, 1)
    return float(_mixture_crps(res, sig, family)[0])


# ---------------------------------------------------------------- kernels


def test_lap_E_abs_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for b, m in [(1.0, 0.0), (0.3, 2.5), (2.0, -1.0), (0.05, 0.4)]:
        x = rng.laplace(0.0, b, 4_000_000)
        mc = np.abs(x - m).mean()
        assert _lap_E_abs(np.array([m]), np.array([b]))[0] == pytest.approx(mc, rel=2e-3)


def test_lap_E_abs_diff_matches_monte_carlo():
    rng = np.random.default_rng(1)
    for m, b1, b2 in [(0.0, 1.0, 1.0), (0.0, 1.0, 0.25), (1.5, 0.4, 0.9),
                      (-2.0, 2.0, 0.1), (0.7, 0.05, 0.05)]:
        mc = np.abs(rng.laplace(m, b1, 4_000_000) - rng.laplace(0.0, b2, 4_000_000)).mean()
        got = _lap_E_abs_diff(np.array([m]), np.array([b1]), np.array([b2]))[0]
        assert got == pytest.approx(mc, rel=3e-3)


def test_lap_E_abs_diff_equal_scales_closed_form():
    """a == c is the 0/0 point of the raw partial-fraction form."""
    b, m = 0.37, 0.9
    expect = m + np.exp(-m / b) * (3 * b + m) / 2          # analytic a == c limit
    got = _lap_E_abs_diff(np.array([m]), np.array([b]), np.array([b]))[0]
    assert got == pytest.approx(expect, rel=1e-14)
    # E|L - L'| = 3b/2 at m = 0
    assert _lap_E_abs_diff(np.array([0.0]), np.array([b]), np.array([b]))[0] == \
        pytest.approx(1.5 * b, rel=1e-14)


def _raw_partial_fraction(m, a, c, dps=60):
    """The module docstring's raw form, at `dps` decimal digits (mpmath).

    Mathematically identical to `_lap_E_abs_diff`, so it is the reference the
    float64 implementation must reproduce; in float64 it is the form that dies.
    """
    import mpmath as mp
    with mp.workdps(dps):
        M, a, c = mp.mpf(abs(m)), mp.mpf(a), mp.mpf(c)
        return float(M + (a**3 * mp.e**(-M / a) - c**3 * mp.e**(-M / c)) / (a**2 - c**2))


def _raw_partial_fraction_float64(m, a, c):
    M = abs(m)
    with np.errstate(invalid='ignore'):     # a == c is a deliberate 0/0 here
        return M + (a**3 * np.exp(-M / a) - c**3 * np.exp(-M / c)) / (a**2 - c**2)


@pytest.mark.parametrize('relgap', [1e-2, 1e-3, 1e-6, 1e-9, 1e-12])
def test_lap_E_abs_diff_stable_near_equal_scales(relgap):
    """Near-equal scales: match the exact value, which float64's raw form cannot.

    The two forms are algebraically the same function, so any disagreement is
    pure floating-point cancellation in the raw one.
    """
    b, m = 0.37, 0.9
    a = b * (1 + relgap)
    exact = _raw_partial_fraction(m, a, b)
    got = _lap_E_abs_diff(np.array([m]), np.array([a]), np.array([b]))[0]
    assert got == pytest.approx(exact, rel=1e-13)


def test_raw_partial_fraction_actually_degrades():
    """Documents why the rearrangement exists: the raw float64 form loses ~6
    digits at a 1e-9 relative scale gap and is nan at a zero gap."""
    b, m = 0.37, 0.9
    exact = _raw_partial_fraction(m, b * (1 + 1e-9), b)
    raw = _raw_partial_fraction_float64(m, b * (1 + 1e-9), b)
    stable = _lap_E_abs_diff(np.array([m]), np.array([b * (1 + 1e-9)]), np.array([b]))[0]
    assert abs(raw - exact) / exact > 1e-9          # raw has lost precision
    assert abs(stable - exact) / exact < 1e-13      # stable has not
    assert np.isnan(_raw_partial_fraction_float64(m, b, b))   # 0/0 at equal scales


def test_lap_E_abs_diff_matches_exact_over_random_scales():
    """Random well-separated and near-equal scales alike, vs 60-digit reference."""
    rng = np.random.default_rng(4)
    for _ in range(40):
        m = float(rng.normal(0, 2))
        a = float(np.exp(rng.normal(-1, 1)))
        c = a * (1 + float(rng.choice([1e-13, 1e-7, 1e-2, 0.5, 3.0])))
        got = _lap_E_abs_diff(np.array([m]), np.array([a]), np.array([c]))[0]
        assert got == pytest.approx(_raw_partial_fraction(m, a, c), rel=1e-12)


def test_lap_E_abs_diff_degenerate_scale_limit():
    """c -> 0 collapses L2 to a point mass: E|L1 - mu2| = _lap_E_abs."""
    m, a = 0.8, 0.5
    got = _lap_E_abs_diff(np.array([m]), np.array([a]), np.array([1e-12]))[0]
    assert got == pytest.approx(_lap_E_abs(np.array([m]), np.array([a]))[0], rel=1e-9)


def test_lap_E_abs_diff_symmetric_in_scales():
    rng = np.random.default_rng(2)
    m = rng.normal(0, 2, 500)
    b1, b2 = np.exp(rng.normal(-1, 1, 500)), np.exp(rng.normal(-1, 1, 500))
    np.testing.assert_allclose(_lap_E_abs_diff(m, b1, b2), _lap_E_abs_diff(m, b2, b1), rtol=1e-13)


# ---------------------------------------------------------------- mixture CRPS


@pytest.mark.parametrize('family', ['gaussian', 'laplace'])
def test_mixture_crps_k1_matches_evaluate_per_entry(family):
    """K=1 must reproduce evaluate.py's per-seed closed form exactly."""
    for sigma, res in [(1.0, 0.0), (0.3, 1.1), (2.0, -0.5), (0.014, 3.0)]:
        got = _mixture_crps(np.array([[res]]), np.array([[sigma]]), family)[0]
        ref = _crps_per_entry(np.array([res]), np.array([sigma]), family, None)[0]
        assert got == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize('family', ['gaussian', 'laplace'])
def test_mixture_crps_matches_quadrature(family):
    """The decisive check: closed form vs the CRPS integral itself, K=5."""
    rng = np.random.default_rng(20260710)
    for _ in range(6):
        mus = rng.normal(0, 1, 5)
        sigmas = np.exp(rng.normal(-1.2, 0.7, 5))
        y = float(rng.normal(0, 1.5))
        assert _call(y, mus, sigmas, family) == pytest.approx(
            _crps_quadrature(y, mus, sigmas, family), rel=1e-9)


def test_mixture_crps_laplace_matches_monte_carlo():
    """CRPS = E|X-y| - 0.5 E|X-X'| sampled straight from the mixture."""
    rng = np.random.default_rng(7)
    mus = np.array([0.1, -0.3, 0.05, 0.4, -0.15])
    sigmas = np.array([0.5, 0.7, 0.45, 0.9, 0.6])
    y = 0.22
    b = sigmas / SQRT2
    n = 6_000_000
    k1, k2 = rng.integers(0, 5, n), rng.integers(0, 5, n)
    x1, x2 = rng.laplace(mus[k1], b[k1]), rng.laplace(mus[k2], b[k2])
    mc = np.abs(x1 - y).mean() - 0.5 * np.abs(x1 - x2).mean()
    assert _call(y, mus, sigmas, 'laplace') == pytest.approx(mc, abs=2e-3)


def test_laplace_and_gaussian_mixtures_differ():
    """Guards the bug this test file exists for: the two families must not agree
    on the same (mu, sigma) — otherwise the dispatch is a no-op."""
    mus = np.array([0.1, -0.3, 0.05, 0.4, -0.15])
    sigmas = np.array([0.5, 0.7, 0.45, 0.9, 0.6])
    g = _call(0.22, mus, sigmas, 'gaussian')
    l = _call(0.22, mus, sigmas, 'laplace')
    assert abs(g - l) > 1e-3


# ---------------------------------------------------------------- PIT


@pytest.mark.parametrize('family', ['gaussian', 'laplace'])
def test_mixture_pit_is_a_probability_and_centred(family):
    rng = np.random.default_rng(3)
    sig = np.exp(rng.normal(-0.5, 0.4, (5, 20000)))
    res = np.zeros((5, 20000))                      # mu == y everywhere
    u = _mixture_pit(res, sig, family)
    assert np.all((u >= 0.0) & (u <= 1.0))
    np.testing.assert_allclose(u, 0.5, atol=1e-12)  # symmetric density at its centre


def test_mixture_pit_laplace_is_uniform_when_correct():
    """Sampling y from the model's own Laplace mixture must give a uniform PIT."""
    rng = np.random.default_rng(11)
    n = 400_000
    mus = rng.normal(0, 0.3, (5, n))
    sig = np.full((5, n), 0.8)
    k = rng.integers(0, 5, n)
    y = rng.laplace(mus[k, np.arange(n)], sig[k, np.arange(n)] / SQRT2)
    u = _mixture_pit(mus - y, sig, 'laplace')
    # mean 1/2, variance 1/12 for U(0,1); 4-sigma-ish tolerances at this n
    assert u.mean() == pytest.approx(0.5, abs=5e-3)
    assert u.var() == pytest.approx(1 / 12, abs=5e-3)


def test_mixture_pit_gaussian_form_on_laplace_data_is_not_uniform():
    """The exact miscalibration this PR fixes, as a test."""
    rng = np.random.default_rng(12)
    n = 400_000
    mus = np.zeros((5, n))
    sig = np.full((5, n), 0.8)
    y = rng.laplace(0.0, 0.8 / SQRT2, n)
    u_wrong = _mixture_pit(mus - y, sig, 'gaussian')
    assert abs(u_wrong.var() - 1 / 12) > 0.005
