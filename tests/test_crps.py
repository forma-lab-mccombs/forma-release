"""Closed-form CRPS in evaluate.py (`_crps_per_entry`).

CRPS is a proper scoring rule in the units of the target; it complements NLL by
being far less tail-sensitive. These tests pin the Gaussian and Laplace closed
forms (Gneiting & Raftery 2007; Jordan et al. 2019) at known points, check the
symmetry/asymptote properties, and confirm unwired families return NaN.
"""
import math
import numpy as np

from forma.scoring.evaluate import _crps_per_entry


def test_gaussian_crps_at_zero_residual():
    # CRPS(N(0,sigma^2), 0) = sigma * (sqrt(2)-1)/sqrt(pi) = 0.2336950*sigma.
    for sigma in (0.5, 1.0, 3.0):
        got = _crps_per_entry(np.array([0.0]), np.array([sigma]), 'gaussian', None)[0]
        assert math.isclose(got, sigma * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi), rel_tol=1e-9)


def test_laplace_crps_at_zero_residual():
    # CRPS(Laplace(0,b), 0) = b*(0 + 1 - 3/4) = 0.25*b, with b = sigma/sqrt(2).
    for sigma in (0.5, 1.0, 3.0):
        got = _crps_per_entry(np.array([0.0]), np.array([sigma]), 'laplace', None)[0]
        assert math.isclose(got, 0.25 * sigma / math.sqrt(2.0), rel_tol=1e-9)


def test_crps_is_even_in_residual():
    res = np.array([0.3, 1.7, 4.2])
    sig = np.array([1.0, 2.0, 0.5])
    for fam in ('gaussian', 'laplace'):
        pos = _crps_per_entry(res, sig, fam, None)
        neg = _crps_per_entry(-res, sig, fam, None)
        assert np.allclose(pos, neg, rtol=1e-12)


def test_crps_asymptotes_to_mae_minus_const():
    # As |res| -> infinity: Gaussian CRPS -> |res| - sigma/sqrt(pi);
    # Laplace CRPS -> |res| - 0.75*b.  (Both < |res|: CRPS rewards sharpness.)
    res = np.array([100.0]); sig = np.array([1.0])
    g = _crps_per_entry(res, sig, 'gaussian', None)[0]
    assert math.isclose(g, 100.0 - 1.0 / math.sqrt(math.pi), rel_tol=1e-9)
    b = 1.0 / math.sqrt(2.0)
    l = _crps_per_entry(res, sig, 'laplace', None)[0]
    assert math.isclose(l, 100.0 - 0.75 * b, rel_tol=1e-9)


def test_crps_nonnegative():
    rng_res = np.linspace(-10, 10, 101)
    sig = np.full_like(rng_res, 1.3)
    for fam in ('gaussian', 'laplace'):
        assert np.all(_crps_per_entry(rng_res, sig, fam, None) >= 0.0)


def test_student_t_returns_nan_not_crash():
    out = _crps_per_entry(np.array([0.0, 1.0]), np.array([1.0, 2.0]), 'student_t', 6.0)
    assert out.shape == (2,) and np.all(np.isnan(out))


def test_unknown_family_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown CRPS family"):
        _crps_per_entry(np.array([0.0]), np.array([1.0]), 'cauchy', None)


def test_gaussian_crps_matches_brute_force_integral():
    # CRPS(F, y) = integral (F(x) - 1{x>=y})^2 dx; check the closed form against a
    # numerical integral. Split at x=y so the integrand discontinuity falls on an
    # endpoint (each piece is smooth) -> the trapezoid rule stays tight.
    from scipy.special import ndtr
    # np.trapz was removed in numpy 2.0 (renamed np.trapezoid). hasattr avoids
    # eagerly evaluating np.trapz as a getattr default (which itself raises on 2.0).
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    sigma, res = 2.0, 1.5            # res = pred - actual; take actual=0, pred=res
    y, mu = 0.0, res                 # forecast mean = res (since actual=0)
    lo, hi = mu - 40 * sigma, mu + 40 * sigma
    left = np.linspace(lo, y, 200001)
    right = np.linspace(y, hi, 200001)
    brute = (trapezoid(ndtr((left - mu) / sigma) ** 2, left)
             + trapezoid((ndtr((right - mu) / sigma) - 1.0) ** 2, right))
    closed = _crps_per_entry(np.array([res]), np.array([sigma]), 'gaussian', None)[0]
    assert math.isclose(brute, closed, rel_tol=1e-5)
