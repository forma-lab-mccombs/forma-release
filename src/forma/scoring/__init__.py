"""Density-track scoring slice (Panel C).

A minimal lift of the research evaluator: ``eval_cache`` (residual/sigma cache)
and ``evaluate`` (seed-mixture NLL + shared truth-path resolution). Kept
torch-free — Panel A point metrics are scored by ``proforma20q evaluate``; this
slice exists for the exact 5-seed mixture NLL and, with
``scripts/mixture_calibration.py``, the exact mixture CRPS / coverage.
"""
