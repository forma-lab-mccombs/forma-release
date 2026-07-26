import torch
import numpy as np
import pandas as pd
from typing import Union, Any

ArrayLike = Union[torch.Tensor, np.ndarray, pd.Series]
ScalarOrArray = Union[float, int, torch.Tensor, np.ndarray, pd.Series]

def transform(v: ArrayLike, k: ScalarOrArray) -> ArrayLike:
    """
    Non-linear transformation: asinh(k * v)

    Note: This replaces the symmetric log transform from the original spec.
    """
    if isinstance(v, torch.Tensor):
        return torch.asinh(k * v)
    else:
        return np.arcsinh(v * k)

def regularize(v: ArrayLike, k: ScalarOrArray, mu: ScalarOrArray, sigma: ScalarOrArray) -> ArrayLike:
    """
    Standardize the transformed value.
    """
    return (transform(v, k) - mu) / (sigma + 1e-8)

def de_regularize(x: ArrayLike, k: ScalarOrArray, mu: ScalarOrArray, sigma: ScalarOrArray) -> ArrayLike:
    """
    Inverse of regularize.
    x = (transform(v) - mu) / sigma
    transform(v) = x * sigma + mu
    v = sinh(transform(v)) / k
    """
    t = x * sigma + mu
    if isinstance(x, torch.Tensor):
        return torch.sinh(t) / (k + 1e-8)
    else:
        return np.sinh(t) / (k + 1e-8)

def get_derivative_s(v: ArrayLike, k: ScalarOrArray, mu: ScalarOrArray, sigma: ScalarOrArray) -> ArrayLike:
    """
    Derivative of regularize(v) with respect to v.

    regularize(v) = (asinh(k*v) - mu) / sigma
    d/dv regularize(v) = (1/sigma) * d/dv asinh(k*v)
                       = (1/sigma) * (k / sqrt(1 + (k*v)^2))
    """
    if isinstance(v, torch.Tensor):
        return k / (sigma * torch.sqrt(1 + torch.pow(k * v, 2)) + 1e-8)
    else:
        return k / (sigma * np.sqrt(1 + np.power(k * v, 2)) + 1e-8)

