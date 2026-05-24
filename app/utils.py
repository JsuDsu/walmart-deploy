import numpy as np

def inverse_log1p(x):
    """Aplica expm1 para revertir log1p"""
    return np.expm1(x)