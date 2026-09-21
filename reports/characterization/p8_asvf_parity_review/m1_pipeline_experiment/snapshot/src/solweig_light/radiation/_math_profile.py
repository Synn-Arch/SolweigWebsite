"""Explicit isolated M1 SLEEF profile; unpromoted experiment."""
import numpy as np
from ._sleef_acos import asvf_fma
from ._sleef_classifier import tan_array, atan_array

PROFILE = "m1-sleef-5a1d179d-fma-f32-classifier-experiment-v1"

def asvf(values):
    values=np.asarray(values)
    if values.dtype != np.float32:
        raise TypeError("Experiment ASVF requires float32 SVF")
    return asvf_fma(values.ravel()).reshape(values.shape)

def tan32(values):
    values=np.asarray(values)
    if values.dtype != np.float32:
        return np.tan(values)
    return tan_array(values.ravel()).reshape(values.shape)

def atan32(values):
    values=np.asarray(values)
    if values.dtype != np.float32:
        return np.arctan(values)
    return atan_array(values.ravel()).reshape(values.shape)
