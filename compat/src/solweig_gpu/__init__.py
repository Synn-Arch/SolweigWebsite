"""Explicitly installed legacy namespace for implemented CPU workflows."""
from ._guard import reject_upstream

reject_upstream()

from solweig_light import __version__
from .solweig_gpu import thermal_comfort, preprocess, run_walls_aspect, calculate_svf, run_utci_tiles, build_inputs, build_wind_ext_coeff

__all__ = ['thermal_comfort', 'preprocess', 'run_walls_aspect', 'calculate_svf', 'run_utci_tiles', 'build_inputs', 'build_wind_ext_coeff']

