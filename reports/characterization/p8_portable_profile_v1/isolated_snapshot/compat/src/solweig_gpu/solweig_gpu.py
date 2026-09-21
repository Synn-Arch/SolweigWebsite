"""Forward supported legacy public functions without changing signatures."""
from solweig_light.api import thermal_comfort, preprocess, run_walls_aspect, calculate_svf, run_utci_tiles, build_inputs, build_wind_ext_coeff

__all__ = ['thermal_comfort', 'preprocess', 'run_walls_aspect', 'calculate_svf', 'run_utci_tiles', 'build_inputs', 'build_wind_ext_coeff']

