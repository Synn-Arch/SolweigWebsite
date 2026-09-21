"""Pytest hook for the isolated angular-moment pipeline experiment only."""
from solweig_light.radiation import patch_radiation
from tools.experiments.p7_angular_moment import moment_longwave


def pytest_configure(config):
    patch_radiation._longwave = moment_longwave
    patch_radiation._longwave_serial = moment_longwave
