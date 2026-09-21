"""Verify built and installed main/companion wheels in isolated subprocesses."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


def command(arguments, **kwargs):
    result = subprocess.run(arguments, text=True, capture_output=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope='module')
def installed_wheels(tmp_path_factory):
    work = tmp_path_factory.mktemp('compat-installed-wheels')
    wheels = work / 'wheels'
    site = work / 'site'
    command([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--wheel-dir', str(wheels), str(REPOSITORY), str(REPOSITORY / 'compat')], cwd=work)
    built = sorted(wheels.glob('*.whl'))
    assert len(built) == 2
    command([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(site), *map(str, built)], cwd=work)
    environment = dict(os.environ, PYTHONPATH=str(site))
    return work, site, environment


def run_installed(installed_wheels, source, *, synthetic=None, check=True):
    work, site, environment = installed_wheels
    if synthetic is not None:
        environment = dict(environment, PYTHONPATH=os.pathsep.join([str(site), str(synthetic)]))
    result = subprocess.run([sys.executable, '-c', source], cwd=work, env=environment, capture_output=True, text=True)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_installed_forwarding_and_matching_dependency(installed_wheels):
    _, site, _ = installed_wheels
    source = f'''
from importlib import metadata, util
import inspect
from pathlib import Path
import solweig_light
import solweig_gpu
from solweig_gpu import solweig_gpu as legacy
names = ('thermal_comfort', 'preprocess', 'run_walls_aspect', 'calculate_svf', 'run_utci_tiles', 'build_inputs', 'build_wind_ext_coeff')
assert Path(solweig_light.__file__).is_relative_to({str(site)!r})
assert Path(solweig_gpu.__file__).is_relative_to({str(site)!r})
for name in names:
    expected = getattr(solweig_light, name)
    assert getattr(solweig_gpu, name) is expected
    assert getattr(legacy, name) is expected
    assert inspect.signature(getattr(legacy, name)) == inspect.signature(expected)
main_version = metadata.version('solweig-light')
assert metadata.version('solweig-light-compat') == main_version
assert metadata.requires('solweig-light-compat') == ['solweig-light==' + main_version]
assert util.find_spec('torch') is None
from solweig_gpu.solweig import Solweig_2022a_calc
from solweig_light.radiation.engine import Solweig_2022a_calc as cpu
assert Solweig_2022a_calc is cpu
from solweig_gpu.shadow import svf_calculator
from solweig_gpu.walls_aspect import findwalls
from solweig_gpu.calculate_utci import utci_calculator
from solweig_gpu.calculate_wbgt import black_globe_temperature
from solweig_gpu.wind_ext_coeff import calculate_wind_ext_coeff
from solweig_gpu.create_inputs import run_create_inputs
from solweig_gpu.preprocessor import process_era5_data, process_wrfout_data
'''
    run_installed(installed_wheels, source)


@pytest.mark.parametrize('module', ['solweig_gpu', 'solweig_gpu.solweig_gpu'])
def test_optional_workflows_import_without_optional_dependencies(installed_wheels, module):
    source = f"""
from {module} import build_inputs, build_wind_ext_coeff
import sys
assert callable(build_inputs) and callable(build_wind_ext_coeff)
for name in ('xarray', 'rasterio', 'pandas', 'ee', 'geemap', 'osmnx', 'torch'):
    assert name not in sys.modules, name
"""
    run_installed(installed_wheels, source)


def test_installed_legacy_executable_help(installed_wheels):
    work, site, environment = installed_wheels
    result = command([str(site / 'bin' / 'thermal_comfort'), '--help'], cwd=work, env=environment)
    assert 'usage:' in result.stdout
    cpu = command([sys.executable, '-c', 'import sys; sys.argv[0] = "thermal_comfort"; from solweig_light.cli import main; main()', '--help'], cwd=work, env=environment)
    assert result.stdout == cpu.stdout


@pytest.mark.parametrize('entry', ['import solweig_gpu', 'from solweig_gpu.solweig_gpu import thermal_comfort', 'from solweig_gpu.cli import main; main()'])
def test_synthetic_upstream_distribution_rejected_before_forwarding(installed_wheels, tmp_path, entry):
    metadata = tmp_path / 'solweig_gpu-2.0.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: solweig-gpu\nVersion: 2.0.0\n')
    result = run_installed(installed_wheels, entry, synthetic=tmp_path, check=False)
    assert result.returncode != 0
    assert 'Conflicting upstream solweig-gpu 2.0.0' in result.stderr
    assert 'Create a clean environment' in result.stderr
    assert 'solweig_light' not in result.stderr

def test_installed_legacy_executable_rejects_synthetic_conflict(installed_wheels, tmp_path):
    metadata = tmp_path / 'solweig_gpu-2.0.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: solweig-gpu\nVersion: 2.0.0\n')
    work, site, environment = installed_wheels
    environment = dict(environment, PYTHONPATH=os.pathsep.join([str(site), str(tmp_path)]))
    result = subprocess.run([str(site / 'bin' / 'thermal_comfort'), '--help'], cwd=work, env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Conflicting upstream solweig-gpu 2.0.0' in result.stderr
