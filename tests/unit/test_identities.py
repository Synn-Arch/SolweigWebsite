"""Cache/checkpoint identities include content and physical policies."""
from pathlib import Path
import os
import shutil
import pytest

from solweig_light.identities import geometry_identity, simulation_identity, InputGuard, InputChangedError

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / 'tests/reference/small_original_cpu/scene/processed_inputs'


def paths_at(directory):
    return {name: next((directory / name).glob('*.txt' if name == 'metfiles' else '*.tif'))
            for name in ('Building_DSM', 'Trees', 'DEM', 'walls', 'aspect', 'metfiles')}


def test_geometry_identity_content_and_patch_order(tmp_path):
    from osgeo import gdal

    shutil.copytree(PREPARED, tmp_path / 'scene')
    paths = paths_at(tmp_path / 'scene')
    initial = geometry_identity(paths, 2)
    assert initial != geometry_identity(paths, 1)
    tree = paths['Trees']
    old = tree.stat()
    dataset = gdal.Open(str(tree), gdal.GA_Update)
    values = dataset.ReadAsArray()
    values[0, 0] += 1
    dataset.GetRasterBand(1).WriteArray(values)
    dataset.FlushCache()
    dataset = None
    os.utime(tree, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert tree.stat().st_size == old.st_size
    assert initial != geometry_identity(paths, 2)


def test_checkpoint_identity_forcing_date_outputs_and_wind(tmp_path):
    shutil.copytree(PREPARED, tmp_path / 'scene')
    paths = paths_at(tmp_path / 'scene')
    location = {'longitude': -74., 'latitude': 40., 'altitude': 3.}

    def identity(date='2020-07-18', flags=None, wind=None):
        return simulation_identity(paths, wind or {}, date, '0_0', flags or {}, location, -4.)

    original = identity()
    assert original != identity(date='2020-07-19')
    assert original != identity(flags={'save_wbgt': True})
    assert original != identity(wind={0: paths['Trees']})
    met = paths['metfiles']
    old = met.stat()
    data = met.read_bytes()
    # Preserve size and timestamps while changing actual forcing/header content.
    met.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    os.utime(met, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert original != identity()


def test_guard_rejects_changed_bytes_with_unchanged_stat(tmp_path):
    source = tmp_path / 'input'
    source.write_bytes(b'first')
    stat = source.stat()
    guard = InputGuard({'scene': source})
    source.write_bytes(b'other')
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(InputChangedError, match='changed during execution'):
        guard.check()
