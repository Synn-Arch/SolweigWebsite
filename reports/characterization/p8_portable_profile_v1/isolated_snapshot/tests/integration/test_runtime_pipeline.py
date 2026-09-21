"""Real chronological checks for P6 execution controls and recovery.

References are the original upstream CPU packet; no numerical pipeline mocks.
Only the interruption test injects a failure after a real checkpoint commit.
"""
import json
from pathlib import Path
import shutil
import weakref

import numpy as np
from osgeo import gdal
import pytest

from solweig_light import run_utci_tiles
from solweig_light.runtime import RuntimeOptions, runtime_options

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'tests/reference/small_original_cpu/scene'
PROTOCOL = json.loads((ROOT / 'benchmarks/protocols/comparison_v1.json').read_text())
FLAGS = {f'save_{name}': True for name in
         ('tmrt', 'kup', 'kdown', 'lup', 'ldown', 'shadow', 'wbgt', 'ta', 'wind')}


def prepare(destination):
    prepared = destination / 'processed_inputs'
    shutil.copytree(REFERENCE / 'processed_inputs', prepared)
    return prepared


def run_direct(destination, prepared, flags):
    # Instrumented kernel/checkpoint checks use the internal in-process entry;
    # public worker tests above exercise the isolated native-thread limits.
    from solweig_light.pipeline import files_by_key, run_tile
    paths = {name: files_by_key(prepared / name)['0_0'] for name in
             ('Building_DSM', 'Trees', 'DEM', 'walls', 'aspect', 'metfiles')}
    run_tile(destination, prepared, '2020-07-18', '0_0', paths, flags)


def compare_original(destination, tile='0_0', extra_svf=False):
    expected = REFERENCE / 'output_folder/0_0'
    actual = destination / 'output_folder' / tile
    names = {p.name for p in expected.glob('*.tif')}
    if extra_svf:
        names.add('SVF_0_0.tif')
    assert {p.name.replace(f'_{tile}.tif', '_0_0.tif') for p in actual.glob('*.tif')} == names
    for path in sorted(expected.glob('*.tif')):
        source = gdal.Open(str(path))
        result = gdal.Open(str(actual / path.name.replace('_0_0.tif', f'_{tile}.tif')))
        assert result.GetGeoTransform() == source.GetGeoTransform()
        assert result.GetProjection() == source.GetProjection()
        assert result.GetMetadata() == source.GetMetadata()
        assert result.RasterCount == source.RasterCount == 24
        field = path.name.split('_')[0]
        engine_field = {'TMRT': 'Tmrt', 'Shadow': 'shadow'}.get(field, field)
        rule = PROTOCOL['other_outputs'].get(field, PROTOCOL['field_rules'].get(
            'Solweig_2022a_calc/' + engine_field))
        for index in range(1, 25):
            a, b = result.GetRasterBand(index), source.GetRasterBand(index)
            assert a.GetMetadata() == b.GetMetadata()
            assert a.GetNoDataValue() == b.GetNoDataValue()
            assert a.DataType == b.DataType
            x, y = a.ReadAsArray(), b.ReadAsArray()
            for mask in (np.isnan, np.isposinf, np.isneginf):
                np.testing.assert_array_equal(mask(x), mask(y))
            np.testing.assert_allclose(x, y, atol=rule.get('max_abs', rule.get('atol', 0)),
                                       rtol=rule.get('rtol', 0), equal_nan=True,
                                       err_msg=f'{tile}/{field} timestep {index}')
        result = source = None


@pytest.mark.parametrize('block_pixels,threads', [(32, 1), (128, 2), (509, 1)])
def test_real_pipeline_execution_controls(tmp_path, block_pixels, threads):
    prepared = prepare(tmp_path)
    options = RuntimeOptions(block_pixels=block_pixels, cpu_budget=threads,
                             threads_per_worker=threads, memory_budget_bytes=4 * 1024**3)
    with runtime_options(options):
        run_utci_tiles(str(tmp_path), str(prepared), '2020-07-18', **FLAGS)
    compare_original(tmp_path)


@pytest.mark.parametrize('cold', [False, True])
@pytest.mark.parametrize('boundary', ['committed', 'uncommitted'])
def test_real_pipeline_resumes_committed_state(tmp_path, monkeypatch, cold, boundary):
    from solweig_light.persistence import TransactionalOutputs

    prepared = prepare(tmp_path)
    if cold:
        shutil.rmtree(prepared / 'SVF')
    flags = dict(FLAGS, save_svf=True)
    checkpoint = TransactionalOutputs.checkpoint
    write = TransactionalOutputs.write

    class InjectedInterruption(RuntimeError):
        pass

    def interrupt_after_commit(self, next_timestep, state):
        result = checkpoint(self, next_timestep, state)
        if next_timestep == 3:
            raise InjectedInterruption('after committed timestep 3')
        return result

    def interrupt_uncommitted(self, timestep, fields):
        result = write(self, timestep, fields)
        if timestep == 2:
            raise InjectedInterruption('uncommitted timestep 3')
        return result

    if boundary == 'committed':
        monkeypatch.setattr(TransactionalOutputs, 'checkpoint', interrupt_after_commit)
    else:
        monkeypatch.setattr(TransactionalOutputs, 'write', interrupt_uncommitted)
    with runtime_options(RuntimeOptions(memory_budget_bytes=4 * 1024**3,
                                        checkpoint_interval=1 if boundary == 'committed' else 2)):
        with pytest.raises(InjectedInterruption, match='timestep 3'):
            run_direct(tmp_path, prepared, flags)
    monkeypatch.setattr(TransactionalOutputs, 'checkpoint', checkpoint)
    monkeypatch.setattr(TransactionalOutputs, 'write', write)
    with runtime_options(RuntimeOptions(resume=True, block_pixels=37,
                                        memory_budget_bytes=4 * 1024**3)):
        run_direct(tmp_path, prepared, flags)
    compare_original(tmp_path, extra_svf=cold)
    if cold:
        actual = gdal.Open(str(tmp_path / 'output_folder/0_0/SVF_0_0.tif'))
        expected = gdal.Open(str(REFERENCE / 'processed_inputs/SVF/SkyViewFactor_0_0.tif'))
        np.testing.assert_allclose(actual.ReadAsArray(), expected.ReadAsArray(), atol=1e-6, rtol=0)


def test_real_multiple_tile_workers(tmp_path):
    prepared = prepare(tmp_path)
    # Independent duplicate logical domains have identical forcing and solar
    # location; only artifact keys differ. Both compare to the original oracle.
    for folder in prepared.iterdir():
        if folder.is_dir():
            for path in list(folder.iterdir()):
                if path.is_file() and '_0_0.' in path.name:
                    shutil.copy2(path, path.with_name(path.name.replace('_0_0.', '_1_0.')))
    with runtime_options(RuntimeOptions(workers=2, cpu_budget=2,
                                        memory_budget_bytes=4 * 1024**3)):
        run_utci_tiles(str(tmp_path), str(prepared), '2020-07-18', **FLAGS)
    compare_original(tmp_path, '0_0')
    compare_original(tmp_path, '1_0')


def test_long_timeline_releases_raster_outputs(tmp_path, monkeypatch):
    from solweig_light import pipeline

    prepared = prepare(tmp_path)
    metfile = next((prepared / 'metfiles').glob('*.txt'))
    header = metfile.read_text().splitlines()[0]
    day = np.loadtxt(metfile, skiprows=1)
    days = []
    for offset in range(4):
        values = day.copy()
        values[:, 1] += offset
        days.append(values)
    np.savetxt(metfile, np.concatenate(days), header=header, comments='', fmt='%.12g')
    original = pipeline.Solweig_2022a_calc
    names = ('Tmrt', 'Kdown', 'Kup', 'Ldown', 'Lup', 'shadow')
    indices = {name: pipeline.RETURN_NAMES.index(name) for name in names}
    references = {name: [] for name in names}
    maximum_live = {name: 0 for name in names}

    def observe(*args, **kwargs):
        # Execute the entire real numerical kernel; retain only weak references.
        result = original(*args, **kwargs)
        for name, index in indices.items():
            value = result[index]
            assert isinstance(value, np.ndarray)
            references[name].append(weakref.ref(value))
            live = sum(reference() is not None for reference in references[name])
            maximum_live[name] = max(maximum_live[name], live)
        return result

    monkeypatch.setattr(pipeline, 'Solweig_2022a_calc', observe)
    with runtime_options(RuntimeOptions(memory_budget_bytes=4 * 1024**3)):
        run_direct(tmp_path, prepared, FLAGS)
    assert all(len(values) == 96 for values in references.values())
    # The first call can leave an unreachable engine frame until cyclic
    # collection. Its single generation is constant, not timeline history.
    assert all(count <= 2 for count in maximum_live.values()), maximum_live
    import gc
    gc.collect()
    assert not any(reference() is not None for values in references.values() for reference in values)
    output = gdal.Open(str(tmp_path / 'output_folder/0_0/UTCI_0_0.tif'))
    assert output.RasterCount == 96


@pytest.mark.parametrize('boundary', ['read', 'geometry'])
def test_changed_input_cannot_publish_geometry(tmp_path, monkeypatch, boundary):
    from solweig_light import pipeline
    from solweig_light.identities import InputChangedError

    prepared = prepare(tmp_path)
    target = next((prepared / 'Trees').glob('*.tif'))
    changed = False

    def mutate():
        nonlocal changed
        if changed:
            return
        changed = True
        dataset = gdal.Open(str(target), gdal.GA_Update)
        values = dataset.ReadAsArray()
        values[0, 0] += 1
        dataset.GetRasterBand(1).WriteArray(values)
        dataset.FlushCache()
        dataset = None

    name = 'read_raster' if boundary == 'read' else 'svf_calculator'
    original = getattr(pipeline, name)

    def racing_read(*args, **kwargs):
        result = original(*args, **kwargs)
        mutate()
        return result

    monkeypatch.setattr(pipeline, name, racing_read)
    with pytest.raises(InputChangedError, match='changed during execution'):
        run_direct(tmp_path, prepared, FLAGS)
    assert changed
    assert not list((prepared / '.solweig-light/cache').rglob('manifest.json'))
    assert not list((tmp_path / 'output_folder').rglob('*.tif'))
