"""Real wind buffers, bounded scheduling and publication ownership; no oracles generated."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import threading
import time
import weakref

import numpy as np
import pytest
import rasterio

from solweig_light import wind
from solweig_light.cache import content_fingerprint
from solweig_light.identities import InputChangedError
from solweig_light.runtime import RuntimeOptions, ResourceAdmissionError, runtime_options
from solweig_light import wind_resources as resources

ROOT = Path(__file__).parents[1] / 'reference' / 'wind_original_cpu'
DIRECTIONS = tuple(range(0, 360, 30))


def fixture(tmp_path):
    source = ROOT / 'precedence'
    inputs = tmp_path / 'inputs'
    shutil.copytree(source / 'inputs', inputs, ignore=shutil.ignore_patterns('WindCoeff_*'))
    return inputs, source / 'era5', wind._find_building_raster(inputs), wind._find_tree_raster(inputs)


def records(tmp_path):
    return list((tmp_path / '.solweig-light' / 'wind-completions').glob('*.json'))


def outputs(inputs):
    return {path.name: content_fingerprint(path) for path in inputs.glob('WindCoeff_*.tif')}


def mutate(path):
    stamp = path.stat()
    with rasterio.open(path, 'r+') as dataset:
        a = dataset.read(1)
        a[a.shape[0] // 2, a.shape[1] // 2] += np.float32(50)
        dataset.write(a, 1)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert path.stat().st_size == stamp.st_size
    assert path.stat().st_mtime_ns == stamp.st_mtime_ns


def test_expanded_inverse_and_skinny_grid_inventory():
    metadata = resources.WindMetadata(7, 91, 1.)
    estimate = resources.estimate_wind_memory(metadata, DIRECTIONS)
    assert estimate.inverse_pixels > estimate.forward_pixels > metadata.rows * metadata.cols
    for angle in DIRECTIONS:
        forward = wind._rotate_full_extent(np.zeros((7, 91), np.float32), angle)
        inverse = wind._rotate_full_extent(forward, -angle)
        assert forward.size <= estimate.forward_pixels
        assert inverse.size <= estimate.inverse_pixels
    huge = resources.estimate_wind_memory(resources.WindMetadata(1, 1000000, 1.), (30,))
    assert huge.inverse_pixels > 1000000
    refined = resources.estimate_wind_memory(metadata, DIRECTIONS, 1000000.)
    assert refined.height_refined and refined.wake_ramp_bytes > estimate.wake_ramp_bytes
    assert refined.job_bytes > estimate.job_bytes
    tiny_pixel = resources.estimate_wind_memory(resources.WindMetadata(7, 91, .001), DIRECTIONS)
    assert tiny_pixel.footprint_bytes > estimate.footprint_bytes
    assert tiny_pixel.gaussian_support_bytes > estimate.gaussian_support_bytes


def test_cpu_worker_maxworkers_and_memory_caps():
    metadata = resources.WindMetadata(64, 64, 1.)
    options = RuntimeOptions(cpu_budget=4, workers=4, memory_budget_bytes=4 * 1024**3)
    full, estimate = resources.admit_wind(metadata, DIRECTIONS, options)
    assert full == 4
    assert resources.admit_wind(metadata, DIRECTIONS, options, max_workers=2)[0] == 2
    assert resources.admit_wind(metadata, DIRECTIONS, replace(options, workers=1))[0] == 1
    assert resources.admit_wind(metadata, DIRECTIONS, replace(options, cpu_budget=2))[0] == 2
    assert resources.admit_wind(metadata, DIRECTIONS, replace(options, threads_per_worker=2))[0] == 2
    one = replace(options, memory_budget_bytes=estimate.job_bytes * 2 - 1)
    assert resources.admit_wind(metadata, DIRECTIONS, one)[0] == 1
    assert resources.admit_wind(metadata, DIRECTIONS[:1], options)[0] == 1
    with pytest.raises(ResourceAdmissionError):
        resources.admit_wind(metadata, DIRECTIONS, replace(options, memory_budget_bytes=estimate.job_bytes - 1))


def test_preflight_rejects_before_raster_load_and_roughness(tmp_path, monkeypatch):
    inputs, era5, _, _ = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('metadata admission must precede full arrays and roughness read')
    monkeypatch.setattr(wind, '_read_building_height', forbidden)
    monkeypatch.setattr(wind, '_read_z0_from_fsr_at_raster_midpoint', forbidden)
    with runtime_options(memory_budget_bytes=1):
        with pytest.raises(ResourceAdmissionError, match='increase the memory budget'):
            wind.build_wind_ext_coeff(inputs, era5)
    assert outputs(inputs) == {} and records(tmp_path) == []
    assert list(tmp_path.glob('.wind-stage-*')) == []


def test_refined_wake_admission_before_coefficients_and_support_allocation(tmp_path, monkeypatch):
    inputs, _, building, tree = fixture(tmp_path)
    metadata = resources.read_metadata(building, tree)
    initial = resources.estimate_wind_memory(metadata, DIRECTIONS)
    original = wind._read_building_height
    def extreme(*args, **kwargs):
        array, profile, transform = original(*args, **kwargs)
        array[:] = np.float32(1e30)
        return array, profile, transform
    monkeypatch.setattr(wind, '_read_building_height', extreme)
    monkeypatch.setattr(wind, '_coeff_at_z_trees', lambda *args, **kwargs: pytest.fail('refined admission must precede coefficients'))
    with pytest.raises(ResourceAdmissionError):
        wind._compute_wind_full_domain(building_fp=building, tree_fp=tree, output_dir=inputs,
                                       runtime=RuntimeOptions(memory_budget_bytes=initial.job_bytes + 1024))
    assert outputs(inputs) == {} and records(tmp_path) == []


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_actual_concurrency_bounded_futures_and_result_lifetimes(tmp_path, monkeypatch, workers):
    inputs, era5, _, _ = fixture(tmp_path)
    lock = threading.Lock()
    running, peak_running, peak_futures = 0, 0, 0
    weak_arrays, weak_futures = [], []
    class ObservedExecutor(ThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            nonlocal running, peak_running, peak_futures
            def tracked():
                nonlocal running, peak_running
                with lock:
                    running += 1
                    peak_running = max(peak_running, running)
                try:
                    time.sleep(.03)
                    value = function(*args, **kwargs)
                    with lock:
                        weak_arrays.append(weakref.ref(value[1]))
                    return value
                finally:
                    with lock:
                        running -= 1
            future = super().submit(tracked)
            with lock:
                weak_futures.append(weakref.ref(future))
                peak_futures = max(peak_futures, sum(ref() is not None for ref in weak_futures))
            return future
    monkeypatch.setattr(wind, 'ThreadPoolExecutor', ObservedExecutor)
    with runtime_options(cpu_budget=workers, workers=workers, threads_per_worker=1, memory_budget_bytes=4 * 1024**3):
        returned = wind.calculate_wind_ext_coeff(inputs, era5, max_workers=workers)
    assert len(returned) == len(weak_arrays) == len(weak_futures) == 12
    assert peak_running == peak_futures == workers
    assert all(ref() is None for ref in weak_arrays + weak_futures)
    completion = json.loads(records(tmp_path)[0].read_text())
    assert completion['resources']['active_workers'] == workers
    assert completion['resources']['peak_pending_futures'] == workers
    assert completion['resources']['inventory']['height_refined']
    assert list(tmp_path.glob('.wind-stage-*')) == []
    assert not any(path.name.startswith('.') for path in inputs.iterdir())
    with np.load(ROOT / 'precedence' / 'fields.npz') as reference:
        for path in returned:
            with rasterio.open(path) as dataset:
                np.testing.assert_array_equal(dataset.read(1), reference[path.stem])


@pytest.mark.parametrize('boundary', ['input_read', 'roughness', 'staging', 'publication'])
def test_content_mutation_refuses_completion_and_preserves_existing_outputs(tmp_path, monkeypatch, boundary):
    inputs, era5, building, _ = fixture(tmp_path)
    wind.calculate_wind_ext_coeff(inputs, era5, directions=(0, 30))
    before = outputs(inputs)
    completion_path = records(tmp_path)[0]
    completion_before = completion_path.read_bytes()
    changed = []
    if boundary == 'input_read':
        original = wind._read_building_height
        def change(*args, **kwargs):
            value = original(*args, **kwargs)
            if not changed:
                changed.append(1)
                mutate(building)
            return value
        monkeypatch.setattr(wind, '_read_building_height', change)
    elif boundary == 'roughness':
        original = wind._read_z0_from_fsr_at_raster_midpoint
        def change(*args, **kwargs):
            value = original(*args, **kwargs)
            mutate(building)
            changed.append(1)
            return value
        monkeypatch.setattr(wind, '_read_z0_from_fsr_at_raster_midpoint', change)
    elif boundary == 'staging':
        original = wind._save_like_meta
        def change(*args, **kwargs):
            value = original(*args, **kwargs)
            if not changed:
                changed.append(1)
                mutate(building)
            return value
        monkeypatch.setattr(wind, '_save_like_meta', change)
    else:
        original = resources.os.replace
        def change(source, target):
            value = original(source, target)
            if not changed and Path(target).parent == inputs:
                changed.append(1)
                mutate(building)
            return value
        monkeypatch.setattr(resources.os, 'replace', change)
    with pytest.raises(InputChangedError):
        wind.calculate_wind_ext_coeff(inputs, era5, directions=(0, 30))
    assert changed == [1]
    assert outputs(inputs) == before
    assert completion_path.read_bytes() == completion_before
    assert list(tmp_path.glob('.wind-stage-*')) == []


def transaction_owner(tmp_path, destination, source):
    from solweig_light.persistence import TransactionalOutputs
    from solweig_light.io.rasters import RasterMetadata
    from solweig_light.models import SimulationState
    array = np.zeros((2, 3), np.float32)
    writer = TransactionalOutputs(tmp_path / 'output_folder' / '0_0', '0_0',
        RasterMetadata(2, 3, (1., 2., 0., 8., 0., -2.), ''), np.zeros((1, 4)), '2020-06-01', ('UTCI',),
        transaction_dir=tmp_path / 'transactions', identity={'wind-lock': 1})
    writer.write(0, {'UTCI': array})
    writer.checkpoint(1, SimulationState.initial(array, np.float64(1 / 24)))
    stage = writer.staging_directory / 'owned-wind.tif'
    shutil.copyfile(source, stage)
    return writer, {destination: stage}


def test_transaction_owner_blocks_wind_before_full_input_read(tmp_path, monkeypatch):
    inputs, era5, _, _ = fixture(tmp_path)
    wind.calculate_wind_ext_coeff(inputs, era5, directions=(0,))
    destination = inputs / 'WindCoeff_dir000.tif'
    writer, extras = transaction_owner(tmp_path, destination, destination)
    try:
        writer.complete(extras)
        before = outputs(inputs)
        monkeypatch.setattr(wind, '_read_building_height', lambda *args, **kwargs: pytest.fail('busy owner must precede full read'))
        with pytest.raises(ValueError, match='already has an owner'):
            wind.calculate_wind_ext_coeff(inputs, era5, directions=(0,))
        assert outputs(inputs) == before
    finally:
        writer.close()


def test_wind_owner_excludes_transaction_actual_external_publication(tmp_path, monkeypatch):
    from solweig_light.persistence import PersistenceError
    inputs, era5, _, _ = fixture(tmp_path)
    wind.calculate_wind_ext_coeff(inputs, era5, directions=(0,))
    destination = inputs / 'WindCoeff_dir000.tif'
    writer, extras = transaction_owner(tmp_path, destination, destination)
    entered, release = threading.Event(), threading.Event()
    original = wind._read_building_height
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return original(*args, **kwargs)
    monkeypatch.setattr(wind, '_read_building_height', paused)
    try:
        with ThreadPoolExecutor(1) as pool:
            request = pool.submit(wind.calculate_wind_ext_coeff, inputs, era5, directions=(0,))
            try:
                assert entered.wait(20)
                with pytest.raises(PersistenceError, match='already has an owner'):
                    writer.complete(extras)
                assert not writer.publication_path.exists()
            finally:
                release.set()
            request.result(timeout=20)
        writer.complete(extras)
        assert writer.completion_path.exists()
    finally:
        release.set()
        writer.close()


@pytest.mark.parametrize('boundary', ['second_destination', 'completion_manifest'])
def test_publication_interruption_restores_previous_set_and_manifest(tmp_path, monkeypatch, boundary):
    inputs, era5, _, _ = fixture(tmp_path)
    wind.calculate_wind_ext_coeff(inputs, era5, directions=(0, 30))
    before = outputs(inputs)
    completion_path = records(tmp_path)[0]
    completion_before = completion_path.read_bytes()
    original = resources.os.replace
    interrupted = []
    def interrupt(source, target):
        target = Path(target)
        match = target == (inputs / 'WindCoeff_dir030.tif' if boundary == 'second_destination' else completion_path)
        if match and not interrupted:
            interrupted.append(1)
            raise OSError('interrupted wind publication')
        return original(source, target)
    monkeypatch.setattr(resources.os, 'replace', interrupt)
    with pytest.raises(OSError, match='interrupted wind publication'):
        wind.calculate_wind_ext_coeff(inputs, era5, directions=(0, 30), coeff_min=.2)
    assert outputs(inputs) == before
    assert completion_path.read_bytes() == completion_before
    assert list(tmp_path.glob('.wind-stage-*')) == []


def test_numpy_options_record_without_changing_numeric_acceptance(tmp_path):
    inputs, era5, _, _ = fixture(tmp_path)
    result = wind.calculate_wind_ext_coeff(inputs, era5, directions=(np.int64(0),), coeff_min=np.float32(.1))
    assert result == [inputs / 'WindCoeff_dir000.tif']
    completion = json.loads(records(tmp_path)[0].read_text())
    assert completion['identity']['physical_options']['coeff_min']['numpy_dtype'] == np.dtype('float32').str


def test_native_gdal_threads_are_dataset_scoped_and_parent_config_preserved(tmp_path, monkeypatch):
    from rasterio.env import get_gdal_config
    inputs, era5, _, _ = fixture(tmp_path)
    original = rasterio.open
    observed = []
    def inspect(*args, **kwargs):
        observed.append(kwargs.get('NUM_THREADS', kwargs.get('num_threads')))
        return original(*args, **kwargs)
    monkeypatch.setattr(rasterio, 'open', inspect)
    with rasterio.Env(GDAL_NUM_THREADS='4'):
        with runtime_options(cpu_budget=1, workers=1, threads_per_worker=1):
            wind.calculate_wind_ext_coeff(inputs, era5, directions=(0, 30))
        assert str(get_gdal_config('GDAL_NUM_THREADS')) == '4'
    assert observed and set(observed) == {'1'}


def test_public_runtime_snapshot_precedes_discovery_and_is_passed_once(tmp_path, monkeypatch):
    from solweig_light import runtime
    inputs, era5, _, _ = fixture(tmp_path)
    initial = RuntimeOptions(cpu_budget=1, workers=1, memory_budget_bytes=4 * 1024**3)
    later = RuntimeOptions(cpu_budget=4, workers=4, memory_budget_bytes=4 * 1024**3)
    current = [initial]
    captured = []
    def snapshot():
        captured.append(current[0])
        return current[0]
    original = wind._find_tree_raster
    def discover(*args):
        current[0] = later
        return original(*args)
    monkeypatch.setattr(runtime, 'get_runtime_options', snapshot)
    monkeypatch.setattr(wind, '_find_tree_raster', discover)
    wind.calculate_wind_ext_coeff(inputs, era5)
    assert captured == [initial]
    assert json.loads(records(tmp_path)[0].read_text())['resources']['active_workers'] == 1
