from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest
from osgeo import gdal

from solweig_light.cache import content_fingerprint, load_legacy_geometry
from solweig_light.geometry import service
from solweig_light.runtime import RuntimeOptions

REFERENCE = Path(__file__).parents[1] / 'reference' / 'small_original_cpu' / 'scene'


def fixture(tmp_path):
    paths = {}
    for name in ('Building_DSM', 'Trees', 'DEM'):
        target = tmp_path / (name + '.tif')
        shutil.copyfile(REFERENCE / (name + '.tif'), target)
        paths[name] = target
    return paths


def artifacts(preprocess):
    output = preprocess / 'SVF'
    return {'svfs': output / 'svfs_0_0.zip', 'shadowmats': output / 'shadowmats_0_0.npz', 'svftotal': output / 'SkyViewFactor_0_0.tif'}


def manifest_path(preprocess):
    return preprocess / '.solweig-light' / 'export-manifests' / 'geometry-exports_0_0.json'


def fingerprints(preprocess):
    return {name: content_fingerprint(path) for name, path in artifacts(preprocess).items()}


def assert_original(preprocess):
    original = REFERENCE / 'processed_inputs' / 'SVF'
    actual = artifacts(preprocess)
    with np.load(actual['shadowmats']) as candidate, np.load(original / 'shadowmats_0_0.npz') as reference:
        assert candidate.files == reference.files
        for name in reference.files:
            np.testing.assert_array_equal(candidate[name], reference[name])
    def tiff_equal(new, old):
        candidate, reference = gdal.Open(str(new)), gdal.Open(str(old))
        assert candidate.GetGeoTransform() == reference.GetGeoTransform()
        assert candidate.GetProjection() == reference.GetProjection()
        assert candidate.GetMetadata() == reference.GetMetadata()
        c, r = candidate.GetRasterBand(1), reference.GetRasterBand(1)
        assert c.DataType == r.DataType == gdal.GDT_Float32
        assert c.GetNoDataValue() == r.GetNoDataValue()
        np.testing.assert_allclose(c.ReadAsArray(), r.ReadAsArray(), rtol=0, atol=1e-6)
        candidate = reference = None
    tiff_equal(actual['svftotal'], original / 'SkyViewFactor_0_0.tif')
    with zipfile.ZipFile(actual['svfs']) as candidate, zipfile.ZipFile(original / 'svfs_0_0.zip') as reference:
        assert candidate.namelist() == reference.namelist()
        for name in reference.namelist():
            tiff_equal(f'/vsizip/{actual["svfs"].resolve()}/{name}', f'/vsizip/{(original / "svfs_0_0.zip").resolve()}/{name}')


def test_original_standalone_cold_warm_native_exports(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    calls = []
    original = service.svf_calculator_compact
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(service, 'svf_calculator_compact', count)
    record = service.prepare_geometry_exports(preprocess, '0_0', paths, 2)
    assert not record['cache_hit']
    assert calls == [1]
    assert_original(preprocess)
    assert sorted(p.name for p in (preprocess / 'SVF').iterdir()) == sorted(p.name for p in artifacts(preprocess).values())
    before = fingerprints(preprocess)
    stamps = {name: path.stat().st_mtime_ns for name, path in artifacts(preprocess).items()}
    service.prepare_geometry_exports(preprocess, '0_0', paths, 2)
    assert calls == [1] and fingerprints(preprocess) == before
    assert {name: path.stat().st_mtime_ns for name, path in artifacts(preprocess).items()} == stamps
    # Native cache alone can recreate deleted legacy outputs without rays.
    artifacts(preprocess)['shadowmats'].unlink()
    refreshed = service.prepare_geometry_exports(preprocess, '0_0', paths, 2)
    assert refreshed['cache_hit'] and calls == [1]
    assert_original(preprocess)


@pytest.mark.parametrize('kind', ['missing', 'corrupt', 'stale_hash', 'stale_identity'])
def test_unproven_complete_equal_exports_compared_and_preserved(tmp_path, monkeypatch, kind):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    path = manifest_path(preprocess)
    if kind == 'missing':
        path.unlink()
    elif kind == 'corrupt':
        path.write_text('{')
    else:
        value = json.loads(path.read_text())
        if kind == 'stale_hash':
            value['artifacts']['svfs']['sha256'] = 'wrong'
        else:
            value['identity']['patch_option'] = 99
        value['manifest_sha256'] = service._manifest_digest(value)
        path.write_text(json.dumps(value))
    before = fingerprints(preprocess)
    calls = []
    original = service._compare
    def compare(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(service, '_compare', compare)
    monkeypatch.setattr(service, 'svf_calculator_compact', lambda *args, **kwargs: pytest.fail('native cache should satisfy rays'))
    record = service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    assert calls == [1]
    assert record['cache_hit'] and fingerprints(preprocess) == before
    assert json.loads(path.read_text())['validated'] == 'exact-values-and-schema'


def change_dsm_preserved_stat(path):
    stamp = path.stat()
    ds = gdal.Open(str(path), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray()
    a[a.shape[0] // 2, a.shape[1] // 2] += np.float32(50)
    band.WriteArray(a)
    ds.FlushCache()
    ds = None
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert path.stat().st_size == stamp.st_size
    assert path.stat().st_mtime_ns == stamp.st_mtime_ns


def test_changed_content_same_size_mtime_reject_preserve_then_overwrite(tmp_path):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 2)
    before = fingerprints(preprocess)
    manifest_before = manifest_path(preprocess).read_bytes()
    change_dsm_preserved_stat(paths['Building_DSM'])
    with pytest.raises(ValueError, match='overwrite=True'):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 2)
    assert fingerprints(preprocess) == before
    assert manifest_path(preprocess).read_bytes() == manifest_before
    refreshed = service.prepare_geometry_exports(preprocess, '0_0', paths, 2, overwrite=True)
    assert refreshed['cache_hit']  # Failed validation already computed fresh cache.
    assert fingerprints(preprocess) != before
    template = gdal.Open(str(paths['Building_DSM']))
    fields = service._producer(paths, 2, template)
    service._compare(artifacts(preprocess), service._schema(template, 2), fields)
    template = None


def test_patch_option_mismatch_rejects_without_modifying_exports(tmp_path):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    before = fingerprints(preprocess)
    with pytest.raises(ValueError, match='overwrite=True'):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 2)
    assert fingerprints(preprocess) == before
    service.prepare_geometry_exports(preprocess, '0_0', paths, 2, overwrite=True)
    template = gdal.Open(str(paths['Building_DSM']))
    service._validate(artifacts(preprocess), service._schema(template, 2))
    template = None


@pytest.mark.parametrize('corrupt', ['zip', 'npz', 'total'])
def test_corrupt_legacy_preserved_until_explicit_overwrite(tmp_path, corrupt):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    path = artifacts(preprocess)[{'zip': 'svfs', 'npz': 'shadowmats', 'total': 'svftotal'}[corrupt]]
    path.write_bytes(b'corrupt')
    before = fingerprints(preprocess)
    with pytest.raises(ValueError, match='Original exports were preserved'):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    assert fingerprints(preprocess) == before
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
    assert fingerprints(preprocess) != before


def test_concurrent_export_owners_single_compute(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    calls = []
    original = service.svf_calculator_compact
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(service, 'svf_calculator_compact', count)
    def request(_):
        return service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    with ThreadPoolExecutor(4) as pool:
        records = list(pool.map(request, range(4)))
    assert calls == [1]
    assert all(r['artifacts'] == fingerprints(preprocess) for r in records)
    assert len(list((preprocess / 'SVF').iterdir())) == 3


@pytest.mark.parametrize('boundary', ['staged_bad', 'second_replace', 'manifest_publish'])
def test_publication_failure_preserves_original_exports(tmp_path, monkeypatch, boundary):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    before = fingerprints(preprocess)
    manifest_before = manifest_path(preprocess).read_bytes()
    if boundary == 'staged_bad':
        original = service.save_svf_zip_npz_outputs
        def bad(output, *args, **kwargs):
            original(output, *args, **kwargs)
            (Path(output) / 'shadowmats_0_0.npz').write_bytes(b'bad')
        monkeypatch.setattr(service, 'save_svf_zip_npz_outputs', bad)
    else:
        original = service.os.replace
        tripped = []
        def interrupt(source, target):
            source, target = Path(source), Path(target)
            if not tripped and ((boundary == 'second_replace' and target == artifacts(preprocess)['shadowmats']) or
                                (boundary == 'manifest_publish' and target == manifest_path(preprocess))):
                tripped.append(1)
                raise OSError('publication interrupted')
            return original(source, target)
        monkeypatch.setattr(service.os, 'replace', interrupt)
    with pytest.raises((OSError, ValueError, zipfile.BadZipFile)):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
    assert fingerprints(preprocess) == before
    assert manifest_path(preprocess).read_bytes() == manifest_before


def test_cache_disabled_and_configured_directory(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    custom = tmp_path / 'custom-cache'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1, runtime=RuntimeOptions(cache_dir=str(custom)))
    assert list(custom.glob('*/manifest.json'))
    calls = []
    original = service.svf_calculator_compact
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(service, 'svf_calculator_compact', count)
    record = service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True,
                                               runtime=RuntimeOptions(cache_enabled=False))
    assert calls == [1] and not record['cache_hit']


def test_legacy_custom_metadata_mismatch_preserved(tmp_path):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    total = artifacts(preprocess)['svftotal']
    dataset = gdal.Open(str(total), gdal.GA_Update)
    dataset.SetMetadataItem('unrecognized', 'stale')
    dataset.GetRasterBand(1).SetDescription('altered-description')
    dataset = None
    before = fingerprints(preprocess)
    with pytest.raises(ValueError, match='metadata differs'):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    assert fingerprints(preprocess) == before
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
    dataset = gdal.Open(str(total))
    assert dataset.GetMetadataItem('unrecognized') is None
    assert dataset.GetRasterBand(1).GetDescription() == ''
    dataset = None


def _process_exports(preprocess, paths, queue, calls):
    original = service.svf_calculator_compact
    def count(*args, **kwargs):
        with Path(calls).open('a') as stream:
            stream.write('produced\n')
        return original(*args, **kwargs)
    service.svf_calculator_compact = count
    try:
        result = service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
        queue.put(result['artifacts'])
    except BaseException as error:
        queue.put(str(error))
        raise


def _crash_after_first_export(preprocess, paths):
    original = service.os.replace
    def interrupt(source, target):
        if Path(target) == artifacts(Path(preprocess))['shadowmats']:
            os._exit(11)
        return original(source, target)
    service.os.replace = interrupt
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)


def test_process_destination_lock_single_producer(tmp_path):
    import multiprocessing
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    calls = tmp_path / 'process-producers.txt'
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    workers = [context.Process(target=_process_exports, args=(preprocess, paths, queue, calls)) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)
        assert not worker.is_alive() and worker.exitcode == 0
    expected = fingerprints(preprocess)
    assert all(queue.get(timeout=2) == expected for _ in workers)
    assert calls.read_text() == 'produced\n'


def test_abrupt_mixed_export_generation_not_accepted(tmp_path):
    import multiprocessing
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    previous_manifest = manifest_path(preprocess).read_bytes()
    change_dsm_preserved_stat(paths['Building_DSM'])
    context = multiprocessing.get_context('spawn')
    worker = context.Process(target=_crash_after_first_export, args=(preprocess, paths))
    worker.start()
    worker.join(20)
    assert not worker.is_alive() and worker.exitcode == 11
    assert manifest_path(preprocess).read_bytes() == previous_manifest
    mixed = fingerprints(preprocess)
    with pytest.raises(ValueError, match='overwrite=True'):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    assert fingerprints(preprocess) == mixed
    # The crashed owner releases the destination lock; explicit repair works.
    record = service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
    assert record['cache_hit']
    assert record['artifacts'] == fingerprints(preprocess)


def transaction_with_geometry_extras(tmp_path, preprocess):
    from solweig_light.persistence import TransactionalOutputs
    from solweig_light.io.rasters import RasterMetadata
    from solweig_light.models import SimulationState
    zero = np.zeros((2, 3), np.float32)
    writer = TransactionalOutputs(tmp_path / 'output_folder' / '0_0', '0_0',
                                  RasterMetadata(2, 3, (1., 2., 0., 8., 0., -2.), ''),
                                  np.zeros((1, 4)), '2020-06-01', ('UTCI',),
                                  transaction_dir=tmp_path / '.solweig-light' / 'transactions',
                                  identity={'cross-workflow-lock': 1})
    writer.write(0, {'UTCI': zero})
    writer.checkpoint(1, SimulationState.initial(zero, np.float64(1 / 24)))
    extras = {}
    for path in artifacts(preprocess).values():
        staged = writer.staging_directory / path.name
        shutil.copyfile(path, staged)
        extras[path] = staged
    return writer, extras


def test_standalone_owner_excludes_transaction_actual_geometry_publication(tmp_path, monkeypatch):
    import threading
    from solweig_light.persistence import PersistenceError
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    entered, release = threading.Event(), threading.Event()
    original = service._producer
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return original(*args, **kwargs)
    monkeypatch.setattr(service, '_producer', paused)
    writer, extras = transaction_with_geometry_extras(tmp_path, preprocess)
    try:
        with ThreadPoolExecutor(1) as pool:
            request = pool.submit(service.prepare_geometry_exports, preprocess, '0_0', paths, 1,
                                  True, RuntimeOptions(cache_enabled=False))
            try:
                assert entered.wait(20)
                before = fingerprints(preprocess)
                with pytest.raises(PersistenceError, match='already has an owner'):
                    writer.complete(extras)
                assert fingerprints(preprocess) == before
                assert not writer.publication_path.exists()
            finally:
                release.set()
            result = request.result(timeout=20)
            assert result['artifacts'] == fingerprints(preprocess)
        writer.complete(extras)
        assert writer.completion_path.exists()
    finally:
        release.set()
        writer.close()


def test_transaction_owner_excludes_standalone_before_production(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    writer, extras = transaction_with_geometry_extras(tmp_path, preprocess)
    try:
        writer.complete(extras)
        before = fingerprints(preprocess)
        manifest_before = manifest_path(preprocess).read_bytes()
        original = service._producer
        monkeypatch.setattr(service, '_producer', lambda *args, **kwargs: pytest.fail('busy destination must fail before producing'))
        with pytest.raises(ValueError, match='already has an owner'):
            service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
        assert fingerprints(preprocess) == before
        assert manifest_path(preprocess).read_bytes() == manifest_before
        monkeypatch.setattr(service, '_producer', original)
    finally:
        writer.close()
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)


def test_busy_later_destination_releases_earlier_subset_locks(tmp_path):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    writer, extras = transaction_with_geometry_extras(tmp_path, preprocess)
    ordered = sorted(path.resolve() for path in extras)
    last = ordered[-1]
    try:
        writer.complete({last: extras[last]})
        with pytest.raises(ValueError, match='already has an owner'):
            service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
        # Service acquired these before finding the busy last destination; they
        # must be released so a subsequent owner can acquire them immediately.
        with service._destination_locks(ordered[:-1]):
            pass
    finally:
        writer.close()


def test_input_mutation_during_producer_refuses_native_and_legacy_publication(tmp_path, monkeypatch):
    from solweig_light.identities import InputChangedError
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    original = service._producer
    def mutate(*args, **kwargs):
        change_dsm_preserved_stat(paths['Building_DSM'])
        return original(*args, **kwargs)
    monkeypatch.setattr(service, '_producer', mutate)
    with pytest.raises(InputChangedError):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    assert list((preprocess / 'SVF').iterdir()) == []
    assert list((preprocess / '.solweig-light' / 'cache').glob('*/manifest.json')) == []
    assert not manifest_path(preprocess).exists()


def test_input_mutation_during_staging_preserves_previous_exports(tmp_path, monkeypatch):
    from solweig_light.identities import InputChangedError
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    before = fingerprints(preprocess)
    manifest_before = manifest_path(preprocess).read_bytes()
    original = service.save_svf_zip_npz_outputs
    def mutate(*args, **kwargs):
        original(*args, **kwargs)
        change_dsm_preserved_stat(paths['Building_DSM'])
    monkeypatch.setattr(service, 'save_svf_zip_npz_outputs', mutate)
    with pytest.raises(InputChangedError):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
    assert fingerprints(preprocess) == before
    assert manifest_path(preprocess).read_bytes() == manifest_before


def test_input_mutation_during_identity_capture_refuses_geometry(tmp_path, monkeypatch):
    from solweig_light.identities import InputChangedError
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    original = service.geometry_identity
    def mutate(*args, **kwargs):
        identity = original(*args, **kwargs)
        change_dsm_preserved_stat(paths['Building_DSM'])
        return identity
    monkeypatch.setattr(service, 'geometry_identity', mutate)
    with pytest.raises(InputChangedError):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    assert list((preprocess / 'SVF').iterdir()) == []
    assert not manifest_path(preprocess).exists()


def test_input_mutation_during_replacement_rolls_back_before_manifest(tmp_path, monkeypatch):
    from solweig_light.identities import InputChangedError
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    service.prepare_geometry_exports(preprocess, '0_0', paths, 1)
    before = fingerprints(preprocess)
    manifest_before = manifest_path(preprocess).read_bytes()
    original = service.os.replace
    changed = []
    def mutate(source, target):
        result = original(source, target)
        if not changed and Path(target) == artifacts(preprocess)['svfs']:
            changed.append(1)
            change_dsm_preserved_stat(paths['Building_DSM'])
        return result
    monkeypatch.setattr(service.os, 'replace', mutate)
    with pytest.raises(InputChangedError):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 1, overwrite=True)
    assert changed == [1]
    assert fingerprints(preprocess) == before
    assert manifest_path(preprocess).read_bytes() == manifest_before


@pytest.mark.parametrize('cache_enabled', [True, False])
def test_geometry_memory_admission_rejects_before_cold_arrays_and_outputs(tmp_path, monkeypatch, cache_enabled):
    from solweig_light.runtime import ResourceAdmissionError
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    def forbidden(*args, **kwargs):
        pytest.fail('memory rejection must precede raster arrays, geometry identity and production')
    monkeypatch.setattr(service, '_producer', forbidden)
    monkeypatch.setattr(service, '_schema', forbidden)
    monkeypatch.setattr(service, 'geometry_identity', forbidden)
    monkeypatch.setattr(gdal.Band, 'ReadAsArray', forbidden)
    monkeypatch.setattr(gdal.Dataset, 'ReadAsArray', forbidden)
    with pytest.raises(ResourceAdmissionError, match='increase the memory budget'):
        service.prepare_geometry_exports(preprocess, '0_0', paths, 2,
                                         runtime=RuntimeOptions(memory_budget_bytes=1, cache_enabled=cache_enabled))
    assert not preprocess.exists()


@pytest.mark.parametrize('patch_option', [1, 2, 3])
def test_geometry_admission_uses_requested_patch_count_metadata_paths(tmp_path, monkeypatch, patch_option):
    paths = fixture(tmp_path)
    preprocess = tmp_path / 'processed'
    requested = int(sum(service.create_patches(patch_option)[4]))
    seen = []
    original = service.plan_admission
    runtime = RuntimeOptions()
    def inspect(jobs, options):
        seen.extend(jobs)
        assert options is runtime
        assert 'rows' not in jobs[0] and 'cols' not in jobs[0] and 'shape' not in jobs[0]
        assert jobs[0]['paths'] is paths
        assert jobs[0]['patches'] == requested
        plan = original(jobs, options)
        assert plan.active_workers == 1
        raise RuntimeError('admission observed')
    monkeypatch.setattr(service, 'plan_admission', inspect)
    monkeypatch.setattr(gdal.Band, 'ReadAsArray', lambda *args, **kwargs: pytest.fail('admission must read metadata only'))
    monkeypatch.setattr(gdal.Dataset, 'ReadAsArray', lambda *args, **kwargs: pytest.fail('admission must read metadata only'))
    with pytest.raises(RuntimeError, match='admission observed'):
        service.prepare_geometry_exports(preprocess, '0_0', paths, patch_option, runtime=runtime)
    assert len(seen) == 1 and not preprocess.exists()
