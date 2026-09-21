"""Standalone SVF construction and provable legacy export reuse.

Input content identifies native geometry. Existing complete legacy exports are
preserved unless they are proven equal or overwrite is explicitly requested.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from ..cache import GeometryStore, content_fingerprint, validate_legacy_geometry, SVF_FIELDS, VISIBILITY_FIELDS
from ..cache.geometry import _lock, canonical_json, _manifest_digest, _read_json
from ..identities import geometry_identity, InputGuard
from ..runtime import get_runtime_options, plan_admission
from .shadows import create_patches
from .svf import svf_calculator_compact, save_svf_zip_npz_outputs
from .visibility import import_visibility_npz, DEFAULT_WORKSPACE_BYTES

RESULT_NAMES = tuple('svf svfaveg svfE svfEaveg svfEveg svfN svfNaveg svfNveg svfS svfSaveg svfSveg svfveg svfW svfWaveg svfWveg vegshmat vbshvegshmat shmat svftotal'.split())
FORMAT = 'solweig-light-standalone-geometry-exports'
VERSION = 1


@contextmanager
def _destination_locks(destinations):
    """Match TransactionalOutputs destination ownership across public workflows.

    Acquisition is sorted and nonblocking, as in the transaction writer. A busy
    destination releases any already acquired locks and fails before production.
    No caller waits while holding a subset of the three publication locks.
    """
    handles = []
    try:
        for destination in sorted({Path(path).resolve() for path in destinations}):
            destination_key = hashlib.sha256(str(destination).encode()).hexdigest()
            lock_root = destination.parent.parent
            if lock_root.name == 'output_folder':
                lock_root = lock_root.parent
            lock_dir = lock_root / '.solweig-light-locks'
            lock_dir.mkdir(parents=True, exist_ok=True)
            handle = open(lock_dir / (destination_key + '.lock'), 'a+b')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException as error:
                handle.close()
                if isinstance(error, BlockingIOError):
                    raise ValueError(f'Geometry export destination already has an owner: {destination}. '
                                     'Retry after the active run finishes.') from error
                raise
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _producer(paths, patch_option, template):
    from osgeo import gdal
    a = template.GetRasterBand(1).ReadAsArray().astype(np.float32)
    def read(path):
        dataset = gdal.Open(str(path))
        try:
            return dataset.ReadAsArray().astype(np.float32)
        finally:
            dataset = None
    tree, dem = read(paths['Trees']), read(paths['DEM'])
    scale = 1 / template.GetGeoTransform()[1]
    tree[tree < 0] = 0
    height = tree + dem
    trunkheight = tree * np.float32(.25) + dem
    bush = np.logical_not(trunkheight * height) * height
    vegdem = tree + a
    vegdem[vegdem == a] = 0
    vegdem2 = tree * np.float32(.25) + a
    vegdem2[vegdem2 == a] = 0
    amaxvalue = np.maximum(a.max(), height.max())
    values = svf_calculator_compact(patch_option, amaxvalue, a, vegdem, vegdem2, bush, scale, save_rasters=False)
    return dict(zip(RESULT_NAMES, values))


def _schema(template, patch_option):
    from osgeo import gdal
    # Ask the installed GTiff driver for the exact defaults used by the legacy
    # writer. Derived subdataset paths describe filenames, not stored metadata.
    with tempfile.TemporaryDirectory(prefix='solweig-export-schema-') as temporary:
        path = Path(temporary) / 'template.tif'
        ds = gdal.GetDriverByName('GTiff').Create(str(path), 1, 1, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(template.GetGeoTransform())
        ds.SetProjection(template.GetProjection())
        ds.GetRasterBand(1).WriteArray(np.zeros((1, 1), np.float32))
        ds = None
        ds = gdal.Open(str(path))
        expected_metadata = _export_metadata(ds)
        ds = None
    return {'shape': (template.RasterYSize, template.RasterXSize),
            'patch_count': int(sum(create_patches(patch_option)[4])),
            'geotransform': tuple(template.GetGeoTransform()),
            'projection': template.GetProjection(), 'nodata': None,
            'export_metadata': expected_metadata}


def _export_metadata(dataset):
    def domains(obj):
        return {name: obj.GetMetadata(name) for name in obj.GetMetadataDomainList() or ['']
                if name != 'DERIVED_SUBDATASETS' and obj.GetMetadata(name)}
    band = dataset.GetRasterBand(1)
    return {'dataset': domains(dataset), 'band': domains(band),
            'description': band.GetDescription(), 'scale': band.GetScale(),
            'offset': band.GetOffset(), 'unit': band.GetUnitType(),
            'color_interpretation': band.GetColorInterpretation(),
            'categories': band.GetCategoryNames(), 'mask_flags': band.GetMaskFlags()}


def _fingerprints(artifacts):
    return {name: content_fingerprint(path) for name, path in artifacts.items()}


def _validate(artifacts, schema):
    from osgeo import gdal
    result = validate_legacy_geometry(artifacts['svfs'], artifacts['shadowmats'], artifacts['svftotal'],
                                      **{key: value for key, value in schema.items() if key != 'export_metadata'})
    for path in [artifacts['svftotal']] + [f'/vsizip/{artifacts["svfs"].resolve()}/{name}.tif' for name in SVF_FIELDS]:
        dataset = gdal.Open(str(path))
        try:
            if _export_metadata(dataset) != schema['export_metadata']:
                raise ValueError(f'Legacy geometry TIFF metadata differs in {path}')
        finally:
            dataset = None
    return result


def _fast_hit(manifest_path, identity, artifacts, schema):
    try:
        value = _read_json(manifest_path)
        if (not isinstance(value, dict) or value.get('format') != FORMAT or value.get('version') != VERSION
                or value.get('manifest_sha256') != _manifest_digest(value)
                or canonical_json(value.get('identity')) != canonical_json(identity)
                or canonical_json(value.get('schema')) != canonical_json(schema)
                or value.get('artifacts') != _fingerprints(artifacts)):
            return None
        _validate(artifacts, schema)
        return value
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError, zipfile.BadZipFile):
        return None


def _compare_tiff(path, expected):
    from osgeo import gdal
    ds = gdal.Open(str(path))
    try:
        band = ds.GetRasterBand(1)
        rows, cols = expected.shape
        width = min(cols, DEFAULT_WORKSPACE_BYTES // 4)
        height = max(1, DEFAULT_WORKSPACE_BYTES // (width * 4))
        for y in range(0, rows, height):
            h = min(height, rows - y)
            for x in range(0, cols, width):
                w = min(width, cols - x)
                data = band.ReadRaster(x, y, w, h, buf_type=gdal.GDT_Float32)
                actual = np.frombuffer(data, dtype=np.float32).reshape(h, w)
                if not np.array_equal(actual.view(np.uint32), expected[y:y+h, x:x+w].view(np.uint32)):
                    raise ValueError(f'Legacy geometry values differ in {path}')
    finally:
        ds = None


def _compare(artifacts, schema, fields):
    _validate(artifacts, schema)
    for name in SVF_FIELDS:
        _compare_tiff(f'/vsizip/{artifacts["svfs"].resolve()}/{name}.tif', fields[name])
    _compare_tiff(artifacts['svftotal'], fields['svftotal'])
    old = import_visibility_npz(artifacts['shadowmats'])
    for name, legacy in (('shmat', 'shadowmat'), ('vegshmat', 'vegshadowmat'), ('vbshvegshmat', 'vbshmat')):
        expected, actual = fields[name], old[legacy]
        pixels = expected.shape[0] * expected.shape[1]
        chunk = max(1, DEFAULT_WORKSPACE_BYTES // 128)
        for patch in range(expected.shape[2]):
            for start in range(0, pixels, chunk):
                stop = min(pixels, start + chunk)
                if not np.array_equal(expected.decode_pixels(patch, start, stop).view(np.uint32),
                                      actual.decode_pixels(patch, start, stop).view(np.uint32)):
                    raise ValueError(f'Legacy geometry values differ in channel {legacy}, patch {patch}')


def _publish_manifest(path, identity, artifacts, schema, cache_hit):
    value = {'format': FORMAT, 'version': VERSION, 'identity': identity,
             'schema': schema, 'artifacts': _fingerprints(artifacts),
             'cache_hit': bool(cache_hit), 'validated': 'exact-values-and-schema'}
    value['manifest_sha256'] = _manifest_digest(value)
    fd, temporary = tempfile.mkstemp(prefix='.geometry-exports-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return value


def _export(output, tile, template, artifacts, fields, schema, manifest_path, publish, check_inputs):
    # All production/schema checks complete before any promised artifact changes.
    with tempfile.TemporaryDirectory(prefix='.geometry-export-', dir=output) as temporary:
        staged = Path(temporary)
        save_svf_zip_npz_outputs(staged, template,
                                *(fields[name] for name in SVF_FIELDS),
                                *(fields[name] for name in VISIBILITY_FIELDS), fields['svftotal'], tile)
        staged_artifacts = {name: staged / path.name for name, path in artifacts.items()}
        _compare(staged_artifacts, schema, fields)
        check_inputs()
        for path in artifacts.values():
            with open(staged / path.name, 'rb') as stream:
                os.fsync(stream.fileno())
        backups = {}
        replaced = []
        previous_manifest = staged / '.previous-manifest'
        had_manifest = manifest_path.exists()
        if had_manifest:
            os.link(manifest_path, previous_manifest)
        try:
            for name, path in artifacts.items():
                if path.exists():
                    backup = staged / (path.name + '.previous')
                    os.link(path, backup)
                    backups[name] = backup
                os.replace(staged / path.name, path)
                replaced.append(name)
            destination_fd = os.open(output, os.O_RDONLY)
            try:
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            return publish()
        except BaseException:
            for name in reversed(replaced):
                if name in backups:
                    os.replace(backups[name], artifacts[name])
                else:
                    artifacts[name].unlink(missing_ok=True)
            if had_manifest:
                os.replace(previous_manifest, manifest_path)
            else:
                manifest_path.unlink(missing_ok=True)
            raise


def prepare_geometry_exports(preprocess_dir, tile, paths, patch_option, overwrite=False, runtime=None):
    """Prepare one standalone logical tile without weakening existing exports.

    A complete identity-less/stale export set requires exact value comparison.
    Mismatches raise with originals preserved; overwrite=True replaces the set.
    A partial set follows the legacy rule and is regenerated as a complete set.
    """
    from osgeo import gdal
    runtime = get_runtime_options() if runtime is None else runtime
    tile = str(tile)
    if Path(tile).name != tile or tile in ('', '.', '..'):
        raise ValueError('Tile must be a single filename component')
    guard = InputGuard({name: paths[name] for name in ('Building_DSM', 'Trees', 'DEM')})
    patch_count = int(sum(create_patches(patch_option)[4]))
    # Admission opens only DSM metadata; no raster arrays exist at this point.
    # The full pipeline inventory conservatively overestimates geometry-only
    # work and preserves logical extent/patch count rather than shrinking either.
    plan_admission([{'tile': tile, 'paths': paths, 'patches': patch_count}], runtime)
    guard.check()
    output = Path(preprocess_dir) / 'SVF'
    output.mkdir(parents=True, exist_ok=True)
    identity = geometry_identity(paths, patch_option)
    guard.check()
    identity['construction'] = 'standalone-svf-v1'
    identity['standalone_implementation'] = content_fingerprint(__file__)
    artifacts = {'svfs': output / f'svfs_{tile}.zip', 'shadowmats': output / f'shadowmats_{tile}.npz',
                 'svftotal': output / f'SkyViewFactor_{tile}.tif'}
    control = Path(preprocess_dir) / '.solweig-light' / 'export-manifests'
    control.mkdir(parents=True, exist_ok=True)
    manifest_path = control / f'geometry-exports_{tile}.json'
    template = gdal.Open(str(paths['Building_DSM']))
    try:
        schema = _schema(template, patch_option)
        with _lock(control / f'geometry-exports_{tile}.lock'), _destination_locks(artifacts.values()):
            guard.check()
            complete = all(path.is_file() for path in artifacts.values())
            if not overwrite and complete:
                existing = _fast_hit(manifest_path, identity, artifacts, schema)
                if existing is not None:
                    guard.check()
                    return existing
            def producer():
                guard.check()
                fields = _producer(paths, patch_option, template)
                guard.check()
                return fields
            def publish(cache_hit):
                guard.check()
                return _publish_manifest(manifest_path, identity, artifacts, schema, cache_hit)
            store = GeometryStore(runtime.cache_dir or Path(preprocess_dir) / '.solweig-light' / 'cache')
            handle = store.get_or_create(identity, producer) if runtime.cache_enabled else None
            try:
                fields = handle.fields if handle is not None else producer()
                cache_hit = handle.hit if handle is not None else False
                guard.check()
                if not overwrite and complete:
                    try:
                        _compare(artifacts, schema, fields)
                    except (OSError, ValueError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
                        raise ValueError(f'Existing geometry exports for tile {tile} cannot be validated against current inputs/patches: {error}. '
                                         'Original exports were preserved; call calculate_svf(..., overwrite=True) to recompute them.') from error
                else:
                    return _export(output, tile, template, artifacts, fields, schema, manifest_path,
                                   lambda: publish(cache_hit), guard.check)
                return publish(cache_hit)
            finally:
                if handle is not None:
                    handle.close()
    finally:
        template = None
