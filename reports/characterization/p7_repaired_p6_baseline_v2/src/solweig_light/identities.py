"""Content identities for reusable geometry and chronological checkpoints.

Operational controls (block size, CPU and memory budgets) deliberately do not
change physical identity. Logical input rasters and their coordinate metadata do.
"""
from importlib.metadata import version
from pathlib import Path

from .cache import array_fingerprint, content_fingerprint, raster_fingerprint

BASELINE = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
PACKAGE = Path(__file__).parent


class InputChangedError(ValueError):
    """A source changed while its identity and numerical values were read."""


class InputGuard:
    """Detect externally modified inputs around reading and cache production.

    No size/mtime shortcut is used. Callers must keep source files unchanged
    for a workflow invocation; detected changes fail rather than publish a
    cache entry whose identity describes different numerical input bytes.
    """
    def __init__(self, paths):
        self.paths = {str(name): Path(path) for name, path in paths.items()}
        self.fingerprints = {name: content_fingerprint(path) for name, path in self.paths.items()}

    def check(self):
        for name, path in self.paths.items():
            if content_fingerprint(path) != self.fingerprints[name]:
                raise InputChangedError(f'Input {name} changed during execution: {path}; retry with stable source files')


def implementation_identity(files):
    return {name: content_fingerprint(PACKAGE / name) for name in sorted(files)}


def geometry_identity(paths, patch_option):
    from .geometry.shadows import create_patches

    return {
        'policy': 'legacy-logical-domain-geometry-v1', 'upstream_commit': BASELINE,
        'inputs': {name: raster_fingerprint(paths[name])
                   for name in ('Building_DSM', 'Trees', 'DEM')},
        'patch_option': int(patch_option),
        'patch_arrays': [array_fingerprint(value) for value in create_patches(patch_option)],
        'normalization': {'raster_dtype': 'float32', 'negative_trees': 'clamp-zero',
                          'trunk_fraction': '.25', 'svf_transmission': '.03',
                          'boundary': 'original-logical-tile', 'nodata': 'original-unmasked'},
        'implementation': implementation_identity([
            'identities.py', 'pipeline.py',
            'geometry/svf.py', 'geometry/shadows.py', 'geometry/sky_compiled.py',
            'geometry/visibility.py', 'geometry/visibility_native.py']),
        'dependencies': {name: version(name) for name in ('numpy', 'numba', 'GDAL')},
    }


def simulation_identity(paths, wind_paths, selected_date, tile, flags, location, utc):
    files = ['pipeline.py', 'models.py', 'data/landcoverclasses_2016a.txt']
    for folder in ('radiation', 'comfort', 'geometry'):
        files.extend(str(path.relative_to(PACKAGE)) for path in (PACKAGE / folder).glob('*.py'))
    return {
        'policy': 'chronological-compatibility-v1', 'upstream_commit': BASELINE,
        'tile': tile, 'selected_date': selected_date,
        'location': {key: float(value).hex() for key, value in location.items()},
        'utc_offset_hours': float(utc).hex(), 'requested_outputs': dict(flags),
        'inputs': {name: (content_fingerprint(path) if name == 'metfiles' else raster_fingerprint(path))
                   for name, path in sorted(paths.items())},
        'wind_inputs': {str(direction): raster_fingerprint(path)
                        for direction, path in sorted(wind_paths.items())},
        'implementation': implementation_identity(files),
        'dependencies': {name: version(name) for name in
                         ('numpy', 'numba', 'scipy', 'GDAL', 'pytz', 'timezonefinder')},
    }
