"""Conservative wind-stage admission and single-owner artifact publication.

These accounting units are an inventory, not a hard RSS bound. Expanded forward
and inverse rotations and untrimmed physical supports are reserved explicitly.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import math
import os
from pathlib import Path
import tempfile

from .cache.geometry import canonical_json
from .cache import content_fingerprint
from .runtime import plan_admission, ResourceAdmissionError


def policy_record(value):
    """Record accepted NumPy/nonfinite numeric options without pickle or NaN JSON."""
    import numpy as np
    if isinstance(value, np.ndarray):
        from .cache import array_fingerprint
        return {'array': array_fingerprint(value)}
    if isinstance(value, np.generic):
        return {'numpy_dtype': value.dtype.str, 'bits': value.tobytes().hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {'float': value.hex()}
    if isinstance(value, dict):
        return {name: policy_record(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [policy_record(item) for item in value]
    return value


@dataclass(frozen=True)
class WindMetadata:
    rows: int
    cols: int
    pixel_size: float


@dataclass(frozen=True)
class WindMemoryEstimate:
    shared_bytes: int
    direction_bytes: int
    forward_pixels: int
    inverse_pixels: int
    footprint_bytes: int
    gaussian_support_bytes: int
    wake_ramp_bytes: int
    height_refined: bool

    @property
    def job_bytes(self):
        return self.shared_bytes + self.direction_bytes


def read_metadata(building_fp, tree_fp):
    import rasterio
    with rasterio.open(building_fp, NUM_THREADS="1") as building, rasterio.open(tree_fp, NUM_THREADS="1") as tree:
        if tree.shape != building.shape:
            raise ValueError(f'Tree raster shape {tree.shape} differs from building raster shape {building.shape}: {tree_fp}')
        rows, cols = building.shape
        px = abs(building.transform.a)
    if not math.isfinite(px):
        raise ResourceAdmissionError('Wind raster pixel size must be finite for resource admission')
    return WindMetadata(rows, cols, px)


def _rotated_shape(rows, cols, angle):
    cardinal = int(round(angle)) % 360
    if cardinal in (0, 180):
        return rows, cols
    if cardinal in (90, 270):
        return cols, rows
    cosine, sine = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    return math.ceil(rows * cosine + cols * sine) + 2, math.ceil(cols * cosine + rows * sine) + 2


def estimate_wind_memory(metadata, directions, height_max=None):
    rows, cols, px = metadata.rows, metadata.cols, metadata.pixel_size
    pixels = rows * cols
    shapes = [_rotated_shape(rows, cols, float(int(angle))) for angle in directions] or [(rows, cols)]
    inverse = [_rotated_shape(r, c, -float(int(angle))) for (r, c), angle in zip(shapes, directions)] or [(rows, cols)]
    forward_pixels = max(r * c for r, c in shapes)
    inverse_pixels = max(r * c for r, c in inverse)
    max_cols = max(c for r, c in shapes)
    radius = max(1, int(round(10.0 / max(px, 1e-6))))
    # int64 broadcast distance scratch, boolean footprint, vectors and native
    # morphology reserve. Empty-tree cases conservatively keep the footprint.
    footprint = 16 * (2 * radius + 1) ** 2 + 32 * (2 * radius + 1)
    sigma = max(40.0 / (2.0 * max(px, 1e-6)), .5)
    gaussian = 32 * (2 * math.ceil(4 * sigma) + 3)
    ramp = 32 * (math.ceil(1.5 * max_cols) + 2)
    if height_max is not None and height_max > 0 and px > 0:
        ratio = px / float(height_max)
        back = 5.4 * max_cols / (ratio ** .3 * (1 + .24 * ratio))
        if not math.isfinite(back):
            raise ResourceAdmissionError('Wind wake support is too large for the supported in-memory stage')
        ramp += 32 * (math.ceil(back * 1.001) + 2)
    # Shared normalized rasters/coefficients and preprocessing scratch. Native
    # overhead is separately charged rather than inferred from ndarray sizes.
    shared = 80 * 4 * pixels + footprint + gaussian + 256 * 1024 * 1024
    # Label object/slice reserve, numeric forward and expanded inverse scratch,
    # smoothing/output copies, per-direction morphology, profiles and native IO.
    direction = (512 * forward_pixels + 64 * 4 * forward_pixels +
                 32 * 8 * inverse_pixels + 32 * 4 * pixels + footprint + gaussian +
                 ramp + 64 * 1024 * 1024)
    return WindMemoryEstimate(shared, direction, forward_pixels, inverse_pixels,
                              footprint, gaussian, ramp, height_max is not None)


def admit_wind(metadata, directions, options, max_workers=None, height_max=None):
    estimate = estimate_wind_memory(metadata, directions, height_max)
    upper = len(directions) if max_workers is None else max(1, int(max_workers))
    selected = replace(options, workers=max(1, min(options.workers, max(1, upper))))
    # Charging shared state once PER job overestimates parallel memory but avoids
    # admitting a count based on a shared buffer whose lifetime is misunderstood.
    jobs = [{'memory_estimate_bytes': estimate.job_bytes} for _ in directions] or [{'memory_estimate_bytes': estimate.shared_bytes}]
    admission = plan_admission(jobs, selected)
    return (admission.active_workers if directions else 0), estimate


@contextmanager
def destination_locks(destinations):
    handles = []
    try:
        for destination in sorted({Path(path).resolve() for path in destinations}):
            key = hashlib.sha256(str(destination).encode()).hexdigest()
            root = destination.parent.parent
            if root.name == 'output_folder':
                root = root.parent
            directory = root / '.solweig-light-locks'
            directory.mkdir(parents=True, exist_ok=True)
            handle = open(directory / (key + '.lock'), 'a+b')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException as error:
                handle.close()
                if isinstance(error, BlockingIOError):
                    raise ValueError(f'Wind output destination already has an owner: {destination}') from error
                raise
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_outputs(output_dir, written, staging, identity, metrics, guard):
    output_dir, staging = Path(output_dir), Path(staging)
    unique = sorted({output_dir / path.name for path in written})
    key = hashlib.sha256(canonical_json(sorted(str(path.resolve()) for path in unique))).hexdigest()
    control = output_dir.parent / '.solweig-light' / 'wind-completions'
    control.mkdir(parents=True, exist_ok=True)
    manifest_path = control / (key + '.json')
    backups, replaced = {}, []
    previous_manifest = staging / '.previous-manifest'
    had_manifest = manifest_path.exists()
    if had_manifest:
        os.link(manifest_path, previous_manifest)
    fd, temporary = tempfile.mkstemp(prefix='.wind-completion-', suffix='.tmp', dir=control)
    os.close(fd)
    try:
        # Validate all staged TIFF blocks before modifying any final destination.
        import rasterio
        for final in unique:
            stage = staging / final.name
            with rasterio.open(stage, NUM_THREADS="1") as dataset:
                for _, window in dataset.block_windows(1):
                    dataset.read(1, window=window)
            with open(stage, 'rb') as stream:
                os.fsync(stream.fileno())
        guard.check()
        for final in unique:
            if final.exists():
                backup = staging / (final.name + '.previous')
                os.link(final, backup)
                backups[final] = backup
            os.replace(staging / final.name, final)
            replaced.append(final)
        _sync_directory(output_dir)
        guard.check()
        record = {'format': 'solweig-light-wind-completion', 'version': 1,
                  'identity': identity, 'resources': metrics,
                  'outputs': {str(path.resolve()): content_fingerprint(path) for path in unique}}
        record['manifest_sha256'] = hashlib.sha256(canonical_json(record)).hexdigest()
        with open(temporary, 'wb') as stream:
            stream.write(canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
        guard.check()
        os.replace(temporary, manifest_path)
        _sync_directory(control)
        return [output_dir / path.name for path in written]
    except BaseException:
        for final in reversed(replaced):
            if final in backups:
                os.replace(backups[final], final)
            else:
                final.unlink(missing_ok=True)
        if had_manifest:
            os.replace(previous_manifest, manifest_path)
        else:
            manifest_path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
