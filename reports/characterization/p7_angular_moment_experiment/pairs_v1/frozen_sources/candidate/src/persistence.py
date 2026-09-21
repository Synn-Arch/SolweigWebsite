"""Crash-recoverable chronological state and GeoTIFF output transactions.

Only a checkpoint pointer commits state and an output-band boundary. Staged
TIFF existence is never completion; a hash-validated completion manifest is.
No numerical model policy is defined here: callers supply the full identity.
"""
from dataclasses import fields as dataclass_fields
import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import uuid

import numpy as np
from osgeo import gdal

from .models import SimulationState
from .io.rasters import StreamingOutputs

SCHEMA = 1


class PersistenceError(ValueError):
    """A transaction cannot be safely reused."""


def _digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path):
    with open(path, "rb") as source:
        os.fsync(source.fileno())


def _atomic_json(path, value):
    """Durably replace one commit record through an atomic rename."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temporary, "x", encoding="utf8") as target:
            target.write(_canonical(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    try:
        with open(path, encoding="utf8") as source:
            record = json.load(source)
            if not isinstance(record, dict):
                raise ValueError("Transaction record must be a JSON object")
            return record
    except (OSError, ValueError) as error:
        raise PersistenceError(f"Missing or corrupt transaction record: {path}") from error


def _encode(value, directory, counter):
    if isinstance(value, (np.ndarray, np.generic)):
        if isinstance(value, np.ndarray) and type(value) is not np.ndarray:
            raise TypeError("Checkpoint arrays must be plain NumPy arrays")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays are not supported in checkpoints")
        name = f"array-{counter[0]}.npy"
        counter[0] += 1
        path = directory / name
        with open(path, "xb") as target:
            np.save(target, array, allow_pickle=False)
            target.flush()
            os.fsync(target.fileno())
        return {"type": "scalar" if isinstance(value, np.generic) else "array",
                "file": name, "sha256": _digest(path)}
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float:
        return {"type": "float", "value": struct.pack(">d", value).hex()}
    if type(value) is complex:
        return {"type": "complex", "real": struct.pack(">d", value.real).hex(),
                "imag": struct.pack(">d", value.imag).hex()}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) in (list, tuple):
        return {"type": type(value).__name__,
                "values": [_encode(item, directory, counter) for item in value]}
    raise TypeError(f"Unsupported checkpoint value type: {type(value).__name__}")


def _decode(value, directory):
    kind = value["type"]
    if kind in ("array", "scalar"):
        name = value["file"]
        if Path(name).name != name:
            raise PersistenceError("Invalid checkpoint payload path")
        path = directory / name
        if _digest(path) != value["sha256"]:
            raise PersistenceError(f"Corrupt checkpoint payload: {path}")
        result = np.load(path, allow_pickle=False)
        if kind == "scalar":
            if result.shape != ():
                raise PersistenceError("Invalid scalar checkpoint shape")
            return result[()]
        return result
    if kind == "none":
        return None
    if kind == "bool":
        if type(value["value"]) is not bool:
            raise PersistenceError("Invalid Boolean checkpoint")
        return value["value"]
    if kind == "int":
        return int(value["value"])
    if kind == "float":
        return struct.unpack(">d", bytes.fromhex(value["value"]))[0]
    if kind == "complex":
        return complex(struct.unpack(">d", bytes.fromhex(value["real"]))[0],
                       struct.unpack(">d", bytes.fromhex(value["imag"]))[0])
    if kind == "str":
        return value["value"]
    if kind in ("list", "tuple"):
        result = [_decode(item, directory) for item in value["values"]]
        return result if kind == "list" else tuple(result)
    raise PersistenceError(f"Invalid checkpoint value type: {kind}")


def save_state(directory, state):
    """Write a new immutable generation, preserving scalar/container types."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        counter = [0]
        encoded = {field.name: _encode(getattr(state, field.name), directory, counter)
                   for field in dataclass_fields(SimulationState)}
        _atomic_json(directory / "state.json", {"schema": SCHEMA, "fields": encoded})
        _sync_directory(directory)
        return _digest(directory / "state.json")
    except BaseException:
        shutil.rmtree(directory)
        raise


def load_state(directory, expected_hash):
    directory = Path(directory)
    try:
        if _digest(directory / "state.json") != expected_hash:
            raise PersistenceError("Corrupt checkpoint state manifest")
        record = _read_json(directory / "state.json")
        names = {field.name for field in dataclass_fields(SimulationState)}
        if record["schema"] != SCHEMA or set(record["fields"]) != names:
            raise PersistenceError("Checkpoint state schema differs from SimulationState")
        return SimulationState(**{name: _decode(value, directory)
                                  for name, value in record["fields"].items()})
    except (OSError, EOFError, KeyError, TypeError, ValueError, struct.error) as error:
        if isinstance(error, PersistenceError):
            raise
        raise PersistenceError("Corrupt checkpoint state") from error


def _band_digest(band, rows, cols):
    """Hash with bounded raster buffers; never read a time history."""
    digest = hashlib.sha256()
    for start in range(0, rows, 64):
        values = band.ReadAsArray(0, start, cols, min(64, rows - start))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


class TransactionalOutputs(StreamingOutputs):
    """Single-owner staged output with explicit checkpoint and completion.

    ``extra_artifacts`` maps explicit final paths to staged paths (e.g.
    conditional SVF and cold-geometry legacy exports outside the tile folder).
    A caller can write those under ``staging_directory`` before ``complete``.
    Context exit closes resources only; success requires explicit ``complete``.
    ``resume=False`` refuses an existing transaction instead of discarding it.
    """
    def __init__(self, directory, tile, metadata, met, selected_date, fields, *,
                 transaction_dir, identity, resume=False):
        self.datasets = {}
        self._locks = []
        self._closed = False
        self._completed = False
        self.metadata = metadata
        self.met = met
        self.date = datetime.datetime.strptime(selected_date, "%Y-%m-%d")
        self.directory = Path(directory).resolve()
        self.fields = tuple(fields)
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("Transaction outputs must be nonempty and unique")
        if any(Path(name).name != name or not name for name in self.fields) or Path(str(tile)).name != str(tile):
            raise ValueError("Output field and tile names must be path components")
        self.destinations = {field: self.directory / f"{field}_{tile}.tif" for field in self.fields}
        self.identity = json.loads(_canonical(identity))
        self.signature = {"schema": SCHEMA, "identity": self.identity,
                          "outputs": [str(self.destinations[name]) for name in self.fields],
                          "rows": metadata.rows, "cols": metadata.cols,
                          "transform": list(metadata.transform), "projection": metadata.projection,
                          "timestamps": [self._timestamp(i) for i in range(len(met))]}
        key = hashlib.sha256(_canonical(sorted(self.signature["outputs"])).encode()).hexdigest()
        self.transaction_directory = Path(transaction_dir).resolve() / key
        # Transaction internals must never appear among legacy output artifacts.
        if self.transaction_directory.is_relative_to(self.directory):
            raise ValueError("transaction_dir must be outside the output directory")
        self.transaction_directory.mkdir(parents=True, exist_ok=True)
        self.record_path = self.transaction_directory / "transaction.json"
        self.completion_path = self.transaction_directory / "complete.json"
        self.publication_path = self.transaction_directory / "publication.json"
        self.next_band = 0
        self._committed = 0
        self._band_hashes = {field: [] for field in self.fields}
        self._restore = None
        try:
            self._acquire_locks(self.destinations.values())
            if self.record_path.exists():
                if not resume:
                    if not self.completion_path.exists():
                        raise PersistenceError("Incomplete output transaction exists; request resume=True explicitly")
                    self._validate_previous_completion()
                    shutil.rmtree(self.transaction_directory)
                    self.transaction_directory.mkdir()
                    self._start_new(directory, tile, metadata, met, selected_date)
                    return
                self._record = _read_json(self.record_path)
                if self._record.get("signature") != self.signature:
                    raise PersistenceError("Checkpoint identity or output schema does not match")
                staging = self._record.get("staging")
                if not isinstance(staging, str):
                    raise PersistenceError("Invalid output staging path")
                self.staging_directory = self.transaction_directory / staging
                if self.staging_directory.parent != self.transaction_directory:
                    raise PersistenceError("Invalid output staging path")
                self._recover()
            else:
                if resume:
                    raise PersistenceError("No output transaction exists to resume")
                if self.completion_path.exists() or self.publication_path.exists():
                    raise PersistenceError("Orphan output completion/publication record")
                self._start_new(directory, tile, metadata, met, selected_date)
        except BaseException:
            self.close()
            raise

    def _start_new(self, directory, tile, metadata, met, selected_date):
        self.staging_directory = self.transaction_directory / ("stage-" + uuid.uuid4().hex)
        super().__init__(self.staging_directory, tile, metadata, met, selected_date, self.fields)
        self.directory = Path(directory).resolve()
        self._record = {"signature": self.signature, "staging": self.staging_directory.name,
                        "checkpoint": None}
        self._flush()
        _atomic_json(self.record_path, self._record)

    def _validate_previous_completion(self):
        record = _read_json(self.record_path)
        completed = _read_json(self.completion_path)
        try:
            if (completed.get("signature") != record.get("signature")
                    or not record.get("checkpoint")
                    or record["checkpoint"].get("next_timestep") != len(record["signature"]["timestamps"])):
                raise PersistenceError("Invalid previous completion record; cannot start a fresh run")
            artifacts = completed["artifacts"]
            if not set(record["signature"]["outputs"]).issubset({item["final"] for item in artifacts}):
                raise PersistenceError("Previous completion omits requested outputs")
            for item in artifacts:
                if _digest(item["final"]) != item["sha256"]:
                    raise PersistenceError("Previous completed artifact is corrupt")
        except (OSError, KeyError, TypeError, AttributeError) as error:
            raise PersistenceError("Previous completion record/artifact is missing or corrupt") from error

    def _timestamp(self, timestep):
        return self.date.replace(hour=int(self.met[timestep, 2]), minute=int(self.met[timestep, 3])).isoformat()

    def _acquire_locks(self, destinations):
        held = {Path(handle.name).stem for handle in self._locks}
        for destination in sorted(Path(path).resolve() for path in destinations):
            destination_key = hashlib.sha256(str(destination).encode()).hexdigest()
            if destination_key in held:
                continue
            lock_root = destination.parent.parent
            if lock_root.name == "output_folder":
                lock_root = lock_root.parent
            lock_dir = lock_root / ".solweig-light-locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path = lock_dir / (destination_key + ".lock")
            handle = open(lock_path, "a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise PersistenceError(f"Output destination already has an owner: {destination}") from error
            self._locks.append(handle)
            held.add(destination_key)

    def _stage_path(self, name):
        return self.staging_directory / self.destinations[name].name

    def _flush(self):
        for name, dataset in self.datasets.items():
            dataset.FlushCache()
            _sync_file(self._stage_path(name))
        _sync_directory(self.staging_directory)

    def _validate_dataset(self, dataset, name, hashes):
        if (dataset.RasterYSize != self.metadata.rows or dataset.RasterXSize != self.metadata.cols
                or dataset.RasterCount != len(self.met)
                or tuple(dataset.GetGeoTransform()) != tuple(self.metadata.transform)
                or dataset.GetProjection() != self.metadata.projection):
            raise PersistenceError(f"Staged TIFF metadata differs: {name}")
        for i in range(len(self.met)):
            band = dataset.GetRasterBand(i + 1)
            if (band.DataType != gdal.GDT_Float32 or band.GetNoDataValue() is not None
                    or band.GetMaskFlags() != gdal.GMF_ALL_VALID):
                raise PersistenceError(f"Staged TIFF band schema differs: {name}")
            if i < self._committed:
                if band.GetMetadata() != {"Time": self._timestamp(i)}:
                    raise PersistenceError(f"Staged TIFF timestamp differs: {name}, band {i + 1}")
                if _band_digest(band, self.metadata.rows, self.metadata.cols) != hashes[i]:
                    raise PersistenceError(f"Corrupt committed TIFF band: {name}, band {i + 1}")

    def _recover(self):
        checkpoint = self._record.get("checkpoint")
        if checkpoint is not None:
            try:
                self._committed = checkpoint["next_timestep"]
                if type(self._committed) is not int or not 0 <= self._committed <= len(self.met):
                    raise PersistenceError("Invalid checkpoint cursor")
                self._band_hashes = checkpoint["band_hashes"]
                if set(self._band_hashes) != set(self.fields) or any(
                        len(value) != self._committed for value in self._band_hashes.values()):
                    raise PersistenceError("Invalid checkpoint band boundaries")
                generation = checkpoint["generation"]
                if Path(generation).name != generation:
                    raise PersistenceError("Invalid state generation path")
                state = load_state(self.transaction_directory / generation, checkpoint["state_sha256"])
                self._restore = (self._committed, state)
            except (KeyError, TypeError) as error:
                raise PersistenceError("Corrupt checkpoint record") from error
        self.next_band = self._committed
        if self.completion_path.exists():
            self._validate_publication(_read_json(self.completion_path), completed=True)
            self._completed = True
            self._prune_state_generations()
            return
        publishing = self.publication_path.exists()
        if publishing:
            self._validate_publication(_read_json(self.publication_path), completed=False)
        for name in self.fields:
            path = self._stage_path(name)
            if not path.exists() and publishing:
                path = self.destinations[name]
            try:
                dataset = gdal.Open(str(path), gdal.GA_ReadOnly if publishing else gdal.GA_Update)
                self._validate_dataset(dataset, name, self._band_hashes[name])
                if publishing:
                    dataset = None
                else:
                    self.datasets[name] = dataset
            except (RuntimeError, AttributeError) as error:
                raise PersistenceError(f"Missing or corrupt staged TIFF: {path}") from error
        self._prune_state_generations()

    def _prune_state_generations(self):
        """Collect only immutable, unreferenced state under destination locks.

        Call after a durable cursor commit or validated recovery. Failed state
        writes/commit attempts may leave orphans, but one successful boundary
        reclaims them. Staged rasters and publication records are never touched.
        """
        checkpoint = self._record.get("checkpoint")
        current = checkpoint["generation"] if checkpoint is not None else None
        removed = False
        for path in self.transaction_directory.iterdir():
            suffix = path.name.removeprefix("state-")
            is_generation = (path.name.startswith("state-") and len(suffix) == 32
                             and all(char in "0123456789abcdef" for char in suffix))
            if (is_generation and path.name != current and path.is_dir()
                    and not path.is_symlink()):
                shutil.rmtree(path)
                removed = True
        if removed:
            _sync_directory(self.transaction_directory)

    def restore(self):
        """Return a fully committed chronological boundary, or ``None``."""
        return self._restore

    def write(self, timestep, fields):
        if self._closed or self._completed or self.publication_path.exists():
            raise RuntimeError("Output transaction is closed or being published")
        super().write(timestep, fields)

    def checkpoint(self, next_timestep, state):
        """Flush output first, then atomically commit complete state and cursor."""
        if self._closed or self._completed or self.publication_path.exists():
            raise RuntimeError("Cannot checkpoint a closed/publishing transaction")
        if type(next_timestep) is not int or next_timestep != self.next_band or next_timestep < self._committed:
            raise ValueError("Checkpoint cursor must match the written chronological boundary")
        self._flush()
        hashes = {name: list(values) for name, values in self._band_hashes.items()}
        for name, dataset in self.datasets.items():
            for i in range(self._committed, next_timestep):
                hashes[name].append(_band_digest(dataset.GetRasterBand(i + 1), self.metadata.rows, self.metadata.cols))
        generation = "state-" + uuid.uuid4().hex
        state_hash = save_state(self.transaction_directory / generation, state)
        checkpoint = {"next_timestep": next_timestep, "generation": generation,
                      "state_sha256": state_hash, "band_hashes": hashes}
        record = dict(self._record, checkpoint=checkpoint)
        _atomic_json(self.record_path, record)
        self._record = record
        self._committed = next_timestep
        self._band_hashes = hashes
        # No retained copy of state/history; restore reloads only on a new owner.
        self._prune_state_generations()

    def _validate_publication(self, publication, *, completed):
        if publication.get("signature") != self.signature or self._committed != len(self.met):
            raise PersistenceError("Completion identity or band boundary differs")
        try:
            artifacts = publication["artifacts"]
            expected = {str(path) for path in self.destinations.values()}
            final_names = [item["final"] for item in artifacts]
            if len(set(final_names)) != len(final_names) or not expected.issubset(final_names):
                raise PersistenceError("Completion manifest omits or duplicates requested TIFFs")
            for item in artifacts:
                final, staged = Path(item["final"]), Path(item["staged"])
                if not final.is_absolute() or staged.parent != self.staging_directory:
                    raise PersistenceError("Invalid completion artifact path")
                path = final if completed or not staged.exists() else staged
                if _digest(path) != item["sha256"]:
                    raise PersistenceError(f"Corrupt published/staged artifact: {path}")
        except (OSError, KeyError, TypeError, AttributeError) as error:
            raise PersistenceError("Missing or corrupt publication record/artifact") from error

    def _publish_artifact(self, staged, final):
        """Separate hook to exercise rename interruption; journal precedes it."""
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staged, final)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # A configured transaction root may be on another filesystem.
            # Copy into the destination filesystem, then atomically rename.
            temporary = final.with_name("." + final.name + "." + self.staging_directory.name + ".tmp")
            try:
                shutil.copyfile(staged, temporary)
                _sync_file(temporary)
                os.replace(temporary, final)
                _sync_directory(final.parent)
                staged.unlink()
            finally:
                temporary.unlink(missing_ok=True)
        _sync_directory(final.parent)
        _sync_directory(staged.parent)

    def complete(self, extra_artifacts=None):
        """Publish TIFFs and extras; write the completion manifest last.

        Publication is recoverable across individual renames, not multi-file
        atomic. Only the separate completion manifest declares success.
        """
        if self._closed:
            raise RuntimeError("Output transaction is closed")
        if self._completed:
            self._validate_publication(_read_json(self.completion_path), completed=True)
            return
        if self._committed != len(self.met) or self.next_band != len(self.met):
            raise RuntimeError("Every requested band must be checkpointed before completion")
        if self.publication_path.exists():
            publication = _read_json(self.publication_path)
            self._validate_publication(publication, completed=False)
            if extra_artifacts:
                supplied = {str(Path(final).resolve()): str(Path(stage).resolve())
                            for final, stage in extra_artifacts.items()}
                recorded = {item["final"]: item["staged"] for item in publication["artifacts"]
                            if item["final"] not in self.signature["outputs"]}
                if supplied != recorded:
                    raise PersistenceError("Extra artifact destinations differ from publication journal")
        else:
            self._flush()
            # Close all TIFF handles before hashing/renaming their durable files.
            self.datasets.clear()
            artifacts = {path: self._stage_path(name) for name, path in self.destinations.items()}
            for final, staged in (extra_artifacts or {}).items():
                final, staged = Path(final).resolve(), Path(staged).resolve()
                if final in artifacts:
                    raise ValueError("Extra artifact collides with a requested output")
                if staged.parent != self.staging_directory:
                    raise ValueError("Extra artifact must be written under staging_directory")
                artifacts[final] = staged
            self._acquire_locks(set(artifacts) - set(self.destinations.values()))
            publication = {"signature": self.signature, "artifacts": [
                {"final": str(final), "staged": str(staged), "sha256": _digest(staged)}
                for final, staged in artifacts.items()]}
            for staged in artifacts.values():
                _sync_file(staged)
            _atomic_json(self.publication_path, publication)
        # Extras on resumed publication also require destination ownership.
        if self.publication_path.exists() and not self.datasets:
            # The initial owner already locked extras above. Track paths to avoid
            # acquiring a second flock on the same destination in this process.
            self._lock_extra_publication(publication)
        for item in publication["artifacts"]:
            staged, final = Path(item["staged"]), Path(item["final"])
            if staged.exists():
                self._publish_artifact(staged, final)
            elif not final.exists() or _digest(final) != item["sha256"]:
                raise PersistenceError(f"Published artifact is missing or corrupt: {final}")
        self._validate_publication(publication, completed=True)
        _atomic_json(self.completion_path, publication)
        self._completed = True

    def _lock_extra_publication(self, publication):
        # Lock handles are keyed by lock filename, whose hash encodes final path.
        held = {Path(handle.name).stem for handle in self._locks}
        missing = [Path(item["final"]) for item in publication["artifacts"]
                   if hashlib.sha256(item["final"].encode()).hexdigest() not in held]
        self._acquire_locks(missing)

    def close(self):
        if self._closed:
            return
        try:
            super().close()
        finally:
            # Release GDAL owners even when a flush raises, before unlocking.
            self.datasets.clear()
            self._closed = True
            for handle in self._locks:
                handle.close()
            self._locks.clear()

    def __exit__(self, exc_type, exc, tb):
        self.close()
