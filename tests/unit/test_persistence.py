"""Real GDAL transaction and complete-state recovery, without model mocks."""
from dataclasses import fields
import json
import subprocess
import sys

import numpy as np
from osgeo import gdal
import pytest

from solweig_light.io.rasters import RasterMetadata, StreamingOutputs, write_single
from solweig_light.models import SimulationState
from solweig_light import persistence
from solweig_light.persistence import PersistenceError, TransactionalOutputs, load_state, save_state


def state():
    result = SimulationState.initial(np.zeros((2, 3), dtype=np.float32), np.float64(1 / 24))
    result.CI = np.float32(-0.0)
    result.firstdaytime = True
    result.timeadd = -0.0
    result.Tgmap1E = np.arange(6, dtype=np.int16).reshape(2, 3)
    result.Tgmap1N = np.arange(6, dtype=np.float64).reshape(2, 3)
    result.Twater = [np.float32(np.nan), 5, None, [np.int64(8), (1 + 2j, "water")]]
    return result


def assert_value(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, (np.ndarray, np.generic)):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_value(a, e)
    elif isinstance(actual, float):
        assert actual.hex() == expected.hex()
    else:
        assert actual == expected


def assert_state(actual, expected):
    for field in fields(SimulationState):
        assert_value(getattr(actual, field.name), getattr(expected, field.name))


@pytest.fixture
def setup(tmp_path):
    metadata = RasterMetadata(2, 3, (1., 2., 0., 8., 0., -2.), "")
    met = np.zeros((4, 4))
    met[:, 2] = [3, 4, 5, 6]
    def create(*, resume=False, identity=None, names=("UTCI", "TMRT")):
        return TransactionalOutputs(tmp_path / "output" / "0_0", "0_0", metadata, met,
                                    "2020-06-01", names, transaction_dir=tmp_path / "transactions",
                                    identity=identity or {"scene": "sha256:a", "forcing": "sha256:b"}, resume=resume)
    return tmp_path, metadata, met, create


def values(i):
    a = np.full((2, 3), i + .25, dtype=np.float32)
    a[0, 0] = np.nan
    return {"UTCI": a, "TMRT": a + 20}


def finish(writer, start=0):
    for i in range(start, 4):
        writer.write(i, values(i))
        writer.checkpoint(i + 1, state())
    writer.complete()


def test_roundtrip_all_state_types(tmp_path):
    original = state()
    original.TgOut1 = np.array(1., dtype=np.float32)
    original.Twater += [np.datetime64("2020-01-01"), np.array(["a", "bb"], dtype="U2"),
                        np.uint64(2**64 - 1), float("inf"), np.complex64(1j)]
    digest = save_state(tmp_path / "state", original)
    assert_state(load_state(tmp_path / "state", digest), original)
    assert len(list((tmp_path / "state").glob("*.npy"))) > 0


def test_state_object_rejected(tmp_path):
    original = state()
    original.Twater = np.array([{}], dtype=object)
    with pytest.raises(TypeError, match="Object"):
        save_state(tmp_path / "state", original)
    assert not (tmp_path / "state").exists()


def test_real_outputs_match_default_no_history(setup):
    tmp_path, metadata, met, create = setup
    with create() as writer:
        finish(writer)
        assert writer.datasets == {}
        assert not hasattr(writer, "history")
        completion = json.loads(writer.completion_path.read_text())
        assert len(completion["artifacts"]) == 2
        assert writer.completion_path.parent.parent == tmp_path / "transactions"
    with StreamingOutputs(tmp_path / "reference", "0_0", metadata, met, "2020-06-01", ("UTCI", "TMRT")) as reference:
        for i in range(4):
            reference.write(i, values(i))
    output = tmp_path / "output" / "0_0"
    assert sorted(p.name for p in output.iterdir()) == ["TMRT_0_0.tif", "UTCI_0_0.tif"]
    for name in ("TMRT", "UTCI"):
        actual = gdal.Open(str(output / f"{name}_0_0.tif"))
        expected = gdal.Open(str(tmp_path / "reference" / f"{name}_0_0.tif"))
        np.testing.assert_array_equal(actual.ReadAsArray(), expected.ReadAsArray())
        assert actual.GetGeoTransform() == expected.GetGeoTransform()
        for i in range(4):
            assert actual.GetRasterBand(i + 1).GetMetadata() == expected.GetRasterBand(i + 1).GetMetadata()
            assert actual.GetRasterBand(i + 1).GetNoDataValue() is None


def test_recover_replays_uncommitted_band(setup):
    tmp_path, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        writer.checkpoint(1, state())
        writer.write(1, {key: val + 999 for key, val in values(1).items()})
        assert not writer.completion_path.exists()
    with create(resume=True) as writer:
        start, restored = writer.restore()
        assert start == 1
        assert_state(restored, state())
        finish(writer, start)
    with create(resume=True) as writer:
        assert writer.restore()[0] == 4
        assert writer.datasets == {}
        writer.complete()
        with pytest.raises(RuntimeError):
            writer.write(4, values(0))


def test_no_checkpoint_replay_from_start(setup):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
    with create(resume=True) as writer:
        assert writer.restore() is None
        finish(writer)


def test_wrong_identity_and_corrupt_state(setup):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        writer.checkpoint(1, state())
        record = writer._record
        transaction = writer.transaction_directory
    with pytest.raises(PersistenceError, match="identity"):
        create(resume=True, identity={"scene": "changed"})
    generation = transaction / record["checkpoint"]["generation"]
    (generation / "array-0.npy").write_bytes(b"corrupt")
    with pytest.raises(PersistenceError, match="Corrupt"):
        create(resume=True)


@pytest.mark.parametrize("kind", ["metadata", "timestamp", "band", "nodata", "mask"])
def test_corrupt_tiff_rejected(setup, kind):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        writer.checkpoint(1, state())
        path = writer._stage_path("UTCI")
    dataset = gdal.Open(str(path), gdal.GA_Update)
    if kind == "metadata":
        dataset.SetGeoTransform((999., 2., 0., 8., 0., -2.))
    elif kind == "timestamp":
        dataset.GetRasterBand(1).SetMetadata({"Time": "wrong"})
    elif kind == "band":
        dataset.GetRasterBand(1).WriteArray(np.zeros((2, 3), dtype=np.float32))
    elif kind == "nodata":
        dataset.GetRasterBand(1).SetNoDataValue(-9999)
    else:
        dataset.CreateMaskBand(gdal.GMF_PER_DATASET)
    dataset = None
    with pytest.raises(PersistenceError, match="TIFF"):
        create(resume=True)


@pytest.mark.parametrize("boundary", ["flush", "state", "commit"])
def test_checkpoint_failure_preserves_previous_boundary(setup, monkeypatch, boundary):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        writer.checkpoint(1, state())
        writer.write(1, values(1))
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise OSError("injected interruption")
            if boundary == "flush":
                patch.setattr(writer, "_flush", fail)
            elif boundary == "state":
                patch.setattr(persistence, "save_state", fail)
            else:
                real = persistence._atomic_json
                def atomic(path, value):
                    if path == writer.record_path:
                        fail()
                    return real(path, value)
                patch.setattr(persistence, "_atomic_json", atomic)
            with pytest.raises(OSError, match="interruption"):
                writer.checkpoint(2, state())
    with create(resume=True) as writer:
        assert writer.restore()[0] == 1
        finish(writer, 1)


def test_failure_between_requested_band_writes(setup, monkeypatch):
    _, _, _, create = setup
    with create() as writer:
        # First field is consumed before second field's invalid shape is found.
        with pytest.raises(ValueError, match="shape"):
            writer.write(0, {"UTCI": values(0)["UTCI"], "TMRT": np.zeros((1, 1))})
    with create(resume=True) as writer:
        assert writer.restore() is None
        finish(writer)


@pytest.mark.parametrize("boundary", ["first_rename", "last_rename", "manifest"])
def test_partial_publication_retry_with_svf(setup, monkeypatch, boundary):
    tmp_path, metadata, _, create = setup
    with create() as writer:
        for i in range(4):
            writer.write(i, values(i))
        writer.checkpoint(4, state())
        svf_final = writer.directory / "SVF_0_0.tif"
        svf_stage = writer.staging_directory / "SVF_0_0.tif"
        write_single(svf_stage, np.ones((2, 3)), metadata)
        with monkeypatch.context() as patch:
            if boundary == "manifest":
                real = persistence._atomic_json
                def atomic(path, value):
                    if path == writer.completion_path:
                        raise OSError("injected manifest")
                    return real(path, value)
                patch.setattr(persistence, "_atomic_json", atomic)
            else:
                real = writer._publish_artifact
                count = [0]
                def publish(stage, final):
                    real(stage, final)
                    count[0] += 1
                    if count[0] == (1 if boundary == "first_rename" else 3):
                        raise OSError("injected rename")
                patch.setattr(writer, "_publish_artifact", publish)
            with pytest.raises(OSError, match="injected"):
                writer.complete({svf_final: svf_stage})
            assert not writer.completion_path.exists()
    with create(resume=True) as writer:
        assert writer.restore()[0] == 4
        writer.complete({svf_final: svf_stage})
        record = json.loads(writer.completion_path.read_text())
        assert len(record["artifacts"]) == 3
    assert len(list((tmp_path / "output" / "0_0").glob("*.tif"))) == 3


def test_competing_destination_owner_and_process_crash(setup):
    tmp_path, _, _, create = setup
    with create() as writer:
        with pytest.raises(PersistenceError, match="owner"):
            create(resume=True)
        writer.write(0, values(0))
        writer.checkpoint(1, state())
    code = '''import os, numpy as np
from solweig_light.persistence import TransactionalOutputs
from solweig_light.io.rasters import RasterMetadata
from pathlib import Path
p=Path(__import__('sys').argv[1]); m=np.zeros((4,4)); m[:,2]=[3,4,5,6]
w=TransactionalOutputs(p/'output'/'0_0','0_0',RasterMetadata(2,3,(1.,2.,0.,8.,0.,-2.),''),m,'2020-06-01',('UTCI','TMRT'),transaction_dir=p/'transactions',identity={'scene':'sha256:a','forcing':'sha256:b'},resume=True)
os._exit(7)
'''
    process = subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=False)
    assert process.returncode == 7
    with create(resume=True) as writer:
        assert writer.restore()[0] == 1
        finish(writer, 1)


def test_fresh_after_completed_and_incomplete_refused(setup):
    _, _, _, create = setup
    with create() as writer:
        finish(writer)
    with create(identity={"scene": "new"}) as writer:
        assert writer.restore() is None
        assert writer.next_band == 0
        assert not writer.completion_path.exists()
    with pytest.raises(PersistenceError, match="resume=True"):
        create()


def test_complete_requires_committed_all_bands_and_corrupt_completion_rejected(setup):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        with pytest.raises(RuntimeError, match="checkpointed"):
            writer.complete()
        writer.checkpoint(1, state())
        finish(writer, 1)
        path = writer.destinations["UTCI"]
    path.write_bytes(b"corrupt")
    with pytest.raises(PersistenceError, match="Corrupt"):
        create(resume=True)
    with pytest.raises(PersistenceError, match="corrupt"):
        create()


def test_nan_python_float_payload_preserved(tmp_path):
    import struct
    original = state()
    original.timeadd = struct.unpack(">d", bytes.fromhex("fff80000000000a1"))[0]
    digest = save_state(tmp_path / "state", original)
    restored = load_state(tmp_path / "state", digest)
    assert struct.pack(">d", restored.timeadd) == struct.pack(">d", original.timeadd)


@pytest.mark.parametrize("record_kind", ["invalid_json", "list", "staging", "cursor", "generation", "hashes"])
def test_malformed_checkpoint_rejected(setup, record_kind):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        writer.checkpoint(1, state())
        path = writer.record_path
    record = json.loads(path.read_text())
    if record_kind == "invalid_json":
        path.write_text("{incomplete")
    elif record_kind == "list":
        path.write_text("[]")
    else:
        if record_kind == "staging":
            del record["staging"]
        elif record_kind == "cursor":
            record["checkpoint"]["next_timestep"] = 1.5
        elif record_kind == "generation":
            record["checkpoint"]["generation"] = "../unsafe"
        else:
            record["checkpoint"]["band_hashes"] = []
        path.write_text(json.dumps(record))
    with pytest.raises(PersistenceError):
        create(resume=True)


def test_cross_filesystem_publication_copy_then_rename(setup, monkeypatch):
    import errno
    _, _, _, create = setup
    with create() as writer:
        for i in range(4):
            writer.write(i, values(i))
        writer.checkpoint(4, state())
        real = persistence.os.replace
        def replace(source, target):
            if source.parent == writer.staging_directory:
                raise OSError(errno.EXDEV, "different filesystems")
            return real(source, target)
        monkeypatch.setattr(persistence.os, "replace", replace)
        writer.complete()
        assert writer.completion_path.exists()
        assert not list(writer.directory.glob(".*.tmp"))


def test_resume_different_transaction_root_cannot_bypass_destination_lock(setup):
    tmp_path, metadata, met, create = setup
    with create():
        with pytest.raises(PersistenceError, match="owner"):
            TransactionalOutputs(tmp_path / "output" / "0_0", "0_0", metadata, met,
                                 "2020-06-01", ("UTCI",), transaction_dir=tmp_path / "other-root",
                                 identity={"other": True})


def test_checkpoint_does_not_retain_live_state_or_output_arrays(setup):
    import gc
    import weakref
    _, _, _, create = setup
    with create() as writer:
        carried = state()
        state_array = weakref.ref(carried.Tgmap1)
        output = values(0)
        output_array = weakref.ref(output["UTCI"])
        writer.write(0, output)
        writer.checkpoint(1, carried)
        del output, carried
        gc.collect()
        assert state_array() is None
        assert output_array() is None


def test_error_after_cursor_publication_recovers_new_committed_boundary(setup, monkeypatch):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        writer.checkpoint(1, state())
        writer.write(1, values(1))
        with monkeypatch.context() as patch:
            real = persistence._atomic_json
            def atomic(path, value):
                real(path, value)
                if path == writer.record_path:
                    raise OSError("interruption after commit")
            patch.setattr(persistence, "_atomic_json", atomic)
            with pytest.raises(OSError, match="after commit"):
                writer.checkpoint(2, state())
    with create(resume=True) as writer:
        assert writer.restore()[0] == 2
        finish(writer, 2)


def test_completed_resume_checks_identity_and_state_payload(setup):
    _, _, _, create = setup
    with create() as writer:
        finish(writer)
        record = writer._record
        transaction = writer.transaction_directory
    with pytest.raises(PersistenceError, match="identity"):
        create(resume=True, identity={"scene": "changed"})
    path = transaction / record["checkpoint"]["generation"] / "state.json"
    path.write_text("{}")
    with pytest.raises(PersistenceError, match="Corrupt"):
        create(resume=True)


def test_external_geometry_exports_partial_publication_retry(setup, monkeypatch):
    tmp_path, _, _, create = setup
    with create() as writer:
        for i in range(4):
            writer.write(i, values(i))
        writer.checkpoint(4, state())
        external = tmp_path / "preprocessing" / "SVF"
        extras = {}
        for name in ("SkyViewFactor_0_0.tif", "svfs_0_0.zip", "shadowmats_0_0.npz"):
            staged = writer.staging_directory / name
            staged.write_bytes(b"test geometry payload: " + name.encode())
            extras[external / name] = staged
        real = writer._publish_artifact
        count = [0]
        def publish(staged, final):
            real(staged, final)
            count[0] += 1
            if count[0] == 4:
                raise OSError("interrupted external export")
        monkeypatch.setattr(writer, "_publish_artifact", publish)
        with pytest.raises(OSError, match="external export"):
            writer.complete(extras)
    with create(resume=True) as writer:
        assert writer.restore()[0] == 4
        # Root need not recreate already-published geometry or staged extras.
        writer.complete({})
        assert len(json.loads(writer.completion_path.read_text())["artifacts"]) == 5
    with create(resume=True) as writer:
        writer.complete({})
    assert len(list(external.iterdir())) == 3


def test_standard_output_folder_has_no_lock_artifacts(setup):
    tmp_path, metadata, met, _ = setup
    directory = tmp_path / "output_folder" / "0_0"
    with TransactionalOutputs(directory, "0_0", metadata, met, "2020-06-01", ("UTCI",),
                              transaction_dir=tmp_path / "tx-standard", identity={"a": 1}) as writer:
        finish(writer)
    assert sorted(path.name for path in directory.parent.iterdir()) == ["0_0"]
    assert (tmp_path / ".solweig-light-locks").exists()


def test_shared_external_extra_destination_has_single_owner(setup):
    tmp_path, metadata, met, create = setup
    with create() as first:
        for i in range(4):
            first.write(i, values(i))
        first.checkpoint(4, state())
        staged_first = first.staging_directory / "shared.zip"
        staged_first.write_bytes(b"first")
        shared = tmp_path / "preprocessing" / "SVF" / "shared.zip"
        first.complete({shared: staged_first})
        with TransactionalOutputs(tmp_path / "output" / "1_0", "1_0", metadata, met,
                                  "2020-06-01", ("UTCI",), transaction_dir=tmp_path / "tx-second",
                                  identity={"second": 1}) as second:
            for i in range(4):
                second.write(i, values(i))
            second.checkpoint(4, state())
            staged_second = second.staging_directory / "shared.zip"
            staged_second.write_bytes(b"second")
            with pytest.raises(PersistenceError, match="owner"):
                second.complete({shared: staged_second})
            assert shared.read_bytes() == b"first"


def test_long_timeline_keeps_one_state_generation_and_collects_failed_commit(tmp_path, monkeypatch):
    metadata = RasterMetadata(2, 3, (1., 2., 0., 8., 0., -2.), "")
    met = np.zeros((48, 4))
    met[:, 2] = np.arange(48) % 24
    def create(resume=False):
        return TransactionalOutputs(tmp_path / "output_folder" / "0_0", "0_0", metadata, met,
                                    "2020-06-01", ("UTCI", "TMRT"), transaction_dir=tmp_path / "tx",
                                    identity={"long-timeline": 48}, resume=resume)
    with create() as writer:
        for i in range(24):
            writer.write(i, values(i))
            writer.checkpoint(i + 1, state())
            generations = list(writer.transaction_directory.glob("state-*"))
            assert len(generations) == 1
            assert generations[0].name == writer._record["checkpoint"]["generation"]
        committed_generation = generations[0]
        retained_bytes = sum(path.stat().st_size for path in committed_generation.iterdir())
        assert writer._stage_path("UTCI").exists()
        writer.write(24, values(24))
        with monkeypatch.context() as patch:
            real = persistence._atomic_json
            def atomic(path, record):
                if path == writer.record_path:
                    raise OSError("interrupted before cursor publication")
                real(path, record)
            patch.setattr(persistence, "_atomic_json", atomic)
            with pytest.raises(OSError, match="cursor publication"):
                writer.checkpoint(25, state())
        assert committed_generation.exists()
        assert len(list(writer.transaction_directory.glob("state-*"))) == 2
        assert not writer.publication_path.exists()
    with create(resume=True) as writer:
        assert writer.restore()[0] == 24
        assert_state(writer.restore()[1], state())
        assert list(writer.transaction_directory.glob("state-*")) == [committed_generation]
        for i in range(24, 48):
            writer.write(i, values(i))
            writer.checkpoint(i + 1, state())
            generations = list(writer.transaction_directory.glob("state-*"))
            assert len(generations) == 1
            assert sum(path.stat().st_size for path in generations[0].iterdir()) == retained_bytes
        writer.complete()
        assert len(list(writer.transaction_directory.glob("state-*"))) == 1
        assert writer.completion_path.exists()


def test_successful_commit_prunes_orphans_without_reopening_writer(setup, monkeypatch):
    _, _, _, create = setup
    with create() as writer:
        writer.write(0, values(0))
        writer.checkpoint(1, state())
        writer.write(1, values(1))
        with monkeypatch.context() as patch:
            real = persistence._atomic_json
            def atomic(path, record):
                if path == writer.record_path:
                    raise OSError("before commit")
                real(path, record)
            patch.setattr(persistence, "_atomic_json", atomic)
            for _ in range(3):
                with pytest.raises(OSError, match="before commit"):
                    writer.checkpoint(2, state())
        assert len(list(writer.transaction_directory.glob("state-*"))) == 4
        writer.checkpoint(2, state())
        assert len(list(writer.transaction_directory.glob("state-*"))) == 1
        assert writer._stage_path("UTCI").exists()
        finish(writer, 2)
        assert writer.publication_path.exists()
        assert writer.completion_path.exists()
