#!/usr/bin/env python3
"""Frozen same-host CPU candidate pairs, with exact output/state checks.

This is candidate-versus-candidate evidence, never an upstream speedup. Each
cell has its own sources/caches; warmed cells share only their own setup across
repetitions. Successful earlier output histories are replaced only after their
comparison and content inventories have been durably recorded. Failed outputs
and the last successful outputs remain available for inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

import run_p7_serial_variant as base

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("UTCI", "TMRT", "Kup", "Kdown", "Lup", "Ldown", "Shadow", "WBGT", "Ta", "Wind")
SVF_FIELDS = ("svf", "svfE", "svfS", "svfW", "svfN", "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
              "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg")
REGIMES = {"first_use", "geometry_cold", "geometry_warm"}
LIMIT = 12 * 1024**3
RESERVE = 10 * 1024**3
STATE_FIELDS = {"CI", "firstdaytime", "timestepdec", "timeadd", "Tgmap1", "Tgmap1E", "Tgmap1S",
                "Tgmap1W", "Tgmap1N", "TgOut1", "Twater"}


def write_json(path, value):
    """Commit evidence before any successful output history is replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def exact_array(left, right):
    """Finite bits (including -0), special masks, dtype and shape must agree."""
    import numpy as np

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        if left.dtype.kind == "c":
            return exact_array(left.real, right.real) and exact_array(left.imag, right.imag)
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            return False
        if not np.array_equal(np.isposinf(left), np.isposinf(right)):
            return False
        if not np.array_equal(np.isneginf(left), np.isneginf(right)):
            return False
        finite = np.isfinite(left)
        return left[finite].tobytes() == right[finite].tobytes()
    return left.tobytes() == right.tobytes()


def _nodata_equal(left, right):
    import math
    return left == right or (isinstance(left, float) and isinstance(right, float)
                             and math.isnan(left) and math.isnan(right))


def compare_tiff(left, right):
    from osgeo import gdal

    gdal.UseExceptions()
    a, b = gdal.Open(str(left)), gdal.Open(str(right))
    signature = lambda ds: (ds.RasterXSize, ds.RasterYSize, ds.RasterCount,
                           ds.GetGeoTransform(), ds.GetProjection(), ds.GetMetadata())
    passed = signature(a) == signature(b)
    records = []
    if passed:
        for index in range(1, a.RasterCount + 1):
            x, y = a.GetRasterBand(index), b.GetRasterBand(index)
            schema = (x.DataType == y.DataType and x.GetMetadata() == y.GetMetadata()
                      and x.GetDescription() == y.GetDescription()
                      and _nodata_equal(x.GetNoDataValue(), y.GetNoDataValue())
                      and x.GetMaskFlags() == y.GetMaskFlags())
            mismatches = 0
            for row in range(0, a.RasterYSize, 256):
                for col in range(0, a.RasterXSize, 256):
                    window = (col, row, min(256, a.RasterXSize-col), min(256, a.RasterYSize-row))
                    valid = exact_array(x.ReadAsArray(*window), y.ReadAsArray(*window))
                    valid &= exact_array(x.GetMaskBand().ReadAsArray(*window),
                                         y.GetMaskBand().ReadAsArray(*window))
                    mismatches += not valid
            records.append({"band": index, "schema_passed": schema, "mismatched_windows": mismatches})
            passed &= schema and mismatches == 0
    a = b = None
    return {"passed": bool(passed), "bands": records}


def _archive_members(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in sorted(names):
            value = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    value.update(block)
            result[name] = value.hexdigest()
    return result


def checkpoint_inventory(scene, timesteps):
    result = {}
    for path in sorted((scene / ".solweig-light/transactions").glob("*/transaction.json")):
        record = json.loads(path.read_text())
        checkpoint = record["checkpoint"]
        if checkpoint["next_timestep"] != timesteps:
            raise ValueError("Incomplete chronological checkpoint")
        generation = checkpoint["generation"]
        if Path(generation).name != generation:
            raise ValueError("Unsafe checkpoint generation path")
        state = path.parent / generation
        if base.digest(state / "state.json") != checkpoint["state_sha256"]:
            raise ValueError("Checkpoint state manifest hash mismatch")
        payload = json.loads((state / "state.json").read_text())
        if payload.get("schema") != 1 or set(payload.get("fields", {})) != STATE_FIELDS:
            raise ValueError("Incomplete checkpoint state schema/fields")

        def verify(value):
            if isinstance(value, dict):
                if "file" in value:
                    name = value["file"]
                    if Path(name).name != name or base.digest(state / name) != value["sha256"]:
                        raise ValueError("Checkpoint array hash mismatch")
                for child in value.values():
                    verify(child)
            elif isinstance(value, list):
                for child in value:
                    verify(child)
        verify(payload)
        key = "|".join(sorted(Path(name).name for name in record["signature"]["outputs"]))
        if key in result:
            raise ValueError("Duplicate output transaction")
        completion_path = path.parent / "complete.json"
        if not completion_path.is_file():
            raise ValueError("Missing output completion record")
        completion = json.loads(completion_path.read_text())
        signature = record["signature"]
        if completion.get("signature") != signature or len(signature.get("timestamps", [])) != timesteps:
            raise ValueError("Completion signature/cursor mismatch")
        requested = signature["outputs"]
        if len(requested) != len(FIELDS) or {Path(name).stem.split("_", 1)[0] for name in requested} != set(FIELDS):
            raise ValueError("Incomplete requested output set")
        artifacts = completion.get("artifacts", [])
        finals = [item["final"] for item in artifacts]
        if len(finals) != len(set(finals)) or not set(requested).issubset(finals):
            raise ValueError("Completion omits or duplicates output artifacts")
        for artifact in artifacts:
            final = Path(artifact["final"])
            if not final.is_absolute() or not final.resolve().is_relative_to(scene.resolve()):
                raise ValueError("Completion artifact escaped owned scene")
            if base.digest(final) != artifact["sha256"]:
                raise ValueError("Completed artifact hash mismatch")
        result[key] = {"next_timestep": checkpoint["next_timestep"], "files": base.tree_inventory(state)}
    if not result:
        raise ValueError("No complete chronological checkpoint")
    return result


def compare_scenes(left, right, shape, timesteps=24, expected_tiles=("0_0",)):
    """Streaming exact TIFF checks plus lossless geometry and final state."""
    from osgeo import gdal

    inventories = [{str(p.relative_to(scene)) for p in (scene / "output_folder").glob("*/*.tif")}
                   for scene in (left, right)]
    if not inventories[0] or inventories[0] != inventories[1]:
        return {"passed": False, "reason": "Output inventory mismatch"}
    records = {}
    fields = {}
    for name in sorted(inventories[0]):
        field = Path(name).stem.split("_", 1)[0]
        fields.setdefault(Path(name).parent.name, set()).add(field)
        dataset = gdal.Open(str(left / name))
        expected_bands = timesteps if field in FIELDS else 1
        expected = ([dataset.RasterYSize, dataset.RasterXSize] == shape
                    and dataset.RasterCount == expected_bands)
        dataset = None
        records[name] = compare_tiff(left / name, right / name)
        records[name]["expected_schema"] = expected
        records[name]["passed"] &= expected
    if (set(fields) != set(expected_tiles)
            or any(not set(FIELDS).issubset(names) or names - set(FIELDS) - {"SVF"} for names in fields.values())):
        return {"passed": False, "reason": "Missing or extra requested fields", "records": records}
    geometry = []
    for scene in (left, right):
        container = scene / "processed_inputs/SVF"
        archives = {str(path.relative_to(container)): _archive_members(path)
                    for path in sorted(container.glob("*")) if path.suffix in {".zip", ".npz"}}
        geometry.append(archives)
    geometry_passed = bool(geometry[0]) and geometry[0] == geometry[1]
    for archives in geometry:
        geometry_passed &= len(archives) == 2
        for name, members in archives.items():
            expected = ({field + ".tif" for field in SVF_FIELDS} if name.endswith(".zip")
                        else {field + ".npy" for field in ("shadowmat", "vegshadowmat", "vbshmat")})
            geometry_passed &= set(members) == expected
    total_paths = [{path.name for path in (scene / "processed_inputs/SVF").glob("SkyViewFactor_*.tif")}
                   for scene in (left, right)]
    total_records = {}
    geometry_passed &= bool(total_paths[0]) and total_paths[0] == total_paths[1]
    if total_paths[0] == total_paths[1]:
        for name in sorted(total_paths[0]):
            total_records[name] = compare_tiff(left / "processed_inputs/SVF" / name,
                                                right / "processed_inputs/SVF" / name)
            geometry_passed &= total_records[name]["passed"]
    state = [checkpoint_inventory(scene, timesteps) for scene in (left, right)]
    passed = all(record["passed"] for record in records.values()) and geometry_passed and state[0] == state[1]
    return {"passed": bool(passed), "records": records, "geometry_archives": geometry,
            "geometry_total_rasters": total_records,
            "geometry_passed": geometry_passed, "final_state": state,
            "final_state_passed": state[0] == state[1],
            "limitation": "Final state alone is not every-timestep evidence; separate trace admission is required."}


def monitored(command, run, env, runtime):
    """Run one owned process group and retain measurement/cleanup evidence.

    This is a drop-in replacement for tools/run_exact_cpu_pairs.py::monitored.
    It relies only on names already imported by that module plus its LIMIT,
    RESERVE, and base constants/modules.
    """
    import psutil

    sample_interval = .02
    cleanup_timeout = 5.0
    maximum_trial_seconds = runtime.get("maximum_trial_seconds")
    if (isinstance(maximum_trial_seconds, bool)
            or not isinstance(maximum_trial_seconds, (int, float))
            or maximum_trial_seconds <= 0):
        raise ValueError("An explicit positive maximum_trial_seconds is required")

    run.mkdir(parents=True, exist_ok=False)
    peak, samples, abort_reason = 0, 0, None
    monitor_error = None
    pending_error = None
    pending_traceback = None
    launcher_returncode = None
    process = None
    started = time.perf_counter()
    worker_finished_at = None
    cleanup_started_at = None
    cleanup_finished_at = None
    cleanup = {
        "timeout_seconds": cleanup_timeout,
        "ownership": None,
        "launcher_exited_with_live_group": False,
        "pre_cleanup_owned_pids": [],
        "pre_cleanup_group_exists": False,
        "group_signal": "not_attempted",
        "launcher_wait": "not_attempted",
        "post_cleanup_owned_pids": [],
        "post_cleanup_group_exists": False,
        "errors": [],
        "passed": False,
    }

    def error_record(error):
        return {
            "type": type(error).__name__,
            "module": type(error).__module__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    def group_exists(pgid):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # A permission failure still proves that the group exists.
            return True

    def owned_pids(pgid, sid):
        """Identify only members of the session created for this invocation."""
        result = []
        for pid in psutil.pids():
            try:
                if os.getpgid(pid) == pgid and os.getsid(pid) == sid:
                    result.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        return sorted(result)

    with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err, \
            (run / "rss_samples.jsonl").open("w") as sample_file:
        process = subprocess.Popen(
            command, cwd=run.parent, env=env, stdout=out, stderr=err,
            start_new_session=True,
        )
        pgid = process.pid
        sid = process.pid
        ownership = {
            "launcher_pid": process.pid,
            "expected_process_group": pgid,
            "expected_session": sid,
            "start_new_session": True,
            # Popen returns only after the child-side start_new_session action
            # succeeds (otherwise it raises). A live OS check strengthens that
            # contract but is allowed to race a very short-lived launcher.
            "basis": "popen_start_new_session_contract",
            "verified": True,
            "live_verified": False,
        }
        cleanup["ownership"] = ownership
        try:
            observed_pgid = os.getpgid(process.pid)
            observed_sid = os.getsid(process.pid)
            ownership.update(observed_process_group=observed_pgid,
                             observed_session=observed_sid,
                             live_verified=observed_pgid == pgid and observed_sid == sid,
                             verified=observed_pgid == pgid and observed_sid == sid)
            if not ownership["verified"]:
                abort_reason = "session_ownership_mismatch"
        except ProcessLookupError as error:
            ownership["live_verification_unavailable"] = {
                "type": type(error).__name__, "message": str(error),
            }
        except BaseException as error:
            ownership["verification_error"] = {
                "type": type(error).__name__, "message": str(error),
            }
            monitor_error = error_record(error)
            pending_error = error
            pending_traceback = error.__traceback__
            abort_reason = "monitor_exception"

        try:
            while abort_reason is None:
                # Construct the root and every member afresh on each 20 ms
                # sample. Reusing a pre-exec psutil.Process can silently lose
                # descendants on macOS.
                try:
                    root = psutil.Process(process.pid)
                    member_pids = [process.pid]
                    member_pids.extend(child.pid for child in root.children(recursive=True))
                except psutil.NoSuchProcess:
                    member_pids = []

                rss = 0
                sampled_pids = []
                for pid in dict.fromkeys(member_pids):
                    try:
                        member = psutil.Process(pid)
                        rss += member.memory_info().rss
                        sampled_pids.append(pid)
                    except psutil.NoSuchProcess:
                        pass

                elapsed = time.perf_counter() - started
                sample_file.write(json.dumps([elapsed, rss, sampled_pids]) + "\n")
                sample_file.flush()
                samples += 1
                peak = max(peak, rss)

                if rss > LIMIT:
                    abort_reason = "process_tree_rss_limit"
                elif ((samples == 1 or samples % 50 == 0)
                      and shutil.disk_usage(run).free < RESERVE):
                    abort_reason = "disk_reserve"
                elif elapsed > maximum_trial_seconds:
                    abort_reason = "time_limit"

                launcher_returncode = process.poll()
                if launcher_returncode is not None:
                    worker_finished_at = time.perf_counter()
                if abort_reason is not None or launcher_returncode is not None:
                    break
                time.sleep(sample_interval)
        except BaseException as error:
            monitor_error = error_record(error)
            pending_error = error
            pending_traceback = error.__traceback__
            if abort_reason is None:
                abort_reason = "monitor_exception"
        finally:
            if worker_finished_at is None:
                worker_finished_at = time.perf_counter()
            cleanup_started_at = time.perf_counter()
            # poll() reaps an exited launcher. Its process group may still own
            # live descendants, so group cleanup must not depend on poll().
            launcher_returncode = process.poll()
            try:
                pre_owned = owned_pids(pgid, sid) if ownership["verified"] else []
                pre_group_exists = group_exists(pgid) if ownership["verified"] else False
                cleanup["pre_cleanup_owned_pids"] = pre_owned
                cleanup["pre_cleanup_group_exists"] = pre_group_exists
                cleanup["launcher_exited_with_live_group"] = bool(
                    launcher_returncode is not None and pre_owned
                )
                if cleanup["launcher_exited_with_live_group"] and abort_reason is None:
                    abort_reason = "launcher_exited_with_descendants"

                if ownership["verified"] and pre_owned:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                        cleanup["group_signal"] = "sent_sigkill"
                    except ProcessLookupError:
                        cleanup["group_signal"] = "group_disappeared_before_signal"
                elif ownership["verified"] and not pre_group_exists:
                    cleanup["group_signal"] = "group_absent"
                elif ownership["verified"]:
                    # A group exists but no process can be attributed to the
                    # verified session. Do not signal an unattributed group.
                    cleanup["group_signal"] = "skipped_unattributed_group"
                else:
                    cleanup["group_signal"] = "skipped_unverified_ownership"

                if process.poll() is None:
                    # Killing the exact Popen child is safe even if group
                    # ownership verification failed.
                    if cleanup["group_signal"] != "sent_sigkill":
                        process.kill()
                    try:
                        launcher_returncode = process.wait(timeout=cleanup_timeout)
                        cleanup["launcher_wait"] = "reaped"
                    except subprocess.TimeoutExpired:
                        process.kill()
                        launcher_returncode = process.wait(timeout=cleanup_timeout)
                        cleanup["launcher_wait"] = "killed_after_wait_timeout"
                else:
                    launcher_returncode = process.wait()
                    cleanup["launcher_wait"] = "already_exited_reaped"

                deadline = time.perf_counter() + cleanup_timeout
                while ownership["verified"]:
                    remaining_owned = owned_pids(pgid, sid)
                    remaining_group = group_exists(pgid)
                    if not remaining_owned and not remaining_group:
                        break
                    if time.perf_counter() >= deadline:
                        break
                    time.sleep(sample_interval)
                cleanup["post_cleanup_owned_pids"] = (
                    owned_pids(pgid, sid) if ownership["verified"] else []
                )
                cleanup["post_cleanup_group_exists"] = (
                    group_exists(pgid) if ownership["verified"] else False
                )
            except BaseException as error:
                cleanup["errors"].append(error_record(error))

            cleanup["passed"] = bool(
                ownership["verified"]
                and not cleanup["post_cleanup_owned_pids"]
                and not cleanup["post_cleanup_group_exists"]
                and not cleanup["errors"]
                and cleanup["launcher_wait"] in {
                    "reaped", "already_exited_reaped", "killed_after_wait_timeout"
                }
            )
            if not cleanup["passed"] and abort_reason is None:
                abort_reason = "cleanup_failure"
            cleanup_finished_at = time.perf_counter()
            for stream in (out, err, sample_file):
                stream.flush()
                os.fsync(stream.fileno())

    rss_valid = samples > 0 and peak > (1 << 20)
    result = {
        "command": command,
        "returncode": launcher_returncode,
        "elapsed_seconds": worker_finished_at - started,
        "cleanup_elapsed_seconds": cleanup_finished_at - cleanup_started_at,
        "supervisor_elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "Launch through first observed launcher exit; cleanup and evidence sync excluded. Poll interval is 20 ms.",
        "sampled_process_tree_peak_rss_bytes": peak,
        "samples": samples,
        "memory_limit_aborted": abort_reason == "process_tree_rss_limit",
        "abort_reason": abort_reason,
        "memory_limit_bytes": LIMIT,
        "disk_reserve_bytes": RESERVE,
        "maximum_trial_seconds": maximum_trial_seconds,
        "sample_interval_seconds": sample_interval,
        "rss_valid": rss_valid,
        "cleanup": cleanup,
        "monitor_error": monitor_error,
        "monitoring_passed": bool(
            abort_reason is None and monitor_error is None and cleanup["passed"] and rss_valid
        ),
        "rss_caveat": (
            "Summed RSS double-counts shared pages; 20 ms sampling can miss shorter peaks."
        ),
    }
    write_json(run / "measurement.json", result)
    if pending_error is not None:
        raise pending_error.with_traceback(pending_traceback)
    return result


def child(spec_path):
    # base._child records the actual imported source and full dependency list.
    code = base._child(spec_path)
    if code == 0:
        import numba
        spec = json.loads(spec_path.read_text())
        threads = numba.get_num_threads()
        write_json(Path(spec["process_run"]) / "launcher_threads.json", {
            "actual_numba_threads": threads, "requested": spec["runtime"]["threads_per_worker"],
            "scope": "launcher; tile subprocess environment is set by the frozen runtime scheduler"})
        if threads != spec["runtime"]["threads_per_worker"]:
            return 1
    return code


def invoke(source, entrypoint, kwargs, options, run, jit, protocol, python):
    spec = {"source": str(source), "process_run": str(run), "entrypoint": entrypoint,
            "kwargs": kwargs, "runtime": options}
    path = run.parent / (run.name + ".spec.json")
    write_json(path, spec)
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", NUMBA_CACHE_DIR=str(jit))
    env.update({key: str(options["threads_per_worker"]) for key in base.THREAD_ENV})
    measurement = monitored([python, str(Path(__file__).resolve()), "--child", str(path)], run, env, protocol["runtime"])
    status = base.execution_status(measurement, run, Path(kwargs["base_path"]))
    status["passed"] &= measurement["monitoring_passed"]
    return {"measurement": measurement, "status": status}


def bindings():
    return {"harness": base.digest(__file__), "helpers": base.digest(base.__file__)}


def validate(protocol_path, execute=False):
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema_version"] != 1 or protocol["purpose"] not in {"development_selection", "promotion"}:
        raise ValueError("Unsupported exact-pair protocol")
    if execute and (protocol["status"] != "reviewed_frozen" or protocol["bindings"] != bindings()):
        raise ValueError("Execution requires reviewed, hash-bound protocol")
    if protocol["timesteps"] != 24 or protocol["patches"] != 153 or protocol["runtime"]["checkpoint_interval"] != 1:
        raise ValueError("Full 24-step / 153-patch / per-step durability workload required")
    if protocol["runtime"]["maximum_trial_seconds"] <= 0:
        raise ValueError("An explicit positive trial time limit is required")
    if "baseline" not in protocol["sources"]:
        raise ValueError("Baseline source is required")
    for name, source in protocol["sources"].items():
        inventory = base.tree_inventory(ROOT / source["path"])
        if not inventory or base.value_digest(inventory) != source["inventory_sha256"]:
            raise ValueError(f"Source binding mismatch: {name}")
        if execute:
            admission = source["admission"]
            path = ROOT / admission["path"]
            if base.digest(path) != admission["sha256"]:
                raise ValueError(f"Changed admission evidence: {name}")
            report = json.loads(path.read_text())
            if report.get("status") != "passed" or report.get("source_inventory_sha256") != source["inventory_sha256"]:
                raise ValueError(f"Admission does not qualify this source: {name}")
            if protocol["purpose"] == "promotion" and not report.get("every_timestep_exactness_passed"):
                raise ValueError(f"Promotion requires separate every-timestep exactness: {name}")
    for name, fixture in protocol["fixtures"].items():
        if base.fixture_inventory(fixture) != fixture["files"] or base.digest(ROOT / fixture["kwargs"]) != fixture["kwargs_sha256"]:
            raise ValueError(f"Fixture binding mismatch: {name}")
        from osgeo import gdal
        dataset = gdal.Open(str(ROOT / fixture["scene"] / "Building_DSM.tif"))
        if [dataset.RasterYSize, dataset.RasterXSize] != fixture["shape"]:
            raise ValueError("Fixture raster dimensions differ from frozen workload")
        dataset = None
        rows = [line for line in (ROOT / fixture["scene"] / "met.txt").read_text().splitlines() if line.strip()]
        if len(rows)-1 != 24:
            raise ValueError("Fixture does not contain the complete 24-step meteorology")
        kwargs = json.loads((ROOT / fixture["kwargs"]).read_text())
        if any(kwargs.get("save_"+flag) is not True for flag in base.FLAGS):
            raise ValueError("Fixture does not request every required output")
        if kwargs["tile_size"] < max(fixture["shape"]) or fixture["expected_tiles"] != ["0_0"]:
            raise ValueError("This protocol version supports intact single-logical-tile fixtures")
    for cell in protocol["cells"]:
        if cell["regime"] not in REGIMES or cell["budget"] not in {1, 4, 10}:
            raise ValueError("Unknown execution regime or native budget")
        if cell["variant"] == "baseline" or cell["variant"] not in protocol["sources"]:
            raise ValueError("Missing distinct candidate source label")
        if min(cell["baseline_block_pixels"], cell["candidate_block_pixels"], cell["repetitions"]) < 1:
            raise ValueError("Nonpositive cell size")
        if cell["fixture"] not in protocol["fixtures"]:
            raise ValueError("Missing fixture")
        if protocol["purpose"] == "promotion" and cell["repetitions"] < 5:
            raise ValueError("Promotion measurements require at least five pairs per cell")
    if not protocol["cells"]:
        raise ValueError("Empty matrix")
    return protocol


def schedule(protocol):
    generator = random.Random(protocol["seed"])
    result = []
    for index, cell in enumerate(protocol["cells"]):
        initial = generator.randrange(2)
        orders = [["baseline", "candidate"] if (rep+initial) % 2 == 0 else ["candidate", "baseline"]
                  for rep in range(cell["repetitions"])]
        result.append({"cell": index, **cell, "orders": orders})
    generator.shuffle(result)
    return result


def execute(protocol_path, output, python):
    protocol = validate(protocol_path, execute=True)
    sources = {}
    for name, source in protocol["sources"].items():
        destination = output / "sources" / name / "src"
        destination.parent.mkdir(parents=True)
        inventory = base._copy_source(ROOT / source["path"], destination)
        if base.value_digest(inventory) != source["inventory_sha256"]:
            raise ValueError("Source changed during snapshot")
        sources[name] = destination
    protocol_hash = base.digest(protocol_path)
    frozen_bindings = bindings()
    plan = schedule(protocol)
    write_json(output / "frozen.json", {"protocol": protocol, "protocol_sha256": protocol_hash,
                    "bindings": frozen_bindings, "hardware": base.hardware_inventory(), "schedule": plan,
                    "python": python, "packages": base.package_environment()})

    def guard():
        if base.digest(protocol_path) != protocol_hash or bindings() != frozen_bindings:
            raise ValueError("Harness/protocol changed during run")
        for name, source in sources.items():
            if base.value_digest(base.tree_inventory(source)) != protocol["sources"][name]["inventory_sha256"]:
                raise ValueError("Frozen source changed")
        validate(protocol_path, execute=True)
        if shutil.disk_usage(output).free < RESERVE:
            raise RuntimeError("10 GiB disk reserve exhausted")

    def input_guard(scene, fixture):
        actual = {name: base.digest(scene / name) for name in fixture["files"]}
        if actual != fixture["files"]:
            raise ValueError("Copied workload changed before/after worker execution")
        return actual

    completed = []
    for cell in plan:
        guard()
        root = output / f"cell-{cell['cell']:03d}"
        fixture = protocol["fixtures"][cell["fixture"]]
        work = {}
        for side in ("baseline", "candidate"):
            scene, kwargs = base.prepare_scene(fixture, root / side / "scene")
            input_guard(scene, fixture)
            options = {"cpu_budget": cell["budget"], "workers": 1, "threads_per_worker": cell["budget"],
                       "memory_budget_bytes": LIMIT, "block_pixels": cell[side+"_block_pixels"],
                       "cache_dir": str(scene / ".runtime_cache"), "cache_enabled": True,
                       "checkpoint_interval": 1, "legacy_cache_policy": "recompute", "resume": False}
            label = "baseline" if side == "baseline" else cell["variant"]
            work[side] = {"scene": scene, "kwargs": kwargs, "runtime": options,
                          "source": sources[label], "jit": root / side / "jit"}
        # Setup never enters the measured distribution. Both sides get a full
        # real setup; independent caches must not be transplanted between sides.
        if cell["regime"] != "first_use":
            setups = {}
            for side in cell["orders"][0]:
                guard()
                w = work[side]
                input_guard(w["scene"], fixture)
                setups[side] = invoke(w["source"], "thermal_comfort", w["kwargs"], w["runtime"],
                                      root / side / "setup", w["jit"], protocol, python)
                input_guard(w["scene"], fixture)
                write_json(root / "setup.json", setups)
                if not setups[side]["status"]["passed"]:
                    raise RuntimeError("Setup failed; outputs and evidence retained")
            setup_comparison = compare_scenes(work["baseline"]["scene"], work["candidate"]["scene"], fixture["shape"])
            write_json(root / "setup_comparison.json", setup_comparison)
            if not setup_comparison["passed"]:
                raise RuntimeError("Setup exactness failed; outputs and evidence retained")
        pairs = []
        for repetition, order in enumerate(cell["orders"]):
            pair = {"repetition": repetition, "order": order, "trials": {}}
            for side in order:
                guard()
                w = work[side]
                input_guard(w["scene"], fixture)
                if cell["regime"] != "first_use" or repetition:
                    reset = base.reset_warm_state(w["scene"])
                    if not reset["passed"]:
                        raise RuntimeError("Warm reset altered retained geometry")
                if cell["regime"] != "geometry_warm":
                    for relative in ("processed_inputs", ".runtime_cache"):
                        target = w["scene"] / relative
                        if target.is_dir():
                            shutil.rmtree(target)
                    if cell["regime"] == "first_use" and w["jit"].exists():
                        shutil.rmtree(w["jit"])
                before = {"jit": base.cache_inventory(w["jit"]), "geometry": base.retained_geometry_snapshot(w["scene"])}
                warm = cell["regime"] == "geometry_warm"
                result = invoke(w["source"], "run_utci_tiles" if warm else "thermal_comfort",
                                base.warm_kwargs(w["kwargs"], w["scene"]) if warm else w["kwargs"],
                                w["runtime"], root / side / f"r{repetition:02d}", w["jit"], protocol, python)
                result["copied_input_inventory"] = input_guard(w["scene"], fixture)
                after = {"jit": base.cache_inventory(w["jit"]), "geometry": base.retained_geometry_snapshot(w["scene"])}
                if warm:
                    cache_valid = before == after and before["jit"]["file_count"] > 0 and before["geometry"]["total_bytes"] > 0
                else:
                    jit_valid = before["jit"]["file_count"] == 0 if cell["regime"] == "first_use" else before["jit"]["file_count"] > 0
                    cache_valid = jit_valid and before["geometry"]["total_bytes"] == 0 and after["geometry"]["total_bytes"] > 0 and after["jit"]["file_count"] > 0
                result.update(cache_valid=cache_valid, cache_before=before, cache_after=after,
                              newly_written_jit_entries=sorted(set(after["jit"]["files"])-set(before["jit"]["files"])),
                              changed_jit_entries=sorted(name for name in set(before["jit"]["files"]) & set(after["jit"]["files"])
                                                         if before["jit"]["files"][name] != after["jit"]["files"][name]),
                              output_inventory=base.tree_inventory(w["scene"] / "output_folder"))
                pair["trials"][side] = result
                write_json(root / f"pair-{repetition:02d}.json", pair)
                if not result["status"]["passed"] or not cache_valid:
                    raise RuntimeError("Measured trial or cache proof failed; retained without reset")
            comparison = compare_scenes(work["baseline"]["scene"], work["candidate"]["scene"], fixture["shape"])
            pair["comparison"] = comparison
            pair["passed"] = comparison["passed"]
            a = pair["trials"]["baseline"]["measurement"]["elapsed_seconds"]
            b = pair["trials"]["candidate"]["measurement"]["elapsed_seconds"]
            pair["baseline_over_candidate"] = a/b
            write_json(root / f"pair-{repetition:02d}.json", pair)
            if not pair["passed"]:
                raise RuntimeError("Exact comparison failed; outputs retained without reset")
            pairs.append(pair)
        ratios = [pair["baseline_over_candidate"] for pair in pairs]
        summary = {"cell": cell, "pair_count": len(pairs), "passed": True, "ratios": ratios,
                   "median_baseline_over_candidate": statistics.median(ratios), "ratio_range": [min(ratios), max(ratios)],
                   "performance_claim_eligible": False,
                   "eligibility_reason": "Only the final complete-matrix report can authorize a scoped claim.",
                   "upstream_speedup_claim_eligible": False}
        write_json(root / "summary.json", summary)
        completed.append(summary)
        write_json(output / "completed_cells.json", completed)
    guard()
    write_json(output / "summary.json", {"status": "passed", "cells": completed,
                    "source_and_fixture_guards_passed": True,
                    "performance_claim_eligible": False,
                    "eligibility_reason": "Requires a separate reviewed selection report evaluating the frozen benefit/regression/uncertainty gates.",
                    "upstream_speedup_claim_eligible": False,
                    "limitations": ["OS page cache uncontrolled; no disk-cold claim.",
                        "Compatible JIT warm still includes compilation for deliberately uncached kernels.",
                        "Single host and candidate baseline; no original-upstream speedup claim.",
                        "Only last output histories retained; every trial keeps content hashes and exact comparison."]})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", type=Path)
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child)
    if args.protocol is None:
        parser.error("--protocol is required")
    protocol = validate(args.protocol, execute=args.execute)
    if not args.execute:
        print(json.dumps({"status": "validated_not_executed", "bindings": bindings(), "schedule": schedule(protocol)}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required with --execute")
    # Allocate exclusively before entering the failure handler: never write a
    # failure record into an earlier run when the requested path already exists.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        return execute(args.protocol.resolve(), args.output, args.python)
    except Exception:
        if args.output.is_dir():
            write_json(args.output / "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
