"""Exact permanent regression for runtime GVF serial/parallel dispatch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numba
import numpy as np
import pytest
import solweig_light
import solweig_light.runtime as runtime_module
from solweig_light.radiation import engine, ground_view
from solweig_light.runtime import RuntimeOptions, runtime_options


def repository_root():
    start = Path(__file__).resolve()
    for candidate in start.parents:
        if (candidate / "tests/reference/ground_view_original_cpu/manifest.json").is_file():
            return candidate
    raise RuntimeError("could not locate ground-view reference repository root")


REPOSITORY = repository_root()
REFERENCE = REPOSITORY / "tests/reference/ground_view_original_cpu"
MANIFEST = json.loads((REFERENCE / "manifest.json").read_text())
CASES = [case for case in MANIFEST["cases"] if case["function"] == "gvf_2018a"]
PACKAGE = Path(solweig_light.__file__).resolve().parent
AVAILABLE_THREADS = max(
    1,
    min(
        int(runtime_module._total_cpus()),
        int(numba.config.NUMBA_NUM_THREADS),
    ),
)


def effective_threads(requested):
    """Exercise each requested budget at the largest locally valid value."""
    return max(1, min(int(requested), AVAILABLE_THREADS))


def fresh(case):
    artifact = case["input_artifact"]
    path = REFERENCE / artifact["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key][()] if archive[key].ndim == 0 else archive[key].copy()
            for key in archive.files
        }


def exact(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    if left.dtype.kind == "f":
        np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
        valid = ~np.isnan(left)
        unsigned = "u" + str(left.dtype.itemsize)
        np.testing.assert_array_equal(
            left[valid].view(unsigned), right[valid].view(unsigned)
        )
    else:
        np.testing.assert_array_equal(left, right)


def test_active_ground_view_modules_share_installed_package_origin():
    for module in (engine, ground_view, runtime_module):
        assert Path(module.__file__).resolve().is_relative_to(PACKAGE)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("requested_threads", [1, 4, 10], ids=lambda value: f"budget-{value}")
def test_dispatch_preserves_every_field_mutation_and_error(
        case, requested_threads, monkeypatch, record_property):
    threads = effective_threads(requested_threads)
    record_property("requested_threads", requested_threads)
    record_property("effective_threads", threads)
    calls = []

    def forwarding_spy(name, function):
        def invoke(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)
        return invoke

    for name in ("gvf_2018a", "gvf_2018a_parallel"):
        monkeypatch.setattr(ground_view, name,
                            forwarding_spy(name, getattr(ground_view, name)))
    direct_inputs = fresh(case)
    dispatched_inputs = fresh(case)
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        options = RuntimeOptions(
            cpu_budget=threads,
            workers=1,
            threads_per_worker=threads,
        )
        assert options.cpu_budget == threads
        assert options.threads_per_worker == threads
        with np.errstate(all="ignore"):
            if case["status"] == "failed":
                error = {
                    "RuntimeError": RuntimeError,
                    "UnboundLocalError": UnboundLocalError,
                }[case["exception"]]
                with pytest.raises(error):
                    ground_view.gvf_2018a(**direct_inputs)
                calls.clear()
                with runtime_options(options):
                    with pytest.raises(error):
                        engine.gvf_2018a(**dispatched_inputs)
            else:
                expected = ground_view.gvf_2018a(**direct_inputs)
                calls.clear()
                with runtime_options(options):
                    actual = engine.gvf_2018a(**dispatched_inputs)
                assert len(actual) == len(expected) == 17
                for expected_field, actual_field in zip(expected, actual, strict=True):
                    exact(expected_field, actual_field)
        assert calls == ["gvf_2018a_parallel" if threads > 1 else "gvf_2018a"]
        assert direct_inputs.keys() == dispatched_inputs.keys()
        for key in direct_inputs:
            exact(direct_inputs[key], dispatched_inputs[key])
    finally:
        numba.set_num_threads(previous)
