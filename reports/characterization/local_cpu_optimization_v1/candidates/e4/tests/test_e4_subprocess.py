"""Draft subprocess admission tests; deliberately not executed during source-only work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


PACKET = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name("e4_probe.py")
BASELINE = Path(os.environ.get(
    "E4_BASELINE_SRC",
    "/Users/alansynn/Workspace/solweig-light/reports/characterization/"
    "local_cpu_optimization_v1/candidates/baseline/src",
)).resolve()
WALL = PACKET / "wall_candidate/src"
MATH = PACKET / "math_candidate/src"


def run_probe(source, cache, run, mode, *, order="serial-first", debug=False):
    run.mkdir(parents=True)
    result = run / "result.json"
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(source), "EXPECTED_SOURCE_ROOT": str(source),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(cache), "NUMBA_DEBUG_CACHE": "1" if debug else "0",
        "NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    command = [sys.executable, str(PROBE), "--mode", mode, "--order", order,
               "--result", str(result)]
    completed = subprocess.run(command, cwd=run, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(result.read_text()), completed.stdout + completed.stderr


def names(cache):
    return sorted(p.name for p in cache.rglob("*") if p.is_file())


def has_artifact(files, function, suffix):
    pattern = re.compile(rf"(?:^|\.){re.escape(function)}-\d+\.py\d+(?:\.\d+)?\.{suffix}$")
    return any(pattern.search(name) for name in files)


def assert_loaded(debug, function):
    assert re.search(rf"index loaded from '.*{re.escape(function)}-\d+\.py\d+\.nbi'", debug)
    assert re.search(rf"data loaded from '.*{re.escape(function)}-\d+\.py\d+\.\d+\.nbc'", debug)


@pytest.mark.parametrize("order", ["serial-first", "parallel-first"])
def test_wall_distinct_dispatchers_orders_and_persistent_hits(tmp_path, order):
    baseline, _ = run_probe(BASELINE, tmp_path / "baseline-cache", tmp_path / "baseline", "wall", order=order)
    cache = tmp_path / "candidate-cache"
    cold, cold_debug = run_probe(WALL, cache, tmp_path / "cold", "wall", order=order, debug=True)
    warm, warm_debug = run_probe(WALL, cache, tmp_path / "warm", "wall", order=order, debug=True)

    assert cold["origins"] == warm["origins"]
    assert cold["separate_py_funcs"] == {"wall13": True, "wall23": True}
    assert cold["results"] == warm["results"] == baseline["results"]
    assert cold["results"]["serial"] == cold["results"]["parallel"]
    files = names(cache)
    for function in ("_wall13_serial", "_wall23_serial"):
        assert has_artifact(files, function, "nbi")
        assert has_artifact(files, function, "nbc")
        assert "saved to" in cold_debug
        assert_loaded(warm_debug, function)
    # Existing parallel dispatchers intentionally remain uncached.
    for function in ("_wall13", "_wall23"):
        assert not has_artifact(files, function, "nbi")
        assert not has_artifact(files, function, "nbc")


def test_wall_order_permutations_have_independent_serial_cache_names(tmp_path):
    inventories = []
    outputs = []
    for order in ("serial-first", "parallel-first"):
        cache = tmp_path / ("cache-" + order)
        payload, _ = run_probe(WALL, cache, tmp_path / ("run-" + order), "wall", order=order)
        inventories.append({n.split("-")[0] for n in names(cache) if n.endswith((".nbi", ".nbc"))})
        outputs.append(payload["results"])
    assert inventories[0] == inventories[1]
    assert any("_wall13_serial" in n for n in inventories[0])
    assert any("_wall23_serial" in n for n in inventories[0])
    assert outputs[0] == outputs[1]


def test_asvf_cache_hit_and_public_profile_exact_special_bits(tmp_path):
    baseline, _ = run_probe(BASELINE, tmp_path / "baseline-cache", tmp_path / "baseline", "math")
    cache = tmp_path / "candidate-cache"
    cold, cold_debug = run_probe(MATH, cache, tmp_path / "cold", "math", debug=True)
    warm, warm_debug = run_probe(MATH, cache, tmp_path / "warm", "math", debug=True)

    assert cold["outputs"] == warm["outputs"] == baseline["outputs"]
    assert cold["profile_id"] == warm["profile_id"] == baseline["profile_id"]
    assert cold["profile_fingerprint"] == warm["profile_fingerprint"]
    assert cold["profile_fingerprint"] != baseline["profile_fingerprint"]
    assert cold["origins"] == warm["origins"]
    assert cold["lowering"] == warm["lowering"] == baseline["lowering"]
    assert all(
        item == {"explicit_fma": True, "fast_flag_absent": True}
        for item in cold["lowering"].values()
    )
    files = names(cache)
    for function in ("asvf_fma", "tan_array", "atan_array"):
        assert has_artifact(files, function, "nbi")
        assert has_artifact(files, function, "nbc")
        assert "saved to" in cold_debug
        assert_loaded(warm_debug, function)
