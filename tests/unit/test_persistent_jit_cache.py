"""Permanent admission tests for distinct persistent radiation JIT caches."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import solweig_light.radiation.patch_radiation as patch_radiation


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(patch_radiation.__file__).resolve()
PACKAGE_ROOT = Path(__import__("solweig_light").__file__).resolve().parents[1]
PROBE = ROOT / "tests/helpers/persistent_jit_cache_probe.py"


def normalized_body(tree, name):
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = ast.fix_missing_locations(ast.parse(ast.unparse(function)).body[0])
    function.name = "normalized"
    function.decorator_list = []
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == "prange":
            node.id = "range"
    return ast.dump(function, include_attributes=False)


@pytest.mark.parametrize("name", ["_shortwave", "_longwave"])
def test_serial_and_parallel_arithmetic_bodies_are_identical(name):
    tree = ast.parse(SOURCE.read_text())
    assert normalized_body(tree, name) == normalized_body(tree, name + "_serial")


def run_probe(cache, run, mode, *, debug=False):
    run.mkdir()
    result = run / "result.json"
    environment = dict(os.environ)
    environment.update({
        # Bind fresh-process probes to the same distribution imported by pytest.
        "PYTHONPATH": str(PACKAGE_ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(cache),
        "NUMBA_DEBUG_CACHE": "1" if debug else "0",
        "NUMBA_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
    })
    completed = subprocess.run(
        [sys.executable, str(PROBE), "--mode", mode, "--result", str(result)],
        cwd=run, env=environment, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return __import__("json").loads(result.read_text()), completed.stdout + completed.stderr


def cache_artifacts(cache, function, suffix):
    pattern = re.compile(rf"patch_radiation\.{function}-\d+\.py\d+(?:\.\d+)?\.{suffix}$")
    return [path for path in cache.rglob("*") if path.is_file() and pattern.search(path.name)]


@pytest.mark.parametrize("order", [("serial", "parallel"), ("parallel", "serial")])
def test_fresh_process_cache_orders_remain_distinct_and_loadable(tmp_path, order):
    cache = tmp_path / "jit"
    compiled = {}
    for index, mode in enumerate(order):
        compiled[mode], _ = run_probe(cache, tmp_path / f"compile-{index}-{mode}", mode)

    for family in ("shortwave", "longwave"):
        parallel = compiled["parallel"]["metadata"][family + "_parallel"]
        serial = compiled["serial"]["metadata"][family + "_serial"]
        assert parallel["parallel_target"] and parallel["parfors_present"]
        assert not serial["parallel_target"] and not serial["parfors_present"]

    functions = ("_shortwave", "_shortwave_serial", "_longwave", "_longwave_serial")
    for function in functions:
        assert len(cache_artifacts(cache, function, "nbi")) == 1
        assert len(cache_artifacts(cache, function, "nbc")) >= 1

    result, debug = run_probe(cache, tmp_path / "fresh-verify", "verify", debug=True)
    assert result["shortwave_exact"] and result["longwave_exact"]
    assert result["shortwave_max_abs"] == 0.0
    assert result["longwave_max_abs"] == 0.0
    assert result["target_options"] == {
        "shortwave_parallel": True,
        "shortwave_serial": False,
        "longwave_parallel": True,
        "longwave_serial": False,
    }
    for function in functions:
        assert re.search(rf"index loaded from '.*{re.escape(function)}-\d+\.py\d+\.nbi'", debug)
        assert re.search(rf"data loaded from '.*{re.escape(function)}-\d+\.py\d+\.\d+\.nbc'", debug)
