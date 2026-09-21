"""Permanent exact regressions for namespaced radiation JIT caches."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import solweig_light
from solweig_light.radiation import (
    _jit_cache,
    _math_profile,
    _sleef_acos,
    _sleef_classifier,
    wall_shadows,
)


TESTS_ROOT = Path(__file__).resolve().parents[1]
PROBE = TESTS_ROOT / "helpers/exact_persistent_jit_cache_probe.py"
PACKAGE = Path(solweig_light.__file__).resolve().parent
PACKAGE_ROOT = PACKAGE.parent
WALL_SOURCE = Path(wall_shadows.__file__).resolve()
ACOS_SOURCE = Path(_sleef_acos.__file__).resolve()
CLASSIFIER_SOURCE = Path(_sleef_classifier.__file__).resolve()
HELPER_SOURCE = Path(_jit_cache.__file__).resolve()
PROFILE_SOURCE = Path(_math_profile.__file__).resolve()


def function(path, name):
    tree = ast.parse(path.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def normalized_function(node):
    node = copy.deepcopy(node)
    node.name = "normalized"
    node.decorator_list = []
    return ast.dump(ast.fix_missing_locations(node), include_attributes=False)


def decorator_keywords(node):
    call = node.decorator_list[0]
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}


def decorator_dumps(node):
    return tuple(
        ast.dump(decorator, include_attributes=False)
        for decorator in node.decorator_list
    )


def cache_binding(node):
    return (
        len(node.decorator_list) == 2
        and isinstance(node.decorator_list[1], ast.Name)
        and node.decorator_list[1].id == "bind_cache_identity"
    )


def assignment(path, name):
    tree = ast.parse(path.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


def test_cache_modules_share_active_installed_package_origin():
    for module in (
        _jit_cache,
        _math_profile,
        _sleef_acos,
        _sleef_classifier,
        wall_shadows,
    ):
        assert Path(module.__file__).resolve().is_relative_to(PACKAGE)


def test_wall_serial_definitions_are_independent_and_body_identical():
    for parallel_name in ("_wall13", "_wall23"):
        parallel = function(WALL_SOURCE, parallel_name)
        serial = function(WALL_SOURCE, parallel_name + "_serial")
        assert normalized_function(parallel) == normalized_function(serial)
        assert decorator_keywords(parallel) == {"fastmath": False, "parallel": True}
        assert decorator_keywords(serial) == {"cache": True, "fastmath": False}
        assert cache_binding(serial)
    tree = ast.parse(WALL_SOURCE.read_text())
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "py_func"
        for node in ast.walk(tree)
    )


def test_classifier_helpers_match_canonical_bodies_and_decorators():
    for name in ("fma", "dfmul"):
        classifier = function(CLASSIFIER_SOURCE, name)
        canonical = function(ACOS_SOURCE, name)
        assert normalized_function(classifier) == normalized_function(canonical)
        assert decorator_dumps(classifier) == decorator_dumps(canonical)
    assert ast.dump(assignment(CLASSIFIER_SOURCE, "F"), include_attributes=False) == ast.dump(
        assignment(ACOS_SOURCE, "F"), include_attributes=False
    )
    classifier_tree = ast.parse(CLASSIFIER_SOURCE.read_text())
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "_sleef_acos"
        for node in ast.walk(classifier_tree)
    )
    for name in ("tan_array", "atan_array"):
        node = function(CLASSIFIER_SOURCE, name)
        assert decorator_keywords(node) == {
            "cache": True,
            "fastmath": False,
            "error_model": "numpy",
        }
        assert cache_binding(node)
    asvf = function(ACOS_SOURCE, "asvf_fma")
    assert decorator_keywords(asvf) == {
        "cache": True,
        "fastmath": False,
        "error_model": "numpy",
    }
    assert cache_binding(asvf)


def test_cache_identity_helper_is_metadata_only_and_profiled():
    binding = function(HELPER_SOURCE, "bind_cache_identity")
    assigned_attributes = [
        node for node in ast.walk(binding)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
    ]
    assert len(assigned_attributes) == 1
    target = assigned_attributes[0]
    assert isinstance(target.value, ast.Name) and target.value.id == "function"
    assert target.attr == "__qualname__"
    assert isinstance(binding.body[-1], ast.Return)
    assert isinstance(binding.body[-1].value, ast.Name)
    assert binding.body[-1].value.id == "function"
    source_files = ast.literal_eval(assignment(PROFILE_SOURCE, "SOURCE_FILES").value)
    assert "_jit_cache.py" in source_files
    assert _math_profile.PROFILE_ID == "solweig-portable-sleef-5a1d179d-v1"


def run_probe(source, cache, run, mode, *, order="serial-first", debug=False,
              inspect_lowering=False):
    run.mkdir(parents=True)
    result = run / "result.json"
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(source),
        "EXPECTED_SOURCE_ROOT": str(source),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(cache),
        "NUMBA_DEBUG_CACHE": "1" if debug else "0",
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    command = [
        sys.executable,
        str(PROBE),
        "--mode", mode,
        "--order", order,
        "--result", str(result),
    ]
    if inspect_lowering:
        command.append("--inspect-lowering")
    completed = subprocess.run(
        command,
        cwd=run,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(result.read_text()), completed.stdout + completed.stderr


def cache_names(cache):
    return sorted(path.name for path in cache.rglob("*") if path.is_file())


def has_artifact(files, function_name, suffix):
    pattern = re.compile(
        rf"(?:^|\.){re.escape(function_name)}(?:__jitcache_[0-9a-f]{{64}})?"
        rf"-\d+\.py\d+(?:\.\d+)?\.{suffix}$"
    )
    return any(pattern.search(name) for name in files)


def assert_loaded(debug, function_name):
    namespace = rf"{re.escape(function_name)}__jitcache_[0-9a-f]{{64}}"
    assert re.search(rf"index loaded from '.*{namespace}-\d+\.py\d+\.nbi'", debug)
    assert re.search(rf"data loaded from '.*{namespace}-\d+\.py\d+\.\d+\.nbc'", debug)


def assert_saved(debug, qualname):
    escaped = re.escape(qualname)
    assert re.search(rf"index saved to '.*{escaped}-\d+\.py\d+\.nbi'", debug)
    assert re.search(rf"data saved to '.*{escaped}-\d+\.py\d+\.\d+\.nbc'", debug)


@pytest.mark.parametrize("order", ["serial-first", "parallel-first"])
def test_wall_cold_warm_orders_are_exact_and_cache_only_serial(tmp_path, order):
    cache = tmp_path / "cache"
    cold, cold_debug = run_probe(
        PACKAGE_ROOT, cache, tmp_path / "cold", "wall", order=order, debug=True,
    )
    reverse = "parallel-first" if order == "serial-first" else "serial-first"
    warm, warm_debug = run_probe(
        PACKAGE_ROOT, cache, tmp_path / "warm", "wall", order=reverse, debug=True,
    )
    assert cold["origins"] == warm["origins"]
    assert cold["separate_py_funcs"] == {"wall13": True, "wall23": True}
    assert cold["results"] == warm["results"]
    assert cold["results"]["serial"] == cold["results"]["parallel"]
    identity = cold["dispatcher_identity"]
    assert identity == warm["dispatcher_identity"]
    assert identity["wall13_parallel_target"]
    assert identity["wall23_parallel_target"]
    assert not identity["wall13_serial_target"]
    assert not identity["wall23_serial_target"]
    assert identity["wall13_parallel_qualname"] != identity["wall13_serial_qualname"]
    assert identity["wall23_parallel_qualname"] != identity["wall23_serial_qualname"]
    files = cache_names(cache)
    for function_name in ("_wall13_serial", "_wall23_serial"):
        assert has_artifact(files, function_name, "nbi")
        assert has_artifact(files, function_name, "nbc")
        assert "saved to" in cold_debug
        assert_loaded(warm_debug, function_name)
    for function_name in ("_wall13", "_wall23"):
        assert not has_artifact(files, function_name, "nbi")
        assert not has_artifact(files, function_name, "nbc")


def test_math_cold_warm_cache_preserves_fma_fallback_and_profile(tmp_path):
    cache = tmp_path / "cache"
    cold, cold_debug = run_probe(
        PACKAGE_ROOT,
        cache,
        tmp_path / "cold",
        "math",
        debug=True,
        inspect_lowering=True,
    )
    warm, warm_debug = run_probe(
        PACKAGE_ROOT, cache, tmp_path / "warm", "math", debug=True,
    )
    assert cold["outputs"] == warm["outputs"]
    assert cold["origins"] == warm["origins"]
    assert cold["profile_id"] == warm["profile_id"] == "solweig-portable-sleef-5a1d179d-v1"
    assert cold["profile_fingerprint"] == warm["profile_fingerprint"]
    assert cold["source_sha256"] == warm["source_sha256"]
    assert set(cold["source_files"]) == set(_math_profile.SOURCE_FILES)
    assert "_jit_cache.py" in cold["source_files"]
    assert all(cold["fallback_exact"].values())
    assert cold["fallback_exact"] == warm["fallback_exact"]
    assert warm["lowering"] is None
    assert all(
        item == {"explicit_fma": True, "fast_flag_absent": True}
        for item in cold["lowering"].values()
    )
    files = cache_names(cache)
    for function_name in ("asvf_fma", "tan_array", "atan_array"):
        assert has_artifact(files, function_name, "nbi")
        assert has_artifact(files, function_name, "nbc")
        assert "saved to" in cold_debug
        assert_loaded(warm_debug, function_name)


def copy_active_source(destination):
    destination.mkdir()
    shutil.copytree(
        PACKAGE,
        destination / "solweig_light",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbi", "*.nbc"),
    )
    # The admitted snapshot or installed distribution may be read-only. Only
    # this owned test copy is made writable for deliberate mutation/cleanup.
    for path in (destination / "solweig_light").rglob("*"):
        path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    package = destination / "solweig_light"
    package.chmod(package.stat().st_mode | 0o700)
    return destination


def mutate_same_size_and_mtime(path, old, new):
    assert len(old) == len(new)
    before = path.stat()
    content = path.read_bytes()
    assert content.count(old) == 1
    changed = content.replace(old, new)
    assert len(changed) == len(content)
    path.write_bytes(changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("target", ["defining-module", "namespace-helper"])
def test_wall_namespace_rejects_same_stamp_stale_cache(tmp_path, target):
    owned = copy_active_source(tmp_path / "owned-source")
    cache = tmp_path / "shared-cache"
    original, _ = run_probe(
        owned, cache, tmp_path / "original-cold", "wall", debug=True,
    )
    old_files = set(cache_names(cache))
    if target == "defining-module":
        mutate_same_size_and_mtime(
            owned / "solweig_light/radiation/wall_shadows.py",
            b"Ray schedules retain",
            b"Ray schedules remain",
        )
    else:
        mutate_same_size_and_mtime(
            owned / "solweig_light/radiation/_jit_cache.py",
            b"cache-namespace-v1",
            b"cache-namespace-v2",
        )
    mutated_warm, debug = run_probe(
        owned,
        cache,
        tmp_path / "mutated-warm",
        "wall",
        order="parallel-first",
        debug=True,
    )
    mutated_cold, _ = run_probe(
        owned,
        tmp_path / "independent-cache",
        tmp_path / "mutated-independent-cold",
        "wall",
        debug=True,
    )
    assert mutated_warm["results"] == mutated_cold["results"] == original["results"]
    assert mutated_warm["cache_qualnames"] == mutated_cold["cache_qualnames"]
    assert mutated_warm["cache_qualnames"] != original["cache_qualnames"]
    new_files = set(cache_names(cache)) - old_files
    for qualname in mutated_warm["cache_qualnames"].values():
        assert any(qualname in filename for filename in new_files)
        assert_saved(debug, qualname)
        assert not re.search(rf"loaded from '.*{re.escape(qualname)}-", debug)


@pytest.mark.parametrize(
    "relative,old,new,affected",
    [
        (
            "solweig_light/radiation/_sleef_acos.py",
            b"operation-order port",
            b"operation-order copy",
            ("asvf_fma",),
        ),
        (
            "solweig_light/radiation/_sleef_classifier.py",
            b"Bounded SLEEF",
            b"Limited SLEEF",
            ("tan_array", "atan_array"),
        ),
        (
            "solweig_light/radiation/_jit_cache.py",
            b"cache-namespace-v1",
            b"cache-namespace-v2",
            ("asvf_fma", "tan_array", "atan_array"),
        ),
    ],
    ids=("acos-module", "classifier-module", "shared-helper"),
)
def test_math_namespace_rejects_same_stamp_stale_cache(
        tmp_path, relative, old, new, affected):
    owned = copy_active_source(tmp_path / "owned-source")
    cache = tmp_path / "shared-cache"
    original, _ = run_probe(
        owned, cache, tmp_path / "original-cold", "math", debug=True,
    )
    old_files = set(cache_names(cache))
    mutate_same_size_and_mtime(owned / relative, old, new)
    mutated_warm, debug = run_probe(
        owned, cache, tmp_path / "mutated-warm", "math", debug=True,
    )
    mutated_cold, _ = run_probe(
        owned,
        tmp_path / "independent-cache",
        tmp_path / "mutated-independent-cold",
        "math",
        debug=True,
    )
    assert mutated_warm["outputs"] == mutated_cold["outputs"] == original["outputs"]
    assert all(mutated_warm["fallback_exact"].values())
    assert mutated_warm["cache_qualnames"] == mutated_cold["cache_qualnames"]
    assert mutated_warm["profile_fingerprint"] == mutated_cold["profile_fingerprint"]
    assert mutated_warm["profile_fingerprint"] != original["profile_fingerprint"]
    assert mutated_warm["lowering"] is None
    assert mutated_cold["lowering"] is None
    all_entrypoints = {"asvf_fma", "tan_array", "atan_array"}
    affected = set(affected)
    changed = {
        name for name in all_entrypoints
        if mutated_warm["cache_qualnames"][name] != original["cache_qualnames"][name]
    }
    assert changed == affected
    new_files = set(cache_names(cache)) - old_files
    for name in affected:
        qualname = mutated_warm["cache_qualnames"][name]
        assert any(qualname in filename for filename in new_files)
        assert_saved(debug, qualname)
        assert not re.search(rf"loaded from '.*{re.escape(qualname)}-", debug)
    for name in all_entrypoints - affected:
        assert_loaded(debug, name)
