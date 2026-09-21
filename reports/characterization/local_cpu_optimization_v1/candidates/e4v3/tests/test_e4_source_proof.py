"""Non-executed source proofs for the isolated E4 candidates."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path


PACKET = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get(
    "E4_BASELINE_SRC",
    "/Users/alansynn/Workspace/solweig-light/reports/characterization/"
    "local_cpu_optimization_v1/candidates/baseline/src",
)).resolve()
WALL = PACKET / "wall_candidate/src"
MATH = PACKET / "math_candidate/src"
MANIFEST = json.loads((PACKET / "baseline_source_manifest.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function(path, name):
    tree = ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def undecorated(node):
    node = copy.deepcopy(node)
    node.name = "normalized"
    node.decorator_list = []
    return ast.dump(ast.fix_missing_locations(node), include_attributes=False)


def decorator_keywords(node):
    call = node.decorator_list[0]
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}


def cache_binding(node):
    assert len(node.decorator_list) == 2
    binding = node.decorator_list[1]
    return isinstance(binding, ast.Name) and binding.id == "bind_cache_identity"


def decorator_dumps(node):
    return tuple(
        ast.dump(decorator, include_attributes=False)
        for decorator in node.decorator_list
    )


def assignment(path, name):
    tree = ast.parse(path.read_text())
    return next(
        n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )


def test_baseline_binding_and_candidate_isolation():
    assert MANIFEST["baseline_commit"] == "8ca23d444a3b05bdeb76655329c0a09c3dc4d6e8"
    for relative, expected in MANIFEST["files"].items():
        assert sha(BASELINE / relative) == expected
    assert sha(WALL / "solweig_light/radiation/_sleef_acos.py") == MANIFEST["files"]["solweig_light/radiation/_sleef_acos.py"]
    assert sha(MATH / "solweig_light/radiation/wall_shadows.py") == MANIFEST["files"]["solweig_light/radiation/wall_shadows.py"]


def test_wall_serial_functions_are_real_and_body_identical():
    path = WALL / "solweig_light/radiation/wall_shadows.py"
    for parallel_name in ("_wall13", "_wall23"):
        serial_name = parallel_name + "_serial"
        parallel = function(path, parallel_name)
        serial = function(path, serial_name)
        assert undecorated(parallel) == undecorated(serial)
        assert decorator_keywords(parallel) == {"fastmath": False, "parallel": True}
        assert decorator_keywords(serial) == {"cache": True, "fastmath": False}
        assert cache_binding(serial)
    tree = ast.parse(path.read_text())
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "py_func"
        for node in ast.walk(tree)
    )


def test_asvf_cache_only_changes_decorator_and_profile_id_is_stable():
    relative = Path("solweig_light/radiation/_sleef_acos.py")
    before = function(BASELINE / relative, "asvf_fma")
    after = function(MATH / relative, "asvf_fma")
    assert undecorated(before) == undecorated(after)
    assert decorator_keywords(before) == {"fastmath": False, "error_model": "numpy"}
    assert decorator_keywords(after) == {"cache": True, "fastmath": False, "error_model": "numpy"}
    assert cache_binding(after)

    profile = Path("solweig_light/radiation/_math_profile.py")
    assert sha(MATH / relative) != sha(BASELINE / relative)
    profile_tree = ast.parse((MATH / profile).read_text())
    profile_id = next(
        ast.literal_eval(n.value) for n in profile_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PROFILE_ID" for t in n.targets)
    )
    assert profile_id == "solweig-portable-sleef-5a1d179d-v1"
    source_files = ast.literal_eval(assignment(MATH / profile, "SOURCE_FILES").value)
    assert "_jit_cache.py" in source_files
    expected_sources = set(ast.literal_eval(assignment(BASELINE / profile, "SOURCE_FILES").value)) | {"_jit_cache.py"}
    assert set(source_files) == expected_sources
    assert set(ast.literal_eval(assignment(WALL / profile, "SOURCE_FILES").value)) == expected_sources
    for name in ("_package_version", "_source_hashes", "profile_identity", "asvf", "tan32", "atan32"):
        assert undecorated(function(MATH / profile, name)) == undecorated(function(BASELINE / profile, name))


def test_classifier_helpers_are_source_local_and_canonical_ast_copies():
    classifier = MATH / "solweig_light/radiation/_sleef_classifier.py"
    canonical = MATH / "solweig_light/radiation/_sleef_acos.py"
    assert undecorated(function(classifier, "fma")) == undecorated(function(canonical, "fma"))
    assert undecorated(function(classifier, "dfmul")) == undecorated(function(canonical, "dfmul"))
    assert decorator_dumps(function(classifier, "fma")) == decorator_dumps(
        function(canonical, "fma")
    )
    assert decorator_dumps(function(classifier, "dfmul")) == decorator_dumps(
        function(canonical, "dfmul")
    )
    assert ast.dump(assignment(classifier, "F"), include_attributes=False) == ast.dump(
        assignment(canonical, "F"), include_attributes=False
    )
    classifier_tree = ast.parse(classifier.read_text())
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "_sleef_acos"
        for node in ast.walk(classifier_tree)
    )

    baseline_classifier = BASELINE / "solweig_light/radiation/_sleef_classifier.py"
    unchanged_scalars = (
        "add_ff", "add_ff2", "add_f2f", "add_f2f2", "normalize", "square",
        "reciprocal", "divide", "tan_fma", "atan_fma",
    )
    for name in unchanged_scalars:
        assert undecorated(function(classifier, name)) == undecorated(function(baseline_classifier, name))


def test_classifier_entrypoint_bodies_unchanged_and_cached():
    path = MATH / "solweig_light/radiation/_sleef_classifier.py"
    baseline = BASELINE / "solweig_light/radiation/_sleef_classifier.py"
    for name in ("tan_array", "atan_array"):
        assert undecorated(function(path, name)) == undecorated(function(baseline, name))
        assert decorator_keywords(function(path, name)) == {
            "cache": True, "fastmath": False, "error_model": "numpy"
        }
        assert cache_binding(function(path, name))


def test_cache_identity_helper_is_metadata_only_and_shared():
    wall_helper = WALL / "solweig_light/radiation/_jit_cache.py"
    math_helper = MATH / "solweig_light/radiation/_jit_cache.py"
    assert wall_helper.read_bytes() == math_helper.read_bytes()
    binding = function(wall_helper, "bind_cache_identity")
    assigned_attributes = [
        node for node in ast.walk(binding)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
    ]
    assert len(assigned_attributes) == 1
    assert isinstance(assigned_attributes[0].value, ast.Name)
    assert assigned_attributes[0].value.id == "function"
    assert assigned_attributes[0].attr == "__qualname__"
    assert isinstance(binding.body[-1], ast.Return)
    assert isinstance(binding.body[-1].value, ast.Name)
    assert binding.body[-1].value.id == "function"
    for candidate in (WALL, MATH):
        for relative, names in (
            ("solweig_light/radiation/wall_shadows.py", ("_wall13_serial", "_wall23_serial")),
            ("solweig_light/radiation/_sleef_acos.py", ("asvf_fma",)),
            ("solweig_light/radiation/_sleef_classifier.py", ("tan_array", "atan_array")),
        ):
            path = candidate / relative
            if path.exists() and "bind_cache_identity" in path.read_text():
                for name in names:
                    node = function(path, name)
                    if len(node.decorator_list) == 2:
                        assert cache_binding(node)
