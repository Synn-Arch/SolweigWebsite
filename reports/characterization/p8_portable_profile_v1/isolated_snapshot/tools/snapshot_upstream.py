#!/usr/bin/env python3
"""Create a static, source-provenanced contract snapshot of SOLWEIG-GPU.

This script deliberately does not import the upstream package.  Importing it
would require optional numerical dependencies and would turn a source contract
snapshot into an execution test.  All evidence in the generated reports is
therefore labelled ``static``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
PUBLIC_FUNCTIONS = [
    "thermal_comfort",
    "preprocess",
    "build_inputs",
    "build_wind_ext_coeff",
    "run_walls_aspect",
    "calculate_svf",
    "run_utci_tiles",
]


def _run(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _source(value: ast.AST | None) -> str | None:
    return ast.unparse(value) if value is not None else None


def _literal(value: ast.AST | None) -> Any:
    """Return JSON-safe literals, retaining source for non-literals."""
    if value is None:
        return None
    try:
        result = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError):
        return {"expression": _source(value)}
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    if isinstance(result, (list, tuple)):
        return [_literal(ast.Constant(v)) for v in result]
    if isinstance(result, dict):
        return {str(k): v for k, v in result.items()}
    return {"expression": _source(value)}


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, path: str) -> dict[str, Any]:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    pos_defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parameters = []
    for arg, default in zip(positional, pos_defaults):
        parameters.append(
            {
                "name": arg.arg,
                "kind": "positional_only" if arg in args.posonlyargs else "positional_or_keyword",
                "annotation": _source(arg.annotation),
                "default": _source(default),
                "default_value": _literal(default),
            }
        )
    if args.vararg:
        parameters.append({"name": args.vararg.arg, "kind": "var_positional", "annotation": _source(args.vararg.annotation), "default": None, "default_value": None})
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(
            {
                "name": arg.arg,
                "kind": "keyword_only",
                "annotation": _source(arg.annotation),
                "default": _source(default),
                "default_value": _literal(default),
            }
        )
    if args.kwarg:
        parameters.append({"name": args.kwarg.arg, "kind": "var_keyword", "annotation": _source(args.kwarg.annotation), "default": None, "default_value": None})
    return {
        "evidence": "static",
        "module": "solweig_gpu.solweig_gpu",
        "file": path,
        "line": node.lineno,
        "name": node.name,
        "parameters": parameters,
        "return_annotation": _source(node.returns),
    }


def _find_functions(tree: ast.AST, path: str) -> dict[str, dict[str, Any]]:
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PUBLIC_FUNCTIONS:
            found[node.name] = _signature(node, path)
    return found


def _cli_contract(path: Path, display_path: str = "solweig_gpu/cli.py") -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    flags = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        names = [_literal(arg) for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        keywords = {kw.arg: _source(kw.value) for kw in node.keywords if kw.arg is not None}
        values = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg is not None}
        flags.append({
            "evidence": "static",
            "flags": names,
            "dest": values.get("dest"),
            "default": values.get("default"),
            "default_expression": keywords.get("default"),
            "required": values.get("required"),
            "action": values.get("action"),
            "type": keywords.get("type"),
            "line": node.lineno,
        })
    flags.sort(key=lambda item: item["line"])
    return {"evidence": "static", "file": display_path, "flags": flags}


def _entry_points(path: Path, display_path: str = "setup.py") -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"):
            continue
        for kw in node.keywords:
            if kw.arg != "entry_points" or not isinstance(kw.value, ast.Dict):
                continue
            for key, value in zip(kw.value.keys, kw.value.values):
                if _source(key) != "'console_scripts'" or not isinstance(value, ast.List):
                    continue
                for item in value.elts:
                    text = _literal(item)
                    if isinstance(text, str) and "=" in text:
                        name, target = text.split("=", 1)
                        found[name.strip()] = target.strip()
    return {"evidence": "static", "file": display_path, "console_scripts": found}


def _manifest(root: Path, commit: str, clean: bool) -> dict[str, Any]:
    paths = _run(root, "ls-files", "-z").split("\0")
    files = []
    for relative in paths:
        if not relative:
            continue
        file_path = root / relative
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        files.append({"path": relative, "bytes": file_path.stat().st_size, "sha256": digest})
    return {
        "evidence": "static",
        "repository": "https://github.com/nvnsudharsan/SOLWEIG-GPU",
        "commit": commit,
        "expected_commit": EXPECTED_COMMIT,
        "clean_worktree": clean,
        "tracked_file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/SOLWEIG-GPU"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    root = args.upstream.resolve()
    reports = args.reports.resolve()
    commit = _run(root, "rev-parse", "HEAD")
    status = _run(root, "status", "--porcelain", "--untracked-files=all")
    clean = not status
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"upstream commit mismatch: {commit} != {EXPECTED_COMMIT}")
    if not clean:
        raise SystemExit(f"upstream worktree is not clean:\n{status}")

    api_path = root / "solweig_gpu" / "solweig_gpu.py"
    cli_path = root / "solweig_gpu" / "cli.py"
    setup_path = root / "setup.py"
    functions = _find_functions(ast.parse(api_path.read_text(encoding="utf-8")), "solweig_gpu/solweig_gpu.py")
    missing = [name for name in PUBLIC_FUNCTIONS if name not in functions]
    if missing:
        raise SystemExit(f"missing public functions: {', '.join(missing)}")

    args.reports.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(root, commit, clean)
    contract = {
        "evidence": "static",
        "repository": manifest["repository"],
        "commit": commit,
        "public_functions": functions,
        "cli": _cli_contract(cli_path),
        "entry_points": _entry_points(setup_path),
        "source_files": {
            "api": "solweig_gpu/solweig_gpu.py",
            "cli": "solweig_gpu/cli.py",
            "packaging": "setup.py",
        },
    }
    (reports / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "contract_snapshot.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {reports / 'source_manifest.json'} ({manifest['tracked_file_count']} tracked files)")
    print(f"wrote {reports / 'contract_snapshot.json'} ({len(functions)} public functions, {len(contract['cli']['flags'])} CLI arguments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
