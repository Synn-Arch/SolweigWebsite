#!/usr/bin/env python3
"""Static AST inventory and call-site graph for the pinned upstream package.

This intentionally does not import the package or execute code.  Names reached
through runtime imports, decorators, getattr, eval, or arbitrary attributes are
reported as unresolved rather than guessed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def span(node: ast.AST) -> dict[str, int]:
    return {
        "line_start": int(getattr(node, "lineno", 0)),
        "line_end": int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
        "col_start": int(getattr(node, "col_offset", 0)),
        "col_end": int(getattr(node, "end_col_offset", 0)),
    }


def import_inventory(tree: ast.AST) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append({"kind": "import", "module": alias.name,
                            "name": None, "asname": alias.asname, **span(node)})
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                out.append({"kind": "from_import", "module": module,
                            "name": alias.name, "asname": alias.asname, **span(node)})
    return sorted(out, key=lambda x: (x["line_start"], x["col_start"], x["kind"]))


class FileScanner(ast.NodeVisitor):
    def __init__(self, module: str, path: str, imported_names: set[str]):
        self.module = module
        self.path = path
        self.imported_names = imported_names
        self.definitions: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self._scope: list[str] = ["<module>"]
        self._scope_local_names: list[set[str]] = [set()]

    @property
    def scope(self) -> str:
        return ".".join([self.module, *(name for name in self._scope if name != "<module>")])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_definition(node, "class")
        self._scope_local_names[-1].add(node.name)
        for child in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(child)
        self._scope.append(node.name)
        self._scope_local_names.append(set())
        for child in node.body:
            self.visit(child)
        self._scope_local_names.pop(); self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_definition(node, "function", async_=False)
        self._scope_local_names[-1].add(node.name)
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_definition(node, "function", async_=True)
        self._scope_local_names[-1].add(node.name)
        self._visit_function(node)

    def _record_definition(self, node: ast.AST, kind: str, async_: bool | None = None) -> None:
        name = getattr(node, "name", "")
        qualname = self.scope + "." + name
        row = {"qualified_name": qualname, "name": name, "kind": kind,
               "scope": self.scope, **span(node)}
        if async_ is not None:
            row["async"] = async_
        self.definitions.append(row)

    def _visit_function(self, node: ast.AST) -> None:
        name = getattr(node, "name", "")
        local = set()
        args = getattr(node, "args", None)
        if args:
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]: local.add(arg.arg)
            if args.vararg: local.add(args.vararg.arg)
            if args.kwarg: local.add(args.kwarg.arg)
        # Defaults/decorators/annotations are syntactically in the enclosing scope.
        # Include arguments: default expressions can themselves contain calls.
        for child in [*node.decorator_list, args, node.returns]:
            if child is not None:
                self.visit(child)
        self._scope.append(name)
        self._scope_local_names.append(local)
        for child in node.body:
            self.visit(child)
        self._scope_local_names.pop(); self._scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        expression = dotted(node.func)
        if expression is None:
            classification = "dynamic"
        elif isinstance(node.func, ast.Name):
            local = any(expression in names for names in reversed(self._scope_local_names))
            classification = "local_name" if local else ("imported_name" if expression in self.imported_names else "unresolved_name")
        else:
            classification = "attribute_unresolved"
        self.calls.append({"caller": self.scope, "callee": expression or ast.dump(node.func),
                           "classification": classification, **span(node)})
        self.generic_visit(node)


def independent_counts(paths: list[Path]) -> dict[str, int]:
    files = functions = classes = calls = 0
    for path in paths:
        files += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): functions += 1
            elif isinstance(node, ast.ClassDef): classes += 1
            elif isinstance(node, ast.Call): calls += 1
    return {"files": files, "functions": functions, "classes": classes, "calls": calls}


def build(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("*.py"))
    modules: list[dict[str, Any]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = path.stem
        imports = import_inventory(tree)
        imported_names = {item["asname"] or (item["name"] if item["kind"] == "from_import" else item["module"].split(".")[0]) for item in imports}
        scanner = FileScanner(module, str(path), imported_names)
        scanner.visit(tree)
        defined = {d["name"] for d in scanner.definitions}
        imported = set()
        for item in imports:
            imported.add(item["asname"] or (item["name"] if item["kind"] == "from_import" else item["module"].split(".")[0]))
        referenced_local = {c["callee"] for c in scanner.calls if c["classification"] == "local_name"}
        unreferenced = sorted(defined - referenced_local)
        modules.append({
            "path": str(path), "module": module, "sha256": sha256(path),
            "line_count": len(path.read_text(encoding="utf-8").splitlines()),
            "imports": import_inventory(tree), "definitions": sorted(scanner.definitions, key=lambda x: (x["line_start"], x["col_start"])),
            "call_sites": sorted(scanner.calls, key=lambda x: (x["line_start"], x["col_start"])),
            "unreferenced_definition_names_static_only": unreferenced,
            "notes": ["Unreferenced means no statically resolved direct call in this file; it is not dead-code proof."],
        })
    reported = {"files": len(paths), "functions": sum(sum(d["kind"] == "function" for d in m["definitions"]) for m in modules), "classes": sum(sum(d["kind"] == "class" for d in m["definitions"]) for m in modules)}
    reported["calls"] = sum(len(m["call_sites"]) for m in modules)
    independent = independent_counts(paths)
    return {
        "schema": "solweig-upstream-static-source-call-graph/v1",
        "scope": "clean pinned upstream solweig_gpu package; static AST only; no imports or execution",
        "source_root": str(root), "files": modules,
        "counts": {"reported": reported, "independent_ast": independent, "match": reported == independent},
        "classification": {
            "local_name": "Name call bound by a visited definition or function argument; runtime callability is not proved",
            "imported_name": "Name call bound by a syntactic import; imported implementation is outside this package graph",
            "unresolved_name": "Name call with no statically visible definition; may be imported, builtin, or dynamic",
            "attribute_unresolved": "Attribute call; receiver/member resolution is intentionally not guessed",
            "dynamic": "Computed call target (for example lambda/subscript/call result)",
        },
        "limitations": ["Attribute, imported, decorator, descriptor, and runtime-generated references are not resolved.", "Call-site graph is static evidence only; it is not a runtime execution graph."],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=Path(".upstream/SOLWEIG-GPU/solweig_gpu"))
    ap.add_argument("--output", type=Path, default=Path("reports/source_call_graph.json"))
    args = ap.parse_args()
    repository = args.source_root.resolve().parent
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if commit != "0d7fe742abeeddd890dd58fc76ed7f78bd47faec" or dirty:
        raise SystemExit("Source graph requires the clean pinned upstream checkout")
    result = build(args.source_root)
    result["source_commit"] = commit
    if not result["counts"]["match"]:
        raise SystemExit(f"AST count mismatch: {result['counts']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
