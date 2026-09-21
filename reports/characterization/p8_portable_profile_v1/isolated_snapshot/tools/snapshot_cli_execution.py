#!/usr/bin/env python3
"""Record the pinned upstream CLI executable and parser contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
EXPECTED_FLAGS = {
    "--base_path", "--date", "--building_dsm", "--dem", "--trees", "--landcover", "--tile_size",
    "--overlap", "--use_own_met", "--own_metfile", "--data_source_type",
    "--data_folder", "--start", "--end", "--era5_z0_find", "--use_uhi",
    "--save_tmrt", "--save_svf", "--save_kup", "--save_kdown", "--save_lup",
    "--save_ldown", "--save_shadow", "--save_wbgt", "--save_ta", "--save_wind",
    "--version",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def run(argv: list[str], *, executable: Path, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run([str(executable), *argv], cwd=cwd, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"argv": argv, "exit_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}

def spy_case(*, python: Path, root: Path, base: Path, met: Path,
             data_folder: Path | None = None) -> dict[str, Any]:
    """Capture forwarding after parser validation; never launch simulation."""
    args = [
        "--base_path", str(base), "--date", "2020-08-13",
        "--building_dsm", "b.tif", "--dem", "d.tif", "--trees", "t.tif",
        "--landcover", "lc.tif", "--tile_size", "101", "--overlap", "7",
        "--use_own_met", "YeS", "--own_metfile", str(met),
        "--data_source_type", "ERA5", "--start", "2020-08-13 00:00:00",
        "--end", "2020-08-13 23:00:00", "--use_uhi", "fAlSe",
        "--save_tmrt", "no", "--save_svf", "TRUE", "--save_kup", "t",
        "--save_kdown", "1", "--save_lup", "false", "--save_ldown", "0",
        "--save_shadow", "yes", "--save_wbgt", "f", "--save_ta", "TrUe",
        "--save_wind", "nO",
    ]
    if data_folder is not None:
        args += ["--data_folder", str(data_folder)]
    code = '''
import json
from solweig_gpu import cli
def spy(**kwargs):
    print(json.dumps({"spy": "parser_forwarding_only", "kwargs": kwargs}, sort_keys=True))
cli.thermal_comfort = spy
cli.main()
'''
    completed = subprocess.run([str(python), "-c", code, *args], cwd=root, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"argv": args, "exit_code": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr,
            "method": "isolated parser with labelled thermal_comfort spy; simulation not launched"}

def boolean_cases(*, python: Path, root: Path, base: Path, met: Path, data: Path) -> dict[str, Any]:
    results = {}
    for token in ("yes", "true", "t", "1", "no", "false", "f", "0",
                  "YeS", "TrUe", "T", "NO", "FaLsE", "F"):
        code = '''
import json
from solweig_gpu import cli
def spy(**kwargs):
    print(json.dumps({"spy": "parser_forwarding_only", "use_own_met": kwargs["use_own_met"]}, sort_keys=True))
cli.thermal_comfort = spy
cli.main()
'''
        argv = ["--base_path", str(base), "--date", "2020-08-13", "--use_own_met", token]
        if token.lower() in {"yes", "true", "t", "1"}:
            argv += ["--own_metfile", str(met)]
        else:
            argv += ["--data_source_type", "ERA5", "--data_folder", str(data),
                     "--start", "2020-08-13 00:00:00", "--end", "2020-08-13 23:00:00"]
        completed = subprocess.run([str(python), "-c", code, *argv], cwd=root, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        results[token] = {"argv": argv, "exit_code": completed.returncode,
                          "stdout": completed.stdout, "stderr": completed.stderr}
    return results

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/SOLWEIG-GPU"))
    parser.add_argument("--python", type=Path, default=Path(".venv-oracle/bin/python"))
    parser.add_argument("--executable", type=Path, default=Path(".venv-oracle/bin/thermal_comfort"))
    parser.add_argument("--output", type=Path, default=Path("reports/cli_execution.json"))
    args = parser.parse_args()
    root, python, executable, output = (p.resolve() for p in (args.upstream, args.python, args.executable, args.output))
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"upstream commit mismatch: {commit} != {EXPECTED_COMMIT}")
    if status:
        raise SystemExit(f"upstream worktree is not clean:\n{status}")
    with tempfile.TemporaryDirectory(prefix="solweig-cli-") as temp:
        temp_root = Path(temp); base = temp_root / "base"; base.mkdir()
        met = temp_root / "met.txt"; met.write_text("spy fixture\n", encoding="utf-8")
        data = temp_root / "data"; data.mkdir()
        cases = {
            "help": run(["--help"], executable=executable, cwd=root),
            "version": run(["--version"], executable=executable, cwd=root),
            "no_args": run([], executable=executable, cwd=root),
            "invalid_bool": run(["--use_own_met", "maybe"], executable=executable, cwd=root),
            "unknown_flag": run(["--definitely_unknown"], executable=executable, cwd=root),
            "forwarding_no_data_folder": spy_case(python=python, root=root, base=base, met=met),
            "forwarding_with_data_folder": spy_case(python=python, root=root, base=base, met=met, data_folder=data),
            "boolean_values": boolean_cases(python=python, root=root, base=base, met=met, data=data),
        }
    env_probe = subprocess.run([str(python), "-c", "import sys, solweig_gpu; print(sys.version); print(solweig_gpu.__version__)"],
                               cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    source = root / "solweig_gpu" / "cli.py"
    probe_lines = env_probe.stdout.splitlines()
    report = {
        "evidence_class": "original_upstream_cli_execution",
        "simulation_launched": False,
        "repository": "https://github.com/nvnsudharsan/SOLWEIG-GPU",
        "source_commit": commit, "source_clean_worktree": not bool(status),
        "source_files": {"cli": {"path": "solweig_gpu/cli.py", "sha256": sha256(source)}},
        "environment": {"python_executable": str(python),
                         "python_version": probe_lines[0] if probe_lines else None,
                         "package_version": probe_lines[1] if len(probe_lines) > 1 else None,
                         "platform": platform.platform(), "machine": platform.machine(), "cwd": str(root)},
        "executable": {"path": str(executable), "sha256": sha256(executable)},
        "cli_flag_inventory": {"count": len(EXPECTED_FLAGS), "flags": sorted(EXPECTED_FLAGS)},
        "cases": cases,
        "era5_z0_find_rule": {"source_default": "None (CLI sentinel)",
            "resolved_without_data_folder": "False", "resolved_with_data_folder": "True",
            "note": "Python thermal_comfort default is ERA_5_z0_find=True; CLI derives unspecified value from data_folder."},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(cases)} cases, {len(EXPECTED_FLAGS)} flags)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
