"""Stream the isolated portable-profile classifier against frozen original masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parents[1]
ORACLE = HERE / "oracle"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_oracles() -> tuple[dict, dict]:
    inputs_manifest = json.loads((ORACLE / "classifier_inputs_manifest.json").read_text())
    oracle_manifest = json.loads((ORACLE / "classifier_oracle_manifest.json").read_text())
    corpus = ORACLE / inputs_manifest["corpus"]
    assert sha256(corpus) == inputs_manifest["corpus_sha256"]
    assert oracle_manifest["input_corpus_sha256"] == inputs_manifest["corpus_sha256"]
    assert len(oracle_manifest["timesteps"]) == 24
    for record in oracle_manifest["timesteps"]:
        assert sha256(ORACLE / record["path"]) == record["sha256"]
    return inputs_manifest, oracle_manifest


def mismatch_record(actual: np.ndarray, packed: np.ndarray, count: int) -> dict:
    expected = np.unpackbits(packed, bitorder="little", count=count).astype(bool, copy=False)
    where = np.flatnonzero(actual != expected)
    return {
        "count": int(where.size),
        "first_input_index": int(where[0]) if where.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", choices=("prepared", "serial", "both"), default="both")
    args = parser.parse_args()

    inputs_manifest, oracle_manifest = verify_oracles()
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported before admission")
    from solweig_light.radiation._math_profile import asvf, profile_identity
    from solweig_light.radiation import engine, patch_radiation

    if "torch" in sys.modules:
        raise RuntimeError("candidate imported torch")
    package = Path(sys.modules["solweig_light"].__file__).resolve()
    expected_root = (HERE / "isolated_snapshot" / "src").resolve()
    if expected_root not in package.parents:
        raise RuntimeError(f"candidate origin {package} is outside {expected_root}")

    corpus = np.load(ORACLE / inputs_manifest["corpus"])
    values = corpus["input_bits"].view(np.float32)
    field = asvf(values).reshape(-1, 1)
    geometry = patch_radiation.patch_geometry(
        np.column_stack((corpus["patch_altitude"], corpus["patch_azimuth"]))
    )
    count = len(values)
    result = {
        "schema": "solweig-light.portable-profile-v1-stage-c-mask-result.v1",
        "candidate_origin": str(package),
        "profile": profile_identity(),
        "torch_imported": "torch" in sys.modules,
        "input_manifest_sha256": sha256(ORACLE / "classifier_inputs_manifest.json"),
        "oracle_manifest_sha256": sha256(ORACLE / "classifier_oracle_manifest.json"),
        "input_count": count,
        "timesteps": [],
    }
    totals = {"prepared_sun": 0, "prepared_shade": 0, "serial_sun": 0, "serial_shade": 0}

    for timestep in oracle_manifest["timesteps"]:
        reference = np.load(ORACLE / timestep["path"])
        altitude = np.asarray(timestep["altitude"])
        azimuth = np.asarray(timestep["azimuth"])
        row = {"timestep": timestep["timestep"], "paths": {}}
        prepared = patch_radiation._class_coefficients(altitude, azimuth, geometry, field)
        if prepared is None:
            raise RuntimeError("prepared classifier rejected frozen accepted inputs")
        coefficient64 = np.empty(geometry.altitude.size, dtype=np.float64)
        for patch in range(geometry.altitude.size):
            difference = np.abs(engine._operate(np.subtract, azimuth, geometry.azimuth[patch]))
            deg2rad = engine._divide(np.pi, 180.0)
            xi = np.cos(engine._operate(np.multiply, difference, deg2rad))
            yi = engine._operate(
                np.multiply,
                engine._operate(np.multiply, 2, xi),
                np.tan(engine._operate(np.multiply, altitude, deg2rad)),
            )
            coefficient64[patch] = np.where(yi > 0, 0.0, yi)
        row["coefficient_bit_mismatches"] = int(
            np.count_nonzero(coefficient64.view(np.uint64) != reference["coefficient_bits"])
        )
        row["coefficient_float32_bit_mismatches"] = int(
            np.count_nonzero(prepared[1].view(np.uint32) != reference["coefficient_float32_bits"])
        )

        if args.paths in ("prepared", "both"):
            sun, shade = patch_radiation._classes(
                altitude, azimuth, geometry, field, 0, count, prepared=prepared
            )
            ps = np.packbits(sun.T, axis=1, bitorder="little")
            ph = np.packbits(shade.T, axis=1, bitorder="little")
            # Compare unpacked columns so padding bits are excluded.
            p_sun = {"count": 0, "first_patch": None, "first_input_index": None}
            p_shade = {"count": 0, "first_patch": None, "first_input_index": None}
            for patch in range(geometry.altitude.size):
                sm = mismatch_record(sun[:, patch], reference["sun_pack"][patch], count)
                hm = mismatch_record(shade[:, patch], reference["shade_pack"][patch], count)
                p_sun["count"] += sm["count"]
                p_shade["count"] += hm["count"]
                if sm["count"] and p_sun["first_patch"] is None:
                    p_sun.update(first_patch=patch, first_input_index=sm["first_input_index"])
                if hm["count"] and p_shade["first_patch"] is None:
                    p_shade.update(first_patch=patch, first_input_index=hm["first_input_index"])
            row["paths"]["prepared"] = {"sun": p_sun, "shade": p_shade}
            totals["prepared_sun"] += p_sun["count"]
            totals["prepared_shade"] += p_shade["count"]
            del sun, shade, ps, ph

        if args.paths in ("serial", "both"):
            s_total = h_total = 0
            s_first = h_first = None
            for patch in range(geometry.altitude.size):
                sun, shade = engine.shaded_or_sunlit(
                    altitude, azimuth, geometry.altitude[patch], geometry.azimuth[patch], field
                )
                sm = mismatch_record(sun[:, 0], reference["sun_pack"][patch], count)
                hm = mismatch_record(shade[:, 0], reference["shade_pack"][patch], count)
                s_total += sm["count"]
                h_total += hm["count"]
                if sm["count"] and s_first is None:
                    s_first = {"patch": patch, "input_index": sm["first_input_index"]}
                if hm["count"] and h_first is None:
                    h_first = {"patch": patch, "input_index": hm["first_input_index"]}
            row["paths"]["serial"] = {
                "sun": {"count": s_total, "first": s_first},
                "shade": {"count": h_total, "first": h_first},
            }
            totals["serial_sun"] += s_total
            totals["serial_shade"] += h_total
        result["timesteps"].append(row)
        print(json.dumps({"timestep": row["timestep"], "paths": row["paths"]}), flush=True)

    result["totals"] = totals
    result["passed"] = all(value == 0 for value in totals.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "totals": totals}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
