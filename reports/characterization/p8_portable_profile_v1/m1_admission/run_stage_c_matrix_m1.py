"""Execute the frozen classifier layout/block/thread matrix without full pipeline work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numba import get_num_threads, set_num_threads


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    oracle = root / "oracle"

    input_manifest = json.loads((oracle / "classifier_inputs_manifest.json").read_text())
    oracle_manifest = json.loads((oracle / "classifier_oracle_manifest.json").read_text())
    assert sha256(oracle / input_manifest["corpus"]) == input_manifest["corpus_sha256"]
    assert oracle_manifest["input_corpus_sha256"] == input_manifest["corpus_sha256"]
    for record in oracle_manifest["timesteps"]:
        assert sha256(oracle / record["path"]) == record["sha256"]

    from solweig_light.radiation._math_profile import asvf, profile_identity
    from solweig_light.radiation import engine, patch_radiation
    if "torch" in sys.modules:
        raise RuntimeError("candidate imported torch")

    z = np.load(oracle / input_manifest["corpus"])
    raw = z["input_bits"].view(np.float32)
    full_field = asvf(raw)
    geometry = patch_radiation.patch_geometry(
        np.column_stack((z["patch_altitude"], z["patch_azimuth"]))
    )
    threads = (1, 4, 10)
    lengths = (1, 7, 153, 1024, 4096)
    offsets = (0, 1, 3, 7, 15)
    blocks = (17, 128, 1023)
    cases = []
    totals = {"prepared_sun": 0, "prepared_shade": 0, "serial_sun": 0, "serial_shade": 0}

    for timestep in oracle_manifest["timesteps"]:
        reference = np.load(oracle / timestep["path"])
        ref_sun = np.unpackbits(reference["sun_pack"], axis=1, bitorder="little")[:, :len(raw)].T.astype(bool, copy=False)
        ref_shade = np.unpackbits(reference["shade_pack"], axis=1, bitorder="little")[:, :len(raw)].T.astype(bool, copy=False)
        altitude = np.asarray(timestep["altitude"])
        azimuth = np.asarray(timestep["azimuth"])
        for thread_count in threads:
            set_num_threads(thread_count)
            assert get_num_threads() == thread_count
            for layout in ("contiguous", "stride2"):
                for length in lengths:
                    for offset in offsets:
                        source = full_field[offset:offset + length]
                        if layout == "contiguous":
                            field = np.ascontiguousarray(source)
                        else:
                            backing = np.empty(length * 2, dtype=np.float32)
                            backing[::2] = source
                            backing[1::2] = np.float32(np.nan)
                            field = backing[::2]
                            # NumPy labels length-one views contiguous regardless of
                            # stride; the physical stride still exercises the layout.
                            assert field.strides == (8,)
                        field2d = field.reshape(-1, 1)
                        expected_sun = ref_sun[offset:offset + length]
                        expected_shade = ref_shade[offset:offset + length]
                        prepared = patch_radiation._class_coefficients(
                            altitude, azimuth, geometry, field2d
                        )
                        if prepared is None:
                            raise RuntimeError("prepared path rejected accepted matrix case")
                        for block in blocks:
                            actual_sun = []
                            actual_shade = []
                            for start in range(0, length, block):
                                stop = min(length, start + block)
                                sun, shade = patch_radiation._classes(
                                    altitude, azimuth, geometry, field2d,
                                    start, stop, prepared=prepared,
                                )
                                actual_sun.append(sun)
                                actual_shade.append(shade)
                            sun = np.concatenate(actual_sun, axis=0)
                            shade = np.concatenate(actual_shade, axis=0)
                            sm = int(np.count_nonzero(sun != expected_sun))
                            hm = int(np.count_nonzero(shade != expected_shade))
                            totals["prepared_sun"] += sm
                            totals["prepared_shade"] += hm
                            cases.append({
                                "path": "prepared", "timestep": timestep["timestep"],
                                "threads": thread_count, "layout": layout,
                                "length": length, "offset": offset, "block": block,
                                "sun_mismatches": sm, "shade_mismatches": hm,
                            })

                        # Serial has no block-size policy; execute once per remaining variant.
                        serial_sun = np.empty((length, geometry.altitude.size), dtype=bool)
                        serial_shade = np.empty_like(serial_sun)
                        for patch in range(geometry.altitude.size):
                            sun, shade = engine.shaded_or_sunlit(
                                altitude, azimuth, geometry.altitude[patch],
                                geometry.azimuth[patch], field2d,
                            )
                            serial_sun[:, patch] = sun[:, 0]
                            serial_shade[:, patch] = shade[:, 0]
                        sm = int(np.count_nonzero(serial_sun != expected_sun))
                        hm = int(np.count_nonzero(serial_shade != expected_shade))
                        totals["serial_sun"] += sm
                        totals["serial_shade"] += hm
                        cases.append({
                            "path": "serial", "timestep": timestep["timestep"],
                            "threads": thread_count, "layout": layout,
                            "length": length, "offset": offset, "block": None,
                            "sun_mismatches": sm, "shade_mismatches": hm,
                        })
        print(json.dumps({"timestep": timestep["timestep"], "totals": totals}), flush=True)

    output = {
        "schema": "solweig-light.portable-profile-v1-linux-matrix.v1",
        "profile": profile_identity(),
        "torch_imported": "torch" in sys.modules,
        "threads": list(threads), "layouts": ["contiguous", "stride2"],
        "lengths": list(lengths), "offsets": list(offsets), "blocks": list(blocks),
        "case_count": len(cases), "totals": totals,
        "passed": all(value == 0 for value in totals.values()),
        "cases": cases,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": output["passed"], "case_count": len(cases), "totals": totals}))
    return 0 if output["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
