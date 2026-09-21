"""Emit the same synthetic 17-field ground-view fixture for one isolated backend."""
import argparse
from pathlib import Path
import numpy as np


def fixture(landcover):
    rows, cols = 9, 11
    row, col = np.indices((rows, cols))
    buildings = np.ones((rows, cols), dtype=np.float32)
    buildings[(row + 2 * col) % 7 == 0] = 0
    walls = (1.0 + ((3 * row + col) % 5) * 0.4).astype(np.float32)
    return {
        "wallsun": np.where((2 * row + col) % 4 == 0, walls, 0).astype(np.float32),
        "walls": walls,
        "buildings": buildings,
        "scale": np.float32(1),
        "shadow": np.asarray((0.0, 0.35, 1.0), np.float32)[(row + col) % 3],
        "first": np.float32(2), "second": np.float32(4),
        "dirwalls": np.asarray((2, 44, 89, 91, 179, 181, 269, 271, 358), np.float32)[(2 * row + 3 * col) % 9],
        "Tg": (1.5 + row * 0.23 - col * 0.07).astype(np.float32),
        "Tgwall": np.float32(4.25), "Ta": np.float32(21.5),
        "emis_grid": (0.91 + ((row + col) % 5) * 0.012).astype(np.float32),
        "ewall": np.float32(0.90),
        "alb_grid": (0.08 + ((2 * row + col) % 7) * 0.045).astype(np.float32),
        "SBC": np.float32(5.67051e-8), "albedo_b": np.float32(0.20),
        "rows": rows, "cols": cols, "Twater": np.float32(18.0),
        "lc_grid": ((row + 2 * col) % 6 + 1).astype(np.int16),
        "landcover": np.float32(landcover),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("upstream", "candidate"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = {}
    for landcover in (0, 1):
        values = fixture(landcover)
        original_tg = values["Tg"].copy()
        if args.backend == "upstream":
            import torch
            from solweig_gpu.solweig import gvf_2018a
            converted = {}
            for key, value in values.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    converted[key] = torch.as_tensor(value.copy())
                else:
                    converted[key] = value
            outputs = gvf_2018a(**converted)
            arrays = [item.detach().cpu().numpy() for item in outputs]
            mutated_tg = converted["Tg"].detach().cpu().numpy()
        else:
            from solweig_light.radiation.ground_view import gvf_2018a
            outputs = gvf_2018a(**values)
            arrays = [np.asarray(item) for item in outputs]
            mutated_tg = values["Tg"]
        for index, array in enumerate(arrays):
            payload[f"land{landcover}_field{index}"] = array
        payload[f"land{landcover}_tg_before"] = original_tg
        payload[f"land{landcover}_tg_after"] = mutated_tg
    np.savez(args.output, **payload)


if __name__ == "__main__":
    main()
