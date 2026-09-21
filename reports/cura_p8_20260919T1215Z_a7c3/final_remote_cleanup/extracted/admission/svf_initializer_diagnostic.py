"""Execute option-2 SVF initializer arithmetic without shadow/SVF evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("upstream", "candidate"), required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "upstream":
        import torch
        from solweig_gpu.shadow import create_patches
        _, _, _, alts, counts, steps0, starts = create_patches(2)
        steps = torch.tensor([360 / patches for patches in counts], device="cpu")
        azis = torch.zeros((1, torch.sum(counts).item()), device="cpu")
        written, index = [], 0
        for band in range(alts.shape[0]):
            count = 0
            for k in range(int(360 / steps[band])):
                azis[0, index] = k * steps[band] + starts[band]
                if azis[0, index] > 360.0:
                    azis[0, index] -= 360.0
                index += 1
                count += 1
            written.append(count)
        result = dict(torch_version=torch.__version__, default_dtype=str(torch.get_default_dtype()),
                      original_step_dtype=str(steps0.dtype), effective_step_dtype=str(steps.dtype),
                      counts_dtype=str(counts.dtype), starts_dtype=str(starts.dtype),
                      declared_counts=counts.cpu().tolist(), effective_steps=steps.cpu().tolist(),
                      starts=starts.cpu().tolist(), written_counts=written,
                      initialized_total=index, declared_total=int(torch.sum(counts).item()),
                      full_azimuth_array=azis[0].cpu().tolist())
    else:
        import numpy as np
        from solweig_light.geometry.shadows import create_patches
        _, _, _, alts, counts, steps, starts = create_patches(2)
        azis = np.zeros(sum(counts), dtype=np.float32)
        written, index = [], 0
        for band in range(len(alts)):
            count = 0
            for k in range(int(np.reciprocal(steps[band]) * np.float32(360))):
                azis[index] = np.float32(k) * steps[band] + np.float32(starts[band])
                if azis[index] > 360:
                    azis[index] -= np.float32(360)
                index += 1
                count += 1
            written.append(count)
        result = dict(numpy_version=np.__version__, steps_dtype=str(steps.dtype),
                      counts_dtype=str(counts.dtype), starts_dtype=str(starts.dtype),
                      declared_counts=counts.tolist(), effective_steps=steps.tolist(),
                      starts=starts.tolist(), written_counts=written,
                      initialized_total=index, declared_total=int(sum(counts)),
                      full_azimuth_array=azis.tolist())
    result.update(schema="svf-initializer-execution-v1", backend=args.backend,
                  option=2, source_file=str(args.source_file.resolve()),
                  source_sha256=sha(args.source_file),
                  scope="initializer arithmetic only; no shadows or SVF evaluated")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
