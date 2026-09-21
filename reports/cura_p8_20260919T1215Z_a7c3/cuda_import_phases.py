"""Bounded CUDA startup phases for process-memory attribution."""
import argparse
import json
import os
from pathlib import Path
import time


def mark(path, phase, **extra):
    with path.open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "phase": phase, "pid": os.getpid(), **extra}) + "\n")
        stream.flush()


p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("torch_import", "cuda_query", "upstream_import", "pool_probe"), required=True)
p.add_argument("--phases", type=Path, required=True)
a = p.parse_args()
mark(a.phases, "start")
import torch
mark(a.phases, "torch_imported", torch=torch.__version__, compiled_cuda=torch.version.cuda)
if a.mode in ("cuda_query", "upstream_import", "pool_probe"):
    available = torch.cuda.is_available()
    mark(a.phases, "cuda_available", available=available, count=torch.cuda.device_count())
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    mark(a.phases, "device_initialized", name=props.name)
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    mark(a.phases, "tensor_synchronized", sum=float(x.sum().cpu()))
if a.mode in ("upstream_import", "pool_probe"):
    import solweig_gpu
    mark(a.phases, "upstream_imported", package=solweig_gpu.__file__)
if a.mode == "pool_probe":
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1))
    future = pool.submit(time.sleep, 1.0)
    mark(a.phases, "pool_submitted", configured_workers=min(32, os.cpu_count() or 1))
    future.result()
    pool.shutdown()
    mark(a.phases, "pool_complete")
time.sleep(0.5)
mark(a.phases, "complete")
