import hashlib
import json
from pathlib import Path
import numba
import numpy as np
import pytest
from solweig_light.radiation import engine, ground_view
from solweig_light.runtime import RuntimeOptions, runtime_options

ROOT=Path("/Users/alansynn/Workspace/solweig-light")
SNAPSHOT=Path(__file__).parent/"src"
assert Path(engine.__file__).resolve().is_relative_to(SNAPSHOT.resolve())
REFERENCE=ROOT/"tests/reference/ground_view_original_cpu"
MANIFEST=json.loads((REFERENCE/"manifest.json").read_text())
CASES=[c for c in MANIFEST["cases"] if c["function"]=="gvf_2018a"]

def fresh(case):
    artifact=case["input_artifact"];path=REFERENCE/artifact["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact["sha256"]
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k][()] if z[k].ndim==0 else z[k].copy() for k in z.files}

def exact(a,b):
    a=np.asarray(a);b=np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype
    if a.dtype.kind=="f":
        np.testing.assert_array_equal(np.isnan(a),np.isnan(b))
        valid=~np.isnan(a)
        np.testing.assert_array_equal(a[valid].view("u"+str(a.dtype.itemsize)),b[valid].view("u"+str(b.dtype.itemsize)))
    else: np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize("case",CASES,ids=lambda c:c["name"])
@pytest.mark.parametrize("threads",[1,4,10])
def test_dispatch_preserves_every_field_and_mutation(case,threads):
    left,right=fresh(case),fresh(case)
    previous=numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        with np.errstate(all="ignore"):
            if case["status"]=="failed":
                error={"RuntimeError":RuntimeError,"UnboundLocalError":UnboundLocalError}[case["exception"]]
                with pytest.raises(error):ground_view.gvf_2018a(**left)
                with runtime_options(RuntimeOptions(cpu_budget=threads,threads_per_worker=threads)):
                    with pytest.raises(error):engine.gvf_2018a(**right)
            else:
                expected=ground_view.gvf_2018a(**left)
                with runtime_options(RuntimeOptions(cpu_budget=threads,threads_per_worker=threads)):
                    actual=engine.gvf_2018a(**right)
                assert len(actual)==len(expected)==17
                for a,b in zip(actual,expected):exact(a,b)
        for key in left:exact(left[key],right[key])
    finally:numba.set_num_threads(previous)
