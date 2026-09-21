"""Source-verified original weights and near-zero SVFsum boundary receivers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from solweig_light.geometry.svf import annulus_weight

ROOT=Path(__file__).parents[1]/'reference/p7_svf_annulus_original_cpu'
MANIFEST=json.loads((ROOT/'manifest.json').read_text())

def checked(name,digest):
 p=ROOT/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 with np.load(p,allow_pickle=False) as archive:return dict(archive)

def test_original_weight_scalar_cpu_characterization():
 assert MANIFEST['evidence_class']=='original_upstream_cpu' and MANIFEST['patch'] is None
 assert MANIFEST['source_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
 archive=checked('original_annulus_weights.npz',MANIFEST['artifact_sha256'])
 actual=np.array([annulus_weight(a,b) for a,b in zip(archive['altitude'],archive['interval'])],dtype=np.float32)
 # Native scalar sinf and original Torch CPU sine differ by one ULP for six
 # entries. This characterization is explicit; it does not loosen SVF or final gates.
 np.testing.assert_array_max_ulp(actual,archive['weight'],maxulp=1)
 assert np.count_nonzero(actual==archive['weight'])>=174

@pytest.mark.parametrize('case',MANIFEST['boundary_cases'],ids=lambda c:c['scene'])
def test_original_visibility_receiver_boundary_sum(case):
 archive=checked(case['path'],case['sha256']);masks=archive['masks'];ann=archive['annulus'];counts=archive['counts'];total=np.zeros(2,dtype=np.float32);index=0
 for band,count in enumerate(counts):
  for _ in range(count):
   for altitude in range(ann[band]+1,ann[band+1]+1):
    weight=annulus_weight(altitude,count)
    total+=weight*masks[:,index]
   index+=1
 np.minimum(total,np.float32(1),out=total)
 np.testing.assert_array_equal(total,np.array([archive['svf'],archive['svfveg']],dtype=np.float32))
 tmp=np.maximum(total[0]+total[1]-np.float32(1),np.float32(0))
 alpha=np.arcsin(np.exp(np.log(np.float32(1)-tmp)/np.float32(2)))
 np.testing.assert_array_equal(alpha,archive['svfalfa'])
 assert tmp==np.float32(2**-23)

def test_original_scalar_zero_and_nonfinite_weight_semantics():
 edge=MANIFEST['edge_artifact'];archive=checked(edge['path'],edge['sha256'])
 with np.errstate(all='ignore'):
  actual=np.array([annulus_weight(a,b) for a,b in archive['inputs']],dtype=np.float32)
 expected=archive['outputs']
 for fn in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(fn(actual),fn(expected))
 finite=np.isfinite(expected)
 np.testing.assert_array_equal(actual[finite],expected[finite])
 np.testing.assert_array_equal(np.signbit(actual[finite]),np.signbit(expected[finite]))
