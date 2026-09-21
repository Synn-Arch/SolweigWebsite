"""Original real-scene angular equality boundaries and deterministic workers."""
import hashlib
import json
import sys
from pathlib import Path

import numba
import numpy as np
import pytest
from solweig_light.radiation import ground_view,engine

ROOT=Path(__file__).parents[1]/'reference/p7_ground_view_angular_original_cpu'
MANIFEST=json.loads((ROOT/'manifest.json').read_text())

def load(case,key):
 artifact=case[key];p=ROOT/artifact['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==artifact['sha256']
 with np.load(p,allow_pickle=False) as archive:return {k:v[()] if not v.ndim else v.copy() for k,v in archive.items()}

@pytest.mark.parametrize('case',MANIFEST['cases'],ids=lambda c:str(c['angle'])+'_'+c.get('profile','float32'))
@pytest.mark.parametrize('threads',[None,1,4,'fallback'])
def test_original_angular_boundaries(case,threads):
 assert MANIFEST['evidence_class']=='original_upstream_cpu'
 assert MANIFEST['source_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
 assert MANIFEST['patch'] is None and not MANIFEST['candidate_import']
 previous=numba.get_num_threads()
 try:
  if threads not in (None,'fallback'):numba.set_num_threads(threads)
  fn=(engine.sunonsurface_2018a_numpy if threads=='fallback' else ground_view.sunonsurface_2018a if threads is None else ground_view.sunonsurface_2018a_parallel)
  masks={};prior_profile=sys.getprofile()
  def capture(frame,event,result):
   if event=='return' and result is not None and 'facesh' in frame.f_locals and frame.f_code in (ground_view._sun.__code__,engine.sunonsurface_2018a_numpy.__code__):
    masks['facesh']=frame.f_locals['facesh'].copy()
  try:
   sys.setprofile(capture)
   with np.errstate(invalid='ignore',divide='ignore'):actual=fn(**load(case,'inputs'))
  finally:sys.setprofile(prior_profile)
  np.testing.assert_array_equal(masks['facesh'],load(case,'mask')['facesh'])
  expected=load(case,'outputs')
  for index,value in enumerate(actual):
   golden=expected['return/'+str(index)]
   assert value.dtype==golden.dtype and value.shape==golden.shape
   if index==1:np.testing.assert_allclose(value,golden,atol=.05,rtol=1e-5,equal_nan=True)
   else:np.testing.assert_allclose(value,golden,atol=1e-6,rtol=0,equal_nan=True)
 finally:numba.set_num_threads(previous)
