#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from solweig_light.radiation import patch_radiation as candidate
from solweig_light.radiation.angular_moments import build_bundle
from tools.experiments.p7_angular_moment import load_packet,compare_tuple

ROOT=Path(__file__).resolve().parents[3]
PACKET_ROOT=ROOT/'tests/reference/patch_radiation_original_cpu'
MANIFEST=json.loads((PACKET_ROOT/'manifest.json').read_text())
results=[]
for case in MANIFEST['cases']:
    if case['status']!='captured' or case['function'] not in ('Lcyl_v2022a','define_patch_characteristics'):continue
    values=load_packet(case,MANIFEST);rows,cols=values['rows'],values['cols']
    if case['function']=='Lcyl_v2022a':geometry=candidate.patch_geometry(values['sky_patches']);solid=geometry.solid_angle
    else:geometry=candidate.patch_geometry(np.column_stack((values['patch_altitude'],values['patch_azimuth'])));solid=values['steradian']
    for name in ('shmat','vegshmat','vbshvegshmat'):
        original=values[name]
        values[name]=np.frombuffer(original.tobytes(order='C'),dtype=np.float32).reshape(original.shape)
    bundle=build_bundle(values['shmat'],values['vegshmat'],values['vbshvegshmat'],solid,
        geometry.sine,geometry.cosine,geometry.longwave_cardinal_cosine,geometry.reflection_cardinal,
        logical_shape=(rows,cols),profile='strict-f32-v1',source_fingerprint='p7-angular-snapshot-v2')
    with np.errstate(all='ignore'):
        output=getattr(candidate,case['function'])(**values,parallel=False,block_pixels=7,_longwave_moments=bundle)
    worst=compare_tuple(output,PACKET_ROOT/case['output'])
    results.append({'label':case['label'],'function':case['function'],'max_abs':worst,'status':'pass'})
path=Path(__file__).with_name('evidence')/'component_results.json'
path.write_text(json.dumps({'status':'pass','count':len(results),'results':results},indent=2,sort_keys=True)+'\n')
print(f'{len(results)} original component cases passed')
