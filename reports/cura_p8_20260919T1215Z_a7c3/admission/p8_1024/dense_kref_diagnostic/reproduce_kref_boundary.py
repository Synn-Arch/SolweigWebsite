"""Reproduce dense1024 band-8 Kside boundary from the captured 1-pixel fixture."""
import json, sys
from pathlib import Path
import numpy as np

from solweig_light.radiation import engine, patch_radiation

fixture=Path(sys.argv[1] if len(sys.argv)>1 else "kref_step7_candidate_input.npz")
with np.load(fixture) as z:
    kwargs={k:z[k] for k in z.files if k!="compiled_output_at_point"}
kwargs.update(rows=1,cols=1,cyl=True,anisotropic_diffuse=1)
candidate_asvf=np.array([[1057360531]],dtype=np.uint32).view(np.float32)
upstream_torch_asvf=np.array([[1057360530]],dtype=np.uint32).view(np.float32)

def evaluate(asvf):
    values=dict(kwargs,asvf=asvf)
    serial=engine._serial_Kside_veg_v2022a(**values)
    compiled=engine.Kside_veg_v2022a(**values)
    geometry=patch_radiation.patch_geometry(values["lv"])
    altitude=engine._array(values["altitude"])
    prepared=patch_radiation._class_coefficients(altitude,values["azimuth"],geometry,asvf)
    sun,shade=patch_radiation._classes(altitude,values["azimuth"],geometry,asvf,0,1,prepared=prepared)
    return [float(v[0,0]) for v in serial],[float(v[0,0]) for v in compiled],sun[0],shade[0],geometry

cs,cc,csun,cshade,geometry=evaluate(candidate_asvf)
us,uc,usun,ushade,_=evaluate(upstream_torch_asvf)
assert cs==cc and us==uc
changed=np.flatnonzero(cshade!=ushade)
assert changed.tolist()==list(range(61,74))+[88]
assert np.array_equal(csun,usun)

surface_sh=np.float32(np.float32(np.float32(kwargs["albedo"])*np.float32(kwargs["radD"]))*np.float32(.5)/np.float32(np.pi))
building=(np.float32(1)-kwargs["shmat"][0,0])*kwargs["vbshvegshmat"][0,0]==1
per_patch=[]
for patch in changed:
    contribution=np.float32(np.float32(np.float32(surface_sh*geometry.solid_angle[patch])*geometry.cosine[patch])*building[patch])
    per_patch.append({"patch":int(patch),"altitude":float(geometry.altitude[patch]),"azimuth":float(geometry.azimuth[patch]),"building":bool(building[patch]),"candidate_extra_Kref_sh":float(contribution)})

result={
 "candidate_asvf":{"value":float(candidate_asvf[0,0]),"uint32_bits":int(candidate_asvf.view(np.uint32)[0,0])},
 "upstream_torch_asvf":{"value":float(upstream_torch_asvf[0,0]),"uint32_bits":int(upstream_torch_asvf.view(np.uint32)[0,0])},
 "classifier":{"sun_sets_equal":True,"shade_changed_patches":changed.tolist()},
 "per_patch":per_patch,
 "candidate_extra_Kref_sh_sum":float(np.sum(np.array([x["candidate_extra_Kref_sh"] for x in per_patch],dtype=np.float32),dtype=np.float32)),
 "serial_outputs_candidate_asvf":cs,
 "serial_outputs_upstream_asvf":us,
 "compiled_outputs_candidate_asvf":cc,
 "compiled_outputs_upstream_asvf":uc,
 "Kside_difference":float(cs[6]-us[6]),
}
print(json.dumps(result,indent=2,sort_keys=True))
