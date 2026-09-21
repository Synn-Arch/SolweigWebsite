"""Freeze accepted domain/dtype bindings before portable-profile admission."""
import hashlib, inspect, json, platform, sys
from pathlib import Path
import numpy as np
import solweig_light
from solweig_light.radiation import engine, patch_radiation, _math_profile as profile

OUT = Path(sys.argv[1])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
functions = ('thermal_comfort','preprocess','build_inputs','build_wind_ext_coeff','run_walls_aspect','calculate_svf','run_utci_tiles')
values = np.array([-np.inf,-10000,-125,-124.999,-1,-0.,0.,1.,124.999,125,10000,np.inf,np.nan],np.float32)
with np.errstate(all='ignore'):
 tan=profile.tan32(values); legacy=np.tan(values); atan=profile.atan32(values)
report={
 'schema':'solweig-light.portable-profile-domain-audit.v1',
 'profile':profile.profile_identity(),
 'public_signatures':{name:str(inspect.signature(getattr(solweig_light,name))) for name in functions},
 'affected_entry_points':{
  'pipeline_asvf':{'input':'internal geometry SVF ndarray','production_dtype':'float32','physical_values':'[0,1]','legacy_invalid':'sqrt/acos NaN propagation','promoted_dtype':'explicit preserved NumPy sqrt/acos'},
  'shaded_or_sunlit':{'signature':str(inspect.signature(engine.shaded_or_sunlit)),'asvf':'array-like; arbitrary finite/nonfinite float values historically accepted by NumPy candidate','solar_altitude':'tensor-origin zero-dimensional ndarray required by frozen compatibility behavior','coefficient':'float64 NumPy cos/tan scalar grouping'},
  'prepared_classifier':{'class_coefficients':str(inspect.signature(patch_radiation._class_coefficients)),'classes':str(inspect.signature(patch_radiation._classes)),'supported_fast_path':'float32 fields and patch tables; scalar ndarray solar inputs','fallback':'unchanged serial engine path for unsupported dtype/layout'},
 },
 'tan_dispatch':{'source_derived_domain':'finite float32 abs(x)<125','preserved_domain':'nonfinite or abs(x)>=125 uses explicit legacy NumPy tan','input_bits':values.view(np.uint32).tolist(),'actual_bits':tan.view(np.uint32).tolist(),'legacy_bits':legacy.view(np.uint32).tolist(),'nan_masks_equal':bool(np.array_equal(np.isnan(tan),np.isnan(legacy)))},
 'atan':{'accepted':'all float32 finite/nonfinite via source-derived helper; promoted dtypes explicit NumPy path','bits':atan.view(np.uint32).tolist()},
 'source':{str(Path(p).name):sha(p) for p in (profile.__file__, inspect.getsourcefile(engine), inspect.getsourcefile(patch_radiation))},
 'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__},
 'nonclaims':['Domain audit does not admit numerical parity.','The explicit large/nonfinite tangent path preserves accepted behavior; it is not used by physical ASVF.']}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
