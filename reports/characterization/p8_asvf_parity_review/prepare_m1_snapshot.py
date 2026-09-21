"""Create an isolated reviewable M1 parity experiment; never modify main src."""
from pathlib import Path
import shutil,json,hashlib,difflib

HERE=Path(__file__).parent.resolve()
ROOT=HERE.parents[2]
OUT=HERE/'m1_pipeline_experiment'
SNAP=OUT/'snapshot'
if SNAP.exists():raise FileExistsError(SNAP)
SNAP.mkdir(parents=True)
for name in ('pyproject.toml','LICENSE','README.md'):
    shutil.copy2(ROOT/name,SNAP/name)
shutil.copytree(ROOT/'src/solweig_light',SNAP/'src/solweig_light',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
PKG=SNAP/'src/solweig_light'
for original,name in [('sleef_acos_prototype.py','_sleef_acos.py'),('sleef_classifier_prototype.py','_sleef_classifier.py')]:
    source=(HERE/original).read_text().split("if __name__ == '__main__':")[0]
    source=source.replace('from sleef_acos_prototype import','from ._sleef_acos import')
    source=source.replace('sleef_source/LICENSE.txt','SLEEF_LICENSE.txt')
    (PKG/'radiation'/name).write_text(source)
shutil.copy2(HERE/'sleef_source/LICENSE.txt',PKG/'radiation/SLEEF_LICENSE.txt')
(PKG/'radiation/_math_profile.py').write_text('''"""Explicit isolated M1 SLEEF profile; unpromoted experiment."""
import numpy as np
from ._sleef_acos import asvf_fma
from ._sleef_classifier import tan_array, atan_array

PROFILE = "m1-sleef-5a1d179d-fma-f32-classifier-experiment-v1"

def asvf(values):
    values=np.asarray(values)
    if values.dtype != np.float32:
        raise TypeError("Experiment ASVF requires float32 SVF")
    return asvf_fma(values.ravel()).reshape(values.shape)

def tan32(values):
    values=np.asarray(values)
    if values.dtype != np.float32:
        return np.tan(values)
    return tan_array(values.ravel()).reshape(values.shape)

def atan32(values):
    values=np.asarray(values)
    if values.dtype != np.float32:
        return np.arctan(values)
    return atan_array(values.ravel()).reshape(values.shape)
''')

def edit(name,old,new):
    p=PKG/name;s=p.read_text()
    assert s.count(old)==1,(name,old,s.count(old))
    p.write_text(s.replace(old,new))

edit('pipeline.py','    asvf = np.arccos(np.sqrt(svf))','    from .radiation._math_profile import asvf as prepare_asvf\n    asvf = prepare_asvf(svf)')
edit('radiation/engine.py','    hsvf = np.tan(asvf)','    from ._math_profile import tan32, atan32\n    hsvf = tan32(asvf)')
edit('radiation/engine.py','    sunlit_degrees = _operate(np.multiply, np.arctan(tan_delta), rad2deg)','    sunlit_degrees = _operate(np.multiply, atan32(tan_delta), rad2deg)')
edit('radiation/patch_radiation.py','            delta=np.add(np.tan(field),coefficients[None,:])','            from ._math_profile import tan32, atan32\n            delta=np.add(tan32(field),coefficients[None,:])')
edit('radiation/patch_radiation.py','            degrees=np.multiply(np.arctan(delta),np.asarray(rad2deg,dtype=field.dtype))','            degrees=np.multiply(atan32(delta),np.asarray(rad2deg,dtype=field.dtype))')
edit('identities.py',"'policy': 'legacy-logical-domain-geometry-v1',", "'math_profile': __import__('solweig_light.radiation._math_profile', fromlist=['PROFILE']).PROFILE,\n        'policy': 'legacy-logical-domain-geometry-v1',")
edit('identities.py',"'policy': 'chronological-compatibility-v1',", "'math_profile': __import__('solweig_light.radiation._math_profile', fromlist=['PROFILE']).PROFILE,\n        'policy': 'chronological-compatibility-v1',")
edit('identities.py',"'identities.py', 'pipeline.py',", "'identities.py', 'pipeline.py',\n            'radiation/_sleef_acos.py', 'radiation/_sleef_classifier.py', 'radiation/_math_profile.py',")
p=SNAP/'pyproject.toml';p.write_text(p.read_text().replace('"comfort/*.json"]','"comfort/*.json", "radiation/SLEEF_LICENSE.txt"]'))

def inventory(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
base=inventory(ROOT/'src/solweig_light');candidate=inventory(PKG)
(OUT/'source_manifest.json').write_text(json.dumps({'baseline':base,'candidate':candidate,'changed':[k for k in candidate if candidate[k]!=base.get(k)]},indent=2)+'\n')
diff=[]
for k in candidate:
    if candidate[k]!=base.get(k):
        left=(ROOT/'src/solweig_light'/k).read_text().splitlines(True) if k in base else []
        diff.extend(difflib.unified_diff(left,(PKG/k).read_text().splitlines(True),fromfile='a/src/solweig_light/'+k,tofile='b/src/solweig_light/'+k))
(OUT/'source.patch').write_text(''.join(diff))
print(SNAP)
