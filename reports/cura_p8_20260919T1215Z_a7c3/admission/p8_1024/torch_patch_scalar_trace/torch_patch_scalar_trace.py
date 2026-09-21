from __future__ import annotations
import hashlib, inspect, json, os, platform, sys
from pathlib import Path
import numpy as np
import torch
import solweig_gpu.solweig as model

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

corpus_path, reference_path, output_dir = map(Path, sys.argv[1:])
output_dir.mkdir(parents=True, exist_ok=False)
corpus = np.load(corpus_path)
reference = np.load(reference_path)
boundary_bits = corpus['boundary_bits']
boundary = boundary_bits.view(np.float32)
pa = torch.from_numpy(corpus['patch_altitude'].copy())
pazi = torch.from_numpy(corpus['patch_azimuth'].copy())
sa = torch.tensor(corpus['solar_altitude'][()], dtype=torch.float64)
saz = torch.tensor(corpus['solar_azimuth'][()], dtype=torch.float64)
asvf = torch.acos(torch.sqrt(torch.from_numpy(boundary.copy())))[:, None]

captures = {'cos': [], 'tan': [], 'atan': []}
original = {name: getattr(torch, name) for name in captures}
def intercept(name):
    def wrapped(value, *args, **kwargs):
        result = original[name](value, *args, **kwargs)
        captures[name].append((value.detach().cpu().clone(), result.detach().cpu().clone()))
        return result
    return wrapped
for name in captures:
    setattr(torch, name, intercept(name))
sun_columns, shade_columns = [], []
try:
    for index in range(pa.numel()):
        sun, shade = model.shaded_or_sunlit(sa, saz, pa[index], pazi[index], asvf)
        sun_columns.append(sun.reshape(-1, 1).cpu().numpy())
        shade_columns.append(shade.reshape(-1, 1).cpu().numpy())
finally:
    for name, function in original.items():
        setattr(torch, name, function)

sun = np.concatenate(sun_columns, axis=1)
shade = np.concatenate(shade_columns, axis=1)
oracle_sun = np.unpackbits(reference['sun_pack'], axis=1, bitorder='big')[:, :pa.numel()].astype(bool)
oracle_shade = np.unpackbits(reference['shade_pack'], axis=1, bitorder='big')[:, :pa.numel()].astype(bool)
assert np.array_equal(sun, oracle_sun)
assert np.array_equal(shade, oracle_shade)

# Each call has cos(patch relation), tan(solar scalar), tan(ASVF), atan(full field).
cos_in = torch.stack([x[0] for x in captures['cos']]).numpy()
xi = torch.stack([x[1] for x in captures['cos']]).numpy()
solar_tan_in = torch.stack([captures['tan'][2*i][0] for i in range(pa.numel())]).numpy()
solar_tan = torch.stack([captures['tan'][2*i][1] for i in range(pa.numel())]).numpy()
asvf_tan_in = torch.stack([captures['tan'][2*i+1][0].reshape(-1) for i in range(pa.numel())]).numpy()
hsvf = torch.stack([captures['tan'][2*i+1][1].reshape(-1) for i in range(pa.numel())], dim=1).numpy()
tan_delta = torch.stack([x[0].reshape(-1) for x in captures['atan']], dim=1).numpy()
atan = torch.stack([x[1].reshape(-1) for x in captures['atan']], dim=1).numpy()
rad2deg = 180.0 / torch.pi
degrees = torch.stack([torch.from_numpy(atan[:, i]) * rad2deg for i in range(pa.numel())], dim=1).numpy()
deg2rad = torch.pi / 180.0
yi_t = torch.stack([2 * torch.from_numpy(xi)[i] * torch.from_numpy(solar_tan)[i] for i in range(pa.numel())])
yi = yi_t.numpy()
yi_clamped_t = torch.stack([torch.where(yi_t[i] > 0, 0.0, yi_t[i]) for i in range(pa.numel())])
yi_clamped = yi_clamped_t.numpy()
replay_delta = torch.stack([torch.from_numpy(hsvf[:, i]) + yi_clamped_t[i] for i in range(pa.numel())], dim=1)
assert np.array_equal(replay_delta.numpy(), tan_delta)
assert np.array_equal(degrees < pa.numpy()[None, :], sun)
assert np.array_equal(degrees > pa.numpy()[None, :], shade)

np.savez_compressed(output_dir/'trace.npz', boundary_bits=boundary_bits, asvf_bits=asvf.numpy().reshape(-1).view(np.uint32),
    patch_altitude=pa.numpy(), patch_azimuth=pazi.numpy(), solar_altitude=sa.numpy(), solar_azimuth=saz.numpy(),
    cos_input=cos_in, xi=xi, solar_tan_input=solar_tan_in, solar_tan=solar_tan,
    asvf_tan_input=asvf_tan_in, hsvf=hsvf, yi=yi, yi_clamped=yi_clamped,
    tan_delta=tan_delta, atan=atan, degrees=degrees,
    sun_pack=np.packbits(sun,axis=1,bitorder='big'), shade_pack=np.packbits(shade,axis=1,bitorder='big'))
def info(a): return {'shape':list(a.shape),'dtype':str(a.dtype)}
meta={'status':'passed','threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
      'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),
      'package':str(Path(inspect.getfile(model)).resolve()),'source_sha256':hashlib.sha256(Path(inspect.getfile(model)).read_bytes()).hexdigest(),
      'corpus_sha256':hashlib.sha256(corpus_path.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
      'oracle_masks_exact':True,'arrays':{k:info(v) for k,v in {'cos_input':cos_in,'xi':xi,'solar_tan':solar_tan,'hsvf':hsvf,'yi':yi,'yi_clamped':yi_clamped,'tan_delta':tan_delta,'atan':atan,'degrees':degrees,'sun':sun,'shade':shade}.items()},
      'capture_counts':{k:len(v) for k,v in captures.items()}}
(output_dir/'meta.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
print(json.dumps(meta,sort_keys=True))
