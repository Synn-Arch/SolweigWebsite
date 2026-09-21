"""Actual upstream per-patch classifier oracle and isolated candidate probe."""
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / 'm1_classifier'
CORPUS = ROOT / 'reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/classifier_mkl_diagnostic/corpus.npz'


def inputs():
    z = np.load(CORPUS)
    frozen = []
    sources = {}
    for p in sorted((ROOT/'tests/reference/svf_original_cpu').glob('*outputs.npz')):
        data = np.load(p)
        if 'svf' in data:
            frozen.append(data['svf'].ravel().astype(np.float32))
            sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    if not frozen:
        raise ValueError('No frozen SVF input fields found')
    frozen = np.unique(np.concatenate(frozen).view(np.uint32))
    bits = np.concatenate((z['boundary_bits'], frozen))
    return z, bits.view(np.float32), {'boundary_count':len(z['boundary_bits']), 'frozen_unique_count':len(frozen), 'frozen_sources':sources}


def oracle():
    import torch
    from solweig_gpu import solweig
    z, x, meta = inputs()
    torch.set_num_threads(1)
    asvf = torch.acos(torch.sqrt(torch.from_numpy(x.copy())))[:, None]
    pa = torch.from_numpy(z['patch_altitude'].copy())
    pazi = torch.from_numpy(z['patch_azimuth'].copy())
    sa, saz = torch.tensor(z['solar_altitude']), torch.tensor(z['solar_azimuth'])
    original = {name:getattr(torch,name) for name in ('cos','tan','atan')}
    trace = []
    def wrap(name):
        def call(value):
            out = original[name](value)
            trace.append((name, out.detach().numpy().copy(), str(value.dtype), list(value.shape)))
            return out
        return call
    for name in original:
        setattr(torch,name,wrap(name))
    suns, shades, xis, ats = [], [], [], []
    for p in range(len(pa)):
        trace.clear()
        sun, shade = solweig.shaded_or_sunlit(sa, saz, pa[p], pazi[p], asvf)
        suns.append(sun.numpy()[:,0]); shades.append(shade.numpy()[:,0])
        xis.append(trace[0][1]); ats.append(trace[-1][1][:,0])
        if p == 0:
            meta['actual_unary_call_dtypes_shapes'] = [{'name':n,'dtype':d,'shape':s} for n,v,d,s in trace]
            hsvf = trace[2][1][:,0]
            solar_tan = trace[1][1]
    for name,fn in original.items():
        setattr(torch,name,fn)
    np.savez_compressed(OUT/'oracle.npz', inputs=x, asvf=asvf.numpy()[:,0],
        sun=np.column_stack(suns), shade=np.column_stack(shades), xi=np.asarray(xis),
        hsvf=hsvf, solar_tan=solar_tan, atan=np.column_stack(ats))
    meta.update(torch=torch.__version__, torch_git=torch.version.git_version,
        function_source=str(Path(solweig.__file__).resolve()), input_count=len(x), patches=len(pa))
    (OUT/'oracle.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


def candidate():
    import sleef_acos_prototype as proto
    from solweig_light.radiation import engine as e, patch_radiation as p
    z,x,meta=inputs(); ref=np.load(OUT/'oracle.npz')
    asvf=proto.asvf_fma(x)[:,None]
    pa,pazi=z['patch_altitude'],z['patch_azimuth']
    sa,saz=np.asarray(z['solar_altitude']),np.asarray(z['solar_azimuth'])
    serial=[e.shaded_or_sunlit(sa,saz,pa[i],pazi[i],asvf) for i in range(len(pa))]
    ss=np.column_stack([v[0][:,0] for v in serial]);sh=np.column_stack([v[1][:,0] for v in serial])
    geom=p.patch_geometry(np.column_stack((pa,pazi)))
    prepared=p._class_coefficients(sa,saz,geom,asvf)
    ps,ph=p._classes(sa,saz,geom,asvf,0,len(x),prepared=prepared)
    xi=np.array([np.cos(e._operate(np.multiply,np.abs(e._operate(np.subtract,saz,pazi[i])),e._divide(np.pi,180.))) for i in range(len(pa))])
    results={'asvf_bit_mismatches':int(np.count_nonzero(asvf[:,0].view(np.uint32)!=ref['asvf'].view(np.uint32))),
        'xi64_bit_mismatches':int(np.count_nonzero(xi.view(np.uint64)!=ref['xi'].view(np.uint64))),
        'hsvf32_bit_mismatches':int(np.count_nonzero(np.tan(asvf[:,0]).view(np.uint32)!=ref['hsvf'].view(np.uint32))),
        'solar_tan64_bits_equal':bool(np.array_equal(np.tan(e._operate(np.multiply,sa,e._divide(np.pi,180.))).view(np.uint64),ref['solar_tan'].view(np.uint64))),
        'serial':{'sun':int(np.count_nonzero(ss!=ref['sun'])),'shade':int(np.count_nonzero(sh!=ref['shade']))},
        'prepared':{'sun':int(np.count_nonzero(ps!=ref['sun'])),'shade':int(np.count_nonzero(ph!=ref['shade']))},
        'serial_prepared_equal':bool(np.array_equal(ss,ps) and np.array_equal(sh,ph)),**meta}
    np.savez_compressed(OUT/'candidate.npz',sun=ss,shade=sh,prepared_sun=ps,prepared_shade=ph)
    (OUT/'candidate.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


def revised():
    import inspect
    import sleef_acos_prototype as a
    import sleef_classifier_prototype as t
    from solweig_light.radiation import engine as e, patch_radiation as p
    z,x,meta=inputs();ref=np.load(OUT/'oracle.npz');asvf=a.asvf_fma(x)[:,None]
    def tan32(v):
        arr=np.asarray(v)
        assert arr.dtype==np.float32
        return t.tan_array(arr.ravel()).reshape(arr.shape)
    def atan32(v):
        arr=np.asarray(v)
        assert arr.dtype==np.float32
        return t.atan_array(arr.ravel()).reshape(arr.shape)
    # Isolated function clones, no main-source change or global monkeypatch.
    en=dict(e.__dict__,tan32=tan32,atan32=atan32)
    source=inspect.getsource(e.shaded_or_sunlit).replace('np.tan(asvf)','tan32(asvf)').replace('np.arctan(tan_delta)','atan32(tan_delta)')
    exec(source,en)
    pn=dict(p.__dict__,tan32=tan32,atan32=atan32)
    source=inspect.getsource(p._classes).replace('np.tan(field)','tan32(field)').replace('np.arctan(delta)','atan32(delta)')
    exec(source,pn)
    pa,pazi=z['patch_altitude'],z['patch_azimuth'];sa,saz=np.asarray(z['solar_altitude']),np.asarray(z['solar_azimuth'])
    serial=[en['shaded_or_sunlit'](sa,saz,pa[i],pazi[i],asvf) for i in range(len(pa))]
    ss=np.column_stack([v[0][:,0] for v in serial]);sh=np.column_stack([v[1][:,0] for v in serial])
    geom=p.patch_geometry(np.column_stack((pa,pazi)));prepared=p._class_coefficients(sa,saz,geom,asvf)
    ps,ph=pn['_classes'](sa,saz,geom,asvf,0,len(x),prepared=prepared)
    report={'hsvf_bit_mismatches':int(np.count_nonzero(tan32(asvf)[:,0].view(np.uint32)!=ref['hsvf'].view(np.uint32))),
        'serial':{'sun':int(np.count_nonzero(ss!=ref['sun'])),'shade':int(np.count_nonzero(sh!=ref['shade']))},
        'prepared':{'sun':int(np.count_nonzero(ps!=ref['sun'])),'shade':int(np.count_nonzero(ph!=ref['shade']))},
        'serial_prepared_equal':bool(np.array_equal(ss,ps) and np.array_equal(sh,ph)),
        'float64_coefficients':'Unchanged NumPy scalar cos/tan; bit mismatches retained in baseline report',
        **meta}
    for label,sl in [('boundary',slice(0,meta['boundary_count'])),('frozen',slice(meta['boundary_count'],None))]:
        report[label]={'sun':int(np.count_nonzero(ss[sl]!=ref['sun'][sl])),'shade':int(np.count_nonzero(sh[sl]!=ref['shade'][sl]))}
    (OUT/'revised.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--oracle',action='store_true');parser.add_argument('--revised',action='store_true');args=parser.parse_args()
    oracle() if args.oracle else revised() if args.revised else candidate()
