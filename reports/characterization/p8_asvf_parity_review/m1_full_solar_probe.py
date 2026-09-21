"""24 frozen solar timesteps, actual upstream per-patch shape oracle."""
import json,sys,inspect
from pathlib import Path
import numpy as np
from m1_classifier_probe import inputs, ROOT

OUT=Path(__file__).parent/'m1_full_solar'
OUT.mkdir(exist_ok=True)
B=ROOT/'tests/reference/small_original_cpu/boundaries'
events=[e for e in json.loads((B/'manifest.json').read_text())['events'] if e['boundary']=='input']
z,x,meta=inputs()

if '--oracle' in sys.argv:
    import torch
    from solweig_gpu import solweig
    torch.set_num_threads(1)
    field=torch.acos(torch.sqrt(torch.from_numpy(x.copy())))[:,None]
    pa=torch.from_numpy(z['patch_altitude'].copy());pazi=torch.from_numpy(z['patch_azimuth'].copy())
    for ev in events:
        data=np.load(B/ev['path']);sa=torch.tensor(data['altitude']);saz=torch.tensor(data['azimuth'])
        result=[solweig.shaded_or_sunlit(sa,saz,pa[p],pazi[p],field) for p in range(153)]
        np.savez_compressed(OUT/f"oracle_{ev['timestep']:02d}.npz",sun=np.column_stack([a.numpy()[:,0] for a,b in result]),shade=np.column_stack([b.numpy()[:,0] for a,b in result]))
    (OUT/'oracle_metadata.json').write_text(json.dumps({'timesteps':24,**meta},indent=2)+'\n')
else:
    from solweig_light.radiation import engine as e,patch_radiation as p
    import sleef_acos_prototype as a, sleef_classifier_prototype as t
    field=a.asvf_fma(x)[:,None]
    def tan32(v):
        v=np.asarray(v);return t.tan_array(v.ravel()).reshape(v.shape)
    def atan32(v):
        v=np.asarray(v);return t.atan_array(v.ravel()).reshape(v.shape)
    en=dict(e.__dict__,tan32=tan32,atan32=atan32)
    exec(inspect.getsource(e.shaded_or_sunlit).replace('np.tan(asvf)','tan32(asvf)').replace('np.arctan(tan_delta)','atan32(tan_delta)'),en)
    pn=dict(p.__dict__,tan32=tan32,atan32=atan32)
    exec(inspect.getsource(p._classes).replace('np.tan(field)','tan32(field)').replace('np.arctan(delta)','atan32(delta)'),pn)
    if '--actual-package' in sys.argv:
        import solweig_light, importlib.util
        from solweig_light.radiation._math_profile import asvf
        assert importlib.util.find_spec('torch') is None
        field=asvf(x)[:,None]
        en=e.__dict__;pn=p.__dict__
    geom=p.patch_geometry(np.column_stack((z['patch_altitude'],z['patch_azimuth'])))
    records=[]
    for ev in events:
        data=np.load(B/ev['path']);sa=np.asarray(data['altitude']);saz=np.asarray(data['azimuth']);ref=np.load(OUT/f"oracle_{ev['timestep']:02d}.npz")
        serial=[en['shaded_or_sunlit'](sa,saz,geom.altitude[j],geom.azimuth[j],field) for j in range(153)]
        ss=np.column_stack([u[:,0] for u,v in serial]);sh=np.column_stack([v[:,0] for u,v in serial])
        record={'step':ev['timestep'],'serial_sun':int(np.count_nonzero(ss!=ref['sun'])),'serial_shade':int(np.count_nonzero(sh!=ref['shade'])),'variants':[]}
        for stride in (1,2):
            values=field[::stride]
            prep=p._class_coefficients(sa,saz,geom,values)
            for block in (17,128,len(values)):
                cs=[];ch=[]
                for start in range(0,len(values),block):
                    u,v=pn['_classes'](sa,saz,geom,values,start,min(start+block,len(values)),prepared=prep);cs.append(u);ch.append(v)
                record['variants'].append({'stride':stride,'block':block,'sun':int(np.count_nonzero(np.concatenate(cs)!=ref['sun'][::stride])),'shade':int(np.count_nonzero(np.concatenate(ch)!=ref['shade'][::stride]))})
        records.append(record)
    report={'passed':all(not r['serial_sun'] and not r['serial_shade'] and all(not v['sun'] and not v['shade'] for v in r['variants']) for r in records),'records':records,'timesteps':24,'input_count':len(x),'patches':153}
    label=sys.argv[sys.argv.index('--label')+1] if '--label' in sys.argv else ''
    if '--actual-package' in sys.argv:report['package']=solweig_light.__file__
    (OUT/('result_'+label+'.json' if label else 'result.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'failing_steps':[r for r in records if r['serial_sun'] or r['serial_shade']]},indent=2))
