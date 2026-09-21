"""Run unmodified, hash-verified upstream optional local fixtures (not benchmarks)."""
from pathlib import Path
import hashlib, json, shutil, subprocess, traceback, contextlib, sys, importlib.metadata, platform
import numpy as np
import torch
from osgeo import gdal
import solweig_gpu
from solweig_gpu.utci_process import compute_utci, nearest_wind_dir_30, map_windcoeff_files_by_key

ROOT=Path(__file__).resolve().parents[1]
COMMIT='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
BASE=ROOT/'reports/runs/optional_local'
REF=ROOT/'tests/reference/optional_original_cpu'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def read(p):
    ds=gdal.Open(str(p)); a=ds.ReadAsArray(); ds=None; return a

def main():
    source=ROOT/'.upstream/SOLWEIG-GPU'; package=Path(solweig_gpu.__file__).parent
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip()
    hashes={}
    for p in sorted((source/'solweig_gpu').iterdir()):
        if p.suffix in ('.py','.txt'):
            assert digest(p)==digest(package/p.name),p.name
            hashes[p.name]=digest(p)
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    BASE.mkdir(parents=True,exist_ok=False)
    original=ROOT/'tests/reference/small_original_cpu/scene'
    geometry=original/'processed_inputs'
    if not geometry.exists(): geometry=ROOT/'reports/runs/boundaries_cpu_provenance/scene/processed_inputs'
    template=geometry/'Building_DSM/Building_DSM_0_0.tif'
    ds=gdal.Open(str(template)); shape=(ds.RasterYSize,ds.RasterXSize); ds=None
    def raster(p,a):
        p.parent.mkdir(parents=True,exist_ok=True)
        ds=gdal.GetDriverByName('GTiff').CreateCopy(str(p),gdal.Open(str(template)))
        ds.GetRasterBand(1).WriteArray(np.asarray(a,dtype=np.float32)); ds=None
    met=np.loadtxt(original/'met.txt',skiprows=1)
    dirs=np.array([0,14.999,15,44.999,45,74.999,75,105,135,165,195,225,255,285,315,344.999,345,359,360,375,-1,-999,np.nan,np.inf])
    met[:,23]=dirs
    coefficients={d:(i-1)*0.1 for i,d in enumerate(range(0,360,30))}
    cases={}; arrays={}
    for name in ['base','uhi','landcover','landcover_normalized','directional','legacy_direct','legacy_wrapper']:
        scene=BASE/name/'scene'; shutil.copytree(original,scene)
        if (scene/'output_folder').exists(): shutil.rmtree(scene/'output_folder')
        pre=scene/'processed_inputs'
        if pre.exists(): shutil.rmtree(pre)
        shutil.copytree(geometry,pre,ignore=shutil.ignore_patterns('output_folder'))
        for p in ['Landcover','WindCoeff']:
            if (pre/p).exists(): shutil.rmtree(pre/p)
        forcing=np.column_stack((met,np.full(len(met),2.5 if name=='uhi' else 0.)))
        header=(original/'met.txt').read_text().splitlines()[0]+' UHI'
        mf=pre/'metfiles/metfile_0_0_2020-07-18.txt'; np.savetxt(mf,forcing,header=header,comments='',fmt='%.8f')
        lc=None; wind=None
        if name.startswith('landcover'):
            lc=pre/'Landcover/Landcover_0_0.tif'
            grid=np.resize(np.arange(9),shape)
            if name=='landcover_normalized':
                grid=np.where((grid<1)|(grid>7),6,grid)
                grid=np.where((grid==3)|(grid==4),5,grid)
            raster(lc,grid)
        if name=='directional':
            wind={}
            for d,c in coefficients.items():
                p=pre/f'WindCoeff/WindCoeff_dir{d:03d}_0_0.tif'; raster(p,np.full(shape,c)); wind[d]=str(p)
        if name.startswith('legacy'):
            p=pre/'WindCoeff/WindCoeff_0_0.tif'; raster(p,np.full(shape,0.25)); wind=str(p)
        output=scene/'output_folder/0_0'; output.mkdir(parents=True)
        flags={f'save_{f}':True for f in ['tmrt','kup','kdown','lup','ldown','shadow','wbgt','ta','wind']}
        invocation={'entrypoint':'compute_utci' if name=='legacy_direct' else 'run_utci_tiles','base_path':str(scene),'preprocess_dir':str(pre),'selected_date_str':'2020-07-18',**flags}
        inputs={str(p.relative_to(scene)):digest(p) for p in sorted(scene.rglob('*')) if p.is_file() and 'output_folder' not in p.parts}
        result={'invocation':invocation,'input_sha256':inputs,'wind_discovery':map_windcoeff_files_by_key(str(pre/'WindCoeff')) if (pre/'WindCoeff').exists() else {}}
        with (BASE/name/'stdout.log').open('w') as out,(BASE/name/'stderr.log').open('w') as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:
                if name=='legacy_direct':
                    compute_utci(*[str(pre/p) for p in ['Building_DSM/Building_DSM_0_0.tif','Trees/Trees_0_0.tif','DEM/DEM_0_0.tif','walls/walls_0_0.tif','aspect/aspect_0_0.tif']],str(lc) if lc else None,wind,forcing,str(output),'0_0','2020-07-18',**flags)
                else:
                    solweig_gpu.run_utci_tiles(str(scene),str(pre),'2020-07-18',tile_keys=['0_0'],**flags)
                result['status']='executed'
                arrays[name]={p.stem.split('_')[0]:read(p) for p in output.glob('*.tif')}
            except BaseException:
                result.update(status='failed',traceback=traceback.format_exc())
        result['output_metadata']={}
        for p in output.glob('*.tif'):
            ds=gdal.Open(str(p)); result['output_metadata'][p.name]={'shape':[ds.RasterCount,ds.RasterYSize,ds.RasterXSize],'geotransform':list(ds.GetGeoTransform()),'projection':ds.GetProjection(),'bands':[{'metadata':ds.GetRasterBand(i).GetMetadata(),'nodata':ds.GetRasterBand(i).GetNoDataValue(),'dtype':gdal.GetDataTypeName(ds.GetRasterBand(i).DataType)} for i in range(1,ds.RasterCount+1)]}; ds=None
        result['output_sha256']={str(p.relative_to(scene)):digest(p) for p in output.glob('*.tif')}
        dump(BASE/name/'outcome.json',result); cases[name]=result
    checks={}
    if 'base' in arrays and 'uhi' in arrays:
        checks['uhi_ta_delta_2_5_exact']=bool(np.array_equal(arrays['uhi']['Ta']-arrays['base']['Ta'],np.full_like(arrays['base']['Ta'],2.5)))
        checks['uhi_ta_matches_float32_addition']=bool(np.array_equal(arrays['uhi']['Ta'],arrays['base']['Ta']+np.float32(2.5)))
        checks['uhi_ta_delta_max_abs_error_c']=float(np.max(np.abs(arrays['uhi']['Ta']-arrays['base']['Ta']-2.5)))
        checks['uhi_radiation_unchanged']={k:bool(np.array_equal(arrays['uhi'][k],arrays['base'][k],equal_nan=True)) for k in ['TMRT','Kup','Kdown','Lup','Ldown','Shadow']}
        checks['uhi_utci_changed']=not np.array_equal(arrays['uhi']['UTCI'],arrays['base']['UTCI'],equal_nan=True)
    for name in ['directional','legacy_direct','legacy_wrapper','base']:
        if name not in arrays: continue
        cs=[coefficients[nearest_wind_dir_30(d)] if nearest_wind_dir_30(d) is not None else 1 for d in dirs] if name=='directional' else [0.25 if name=='legacy_direct' else 1]*len(met)
        expected=np.maximum(np.asarray(cs,dtype=np.float32)*met[:,9].astype(np.float32),np.float32(.15))
        observed=arrays[name]['Wind']
        checks[name+'_wind_exact']=bool(np.array_equal(observed,np.broadcast_to(expected[:,None,None],observed.shape)))
        checks[name+'_wind_per_band']=[float(v) for v in observed[:,0,0]]
    if 'legacy_wrapper' in arrays and 'base' in arrays:
        checks['legacy_wrapper_ignored_equals_base']={k:bool(np.array_equal(arrays['legacy_wrapper'][k],v,equal_nan=True)) for k,v in arrays['base'].items()}
    if 'landcover' in arrays and 'landcover_normalized' in arrays:
        checks['landcover_normalization_exact_outputs']={k:bool(np.array_equal(arrays['landcover'][k],v,equal_nan=True)) for k,v in arrays['landcover_normalized'].items()}
    report={'evidence_class':'original_upstream_cpu','source_commit':COMMIT,'oracle_patch_hash':None,'source_sha256':hashes,'package_path':str(package),'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gdal_version':gdal.VersionInfo(),'cuda':'not available'},'torch_threads':1,'harness_sha256':digest(Path(__file__)),'invocation':[sys.executable,str(Path(__file__).resolve())],'measurement_class':'behavior characterization; no timing/performance claim','direction_inputs':dirs.tolist(),'direction_coefficients':coefficients,'landcover_input_values':list(range(9)),'cases':cases,'checks':checks}
    dump(ROOT/'reports/optional_local_characterization.json',report)
    REF.mkdir(parents=True,exist_ok=False)
    for name,r in cases.items():
        target=REF/name; target.mkdir(); shutil.copyfile(BASE/name/'outcome.json',target/'outcome.json')
        shutil.copytree(BASE/name/'scene/processed_inputs',target/'processed_inputs')
        if r['status']=='executed': shutil.copytree(BASE/name/'scene/output_folder',target/'output_folder')
    shutil.copyfile(ROOT/'reports/optional_local_characterization.json',REF/'manifest.json')
    print(json.dumps({'status':{k:v['status'] for k,v in cases.items()},'checks':checks},indent=2))
if __name__=='__main__': main()
