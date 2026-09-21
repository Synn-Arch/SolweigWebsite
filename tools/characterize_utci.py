"""Extract UTCI expressions, generate CPU scalars, and capture isolated originals.

Reference mode must run under .venv-oracle. Generate mode never imports upstream.
No numerical or performance gate is adjusted by this tool.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
SOURCE = ROOT / '.upstream/SOLWEIG-GPU/solweig_gpu/calculate_utci.py'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def expression():
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'utci_polynomial')
    return next(n.value for n in function.body if isinstance(n, ast.Assign))


def ordered_terms(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return ordered_terms(node.left) + ordered_terms(node.right)
    return [node]


def monomial(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        a, x = monomial(node.left); b, y = monomial(node.right)
        return a*b, {key:x.get(key,0)+y.get(key,0) for key in set(x)|set(y)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        assert isinstance(node.left, ast.Name)
        return 1, {node.left.id:ast.literal_eval(node.right)}
    if isinstance(node, ast.Name): return 1, {node.id:1}
    return ast.literal_eval(node), {}


class Float32Operations(ast.NodeTransformer):
    """Round each scalar arithmetic result and each source literal to float32."""
    @staticmethod
    def cast(node):
        return ast.Call(ast.Attribute(ast.Name('np',ast.Load()),'float32',ast.Load()),[node],[])

    def visit_Constant(self, node):
        return self.cast(node) if isinstance(node.value,(int,float)) else node

    def visit_BinOp(self, node):
        # Keep integer exponents as integers, preserving upstream power operations.
        left = self.visit(node.left)
        right = node.right if isinstance(node.op, ast.Pow) else self.visit(node.right)
        return self.cast(ast.BinOp(left,node.op,right))

    def visit_UnaryOp(self, node): return self.cast(ast.UnaryOp(node.op,self.visit(node.operand)))


def generate():
    raw = expression()
    terms = []
    for index,node in enumerate(ordered_terms(raw)):
        coefficient, powers = monomial(node)
        terms.append({'index':index,'coefficient':coefficient,'powers':{key:powers.get(key,0) for key in ('Ta','va','D_Tmrt','Pa')},'expression':ast.unparse(node)})
    # Exact expression reconstruction, without rewriting any monomial.
    reconstructed = ordered_terms(raw)[0]
    for node in ordered_terms(raw)[1:]: reconstructed = ast.BinOp(reconstructed,ast.Add(),node)
    assert ast.dump(reconstructed)==ast.dump(raw)
    statements=['    result = np.float32(Ta)']
    for term in ordered_terms(raw)[1:]:
        compiled=Float32Operations().visit(ast.parse(ast.unparse(term),mode='eval').body)
        statements.append('    result = np.float32(result + '+ast.unparse(ast.fix_missing_locations(compiled))+')')
    statements.append('    return result')
    text='''"""Generated float32 UTCI scalar; regenerate with tools/characterize_utci.py.

Derived from SOLWEIG-GPU calculate_utci.py at
0d7fe742abeeddd890dd58fc76ed7f78bd47faec.
Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan.
GPL-3.0-or-later. No coefficient, term, or reduction-order rewriting.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, error_model='numpy')
def polynomial_scalar(D_Tmrt, Ta, va, Pa):
'''+ '\n'.join(statements)+'''\n

@njit(cache=True, fastmath=False, error_model='numpy')
def saturation_scalar(Ta):
    tk = np.float32(Ta + np.float32(273.15))
    g = (-2836.5744, -6028.076559, 19.54263612, -.02737830188,
         .000016261698, .00000000070229056, -.00000000000018680009, 2.7150305)
    es = np.float32(np.float32(g[7]) * np.log(tk))
    for i in range(7):
        power = np.float32(tk ** np.float32(i - 2))
        term = np.float32(np.float32(g[i]) * power)
        es = np.float32(es + term)
    es = np.float32(np.exp(es) * np.float32(.01))
    return es


@njit(cache=True, fastmath=False, error_model='numpy')
def utci_scalar(Ta, RH, Tmrt, va10m):
    if Ta <= -999 or RH <= -999 or Tmrt <= -999 or va10m <= -999:
        return np.float32(-999)
    es = saturation_scalar(Ta)
    ehPa = np.float32(np.float32(es * RH) / np.float32(100))
    Pa = np.float32(ehPa / np.float32(10))
    delta = np.float32(Tmrt - Ta)
    return polynomial_scalar(delta, Ta, va10m, Pa)
'''
    path=ROOT/'src/solweig_light/comfort/_utci_scalar.py';path.write_text(text)
    manifest={'source_commit':COMMIT,'source_sha256':sha(SOURCE),'original_expression_ast_sha256':hashlib.sha256(ast.dump(raw).encode()).hexdigest(),'term_count':len(terms),'ordered_terms':terms,'reconstruction_ast_equal':True,'scalar_sha256':sha(path),'generator_sha256':sha(Path(__file__)),'floating_policy':'float32 constants and every binary-operation result; written powers/multiplications/additions retained; fastmath=False; no algebraic monomial rewriting'}
    write_json(ROOT/'src/solweig_light/comfort/utci_coefficients.json',manifest)
    print('Generated',len(terms),'ordered terms with exact source expression reconstruction.')


def reference(directory):
    import numpy as np
    import torch
    import solweig_gpu
    from solweig_gpu.calculate_utci import utci_calculator, utci_polynomial
    source=SOURCE.parents[1]
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip()
    installed=Path(solweig_gpu.__file__).parent
    hashes={}
    for path in sorted((source/'solweig_gpu').iterdir()):
        if path.suffix in ('.py','.txt'):
            assert sha(path)==sha(installed/path.name),path
            hashes[path.name]=sha(path)
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    directory.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20260918)
    count=10000
    inputs=[rng.uniform(a,b,count).astype(np.float32) for a,b in [(-50,50),(0,100),(-50,100),(.15,17)]]
    lattice=np.array(np.meshgrid([-50,-25,0,25,50],[0,25,50,75,100],[-50,0,25,50,100],[.15,.5,1,5,10,17],indexing='ij'),dtype=np.float32).reshape(4,-1)
    specials=np.tile(np.array([[25],[50],[30],[1]],np.float32),(1,32))
    edge=[-999,-1000,np.nan,np.inf,-np.inf,np.nextafter(np.float32(-999),np.float32(np.inf)),0,-998]
    for channel in range(4):specials[channel,channel*8:(channel+1)*8]=edge
    nearwind=np.tile(np.array([[25],[50],[30],[1]],np.float32),(1,8))
    nearwind[3]=[0,np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(.15),np.float32(0)),.15,np.nextafter(np.float32(.15),np.float32(1)),.5,np.nextafter(np.float32(17),np.float32(0)),17]
    specials=np.concatenate((specials,nearwind),axis=1)
    values=np.vstack([np.concatenate((x,lattice[i],specials[i])) for i,x in enumerate(inputs)])
    # Pa/delta are original executed humidity intermediates, captured separately
    # to distinguish scalar polynomial error from humidity propagation error.
    ta,rh,tmrt,wind=[torch.from_numpy(x) for x in values]
    output=utci_calculator(ta,rh,tmrt,wind).numpy()
    invalid=(ta<=-999)|(rh<=-999)|(tmrt<=-999)|(wind<=-999)
    valid=~invalid
    tk=ta[valid]+273.15
    g=torch.tensor([-2.8365744E3,-6.028076559E3,1.954263612E1,-2.737830188E-2,1.6261698E-5,7.0229056E-10,-1.8680009E-13,2.7150305],dtype=torch.float32)
    es=g[7]*torch.log(tk)
    for i in range(7):es=es+g[i]*tk**(i-2.)
    es=torch.exp(es)*.01
    pa=np.full(output.shape,np.nan,np.float32);pa[valid.numpy()]=(es*rh[valid]/100./10.).numpy()
    path=directory/'cases.npz'
    saturation=np.full(output.shape,np.nan,np.float32); saturation[valid.numpy()]=es.numpy()
    polynomial=np.full(output.shape,-999,np.float32); polynomial[valid.numpy()]=utci_polynomial(tmrt[valid]-ta[valid],ta[valid],wind[valid],torch.from_numpy(pa)[valid]).numpy()
    np.savez_compressed(path,Ta=values[0],RH=values[1],Tmrt=values[2],wind=values[3],UTCI=output,Pa=pa,saturation=saturation,polynomial=polynomial,invalid=invalid.numpy())
    profile_hashes={}
    typed_outcomes={}
    typed_profiles=[('float64',[np.float64]*4),('mixed',[np.float64,np.float32,np.float32,np.float64])]
    typed_profiles.extend((f'mixed_{mask:04b}',[np.float64 if mask&(1<<i) else np.float32 for i in range(4)]) for mask in range(1,15) if mask!=9)
    for name,dtypes in typed_profiles:
        arrays=[values[i].astype(dtype) for i,dtype in enumerate(dtypes)]
        tensors=[torch.from_numpy(a) for a in arrays];a,h,t,v=tensors
        try:
            result=utci_calculator(a,h,t,v).numpy()
            typed_outcomes[name]={'status':'executed'}
        except Exception:
            result=None
            typed_outcomes[name]={'status':'failed_original_upstream','traceback':traceback.format_exc()}
        mask=~((a<=-999)|(h<=-999)|(t<=-999)|(v<=-999));kelvin=a[mask]+273.15
        saturation=g[7]*torch.log(kelvin)
        for i in range(7):saturation=saturation+g[i]*kelvin**(i-2.)
        saturation=torch.exp(saturation)*.01
        pressure=saturation*h[mask]/100./10.
        pv=utci_polynomial(t[mask]-a[mask],a[mask],v[mask],pressure)
        sat=np.full(a.shape,np.nan,saturation.numpy().dtype);sat[mask.numpy()]=saturation.numpy()
        pp=np.full(a.shape,np.nan,pressure.numpy().dtype);pp[mask.numpy()]=pressure.numpy()
        pol=np.full(a.shape,-999,pv.numpy().dtype);pol[mask.numpy()]=pv.numpy()
        typed=directory/(name+'.npz'); payload=dict(Ta=arrays[0],RH=arrays[1],Tmrt=arrays[2],wind=arrays[3],saturation=sat,Pa=pp,polynomial=pol,invalid=(~mask).numpy())
        if result is not None:payload['UTCI']=result
        np.savez_compressed(typed,**payload)
        profile_hashes[typed.name]=sha(typed)
    env={'source_commit':COMMIT,'evidence_class':'original_upstream_cpu','oracle_patch_hash':None,'source_package_sha256':hashes,'oracle_package':str(installed),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'torch_threads':torch.get_num_threads(),'cuda':'not_available','seed':20260918,'random_count':count,'lattice_count':lattice.shape[1],'special_count':specials.shape[1],'total_count':values.shape[1],'fixture_sha256':sha(path),'typed_fixture_sha256':profile_hashes,'typed_outcomes':typed_outcomes,'compiled_profile':'finite float32 arrays in Ta[-50,50], RH[0,100], Tmrt[-50,100], wind[0,17]; exact sentinels; other inputs retain public NumPy evaluator; no silent narrowing','out_of_range_policy':'original accepts finite out-of-range inputs; retain existing NumPy evaluation for those positions and report separately; nominal .02 C budget applies only characterized reference-valid ranges','generator_sha256':sha(Path(__file__)),'invocation':sys.argv,'ranges':{'Ta':[-50,50],'RH':[0,100],'Tmrt':[-50,100],'wind':[.15,17]},'cache_state':'new deterministic function fixture; no geometry/cache use','comparison_budget_c':.02}
    write_json(directory/'manifest.json',env)
    print('Captured',values.shape[1],'original CPU values:',directory)


def verify(directory, report):
    import contextlib
    import io
    import numpy as np
    import numba
    from osgeo import gdal
    from solweig_light.comfort.utci import (utci_calculator, utci_polynomial,
        utci_calculator_compiled, utci_calculator_uniform, _utci_points_serial, _utci_points_parallel,
        _utci_uniform_serial, _utci_uniform_parallel)
    from solweig_light.comfort._utci_scalar import polynomial_scalar, saturation_scalar
    manifest=json.loads((directory/'manifest.json').read_text())
    assert sha(directory/'cases.npz')==manifest['fixture_sha256']
    with np.load(directory/'cases.npz') as f: fixture={key:f[key] for key in f.files}
    keys=['Ta','RH','Tmrt','wind']
    inputs=[fixture[key] for key in keys]
    ta,rh,tmrt,wind=inputs
    domain=(ta>=-50)&(ta<=50)&(rh>=0)&(rh<=100)&(tmrt>=-50)&(tmrt<=100)&(wind>=0)&(wind<=17)
    def comparison(candidate, original, mask=None, budget=.02):
        special={name:bool(np.array_equal(function(candidate),function(original))) for name,function in [('nan',np.isnan),('posinf',np.isposinf),('neginf',np.isneginf)]}
        special['sentinel']=bool(np.array_equal(candidate==-999,original==-999))
        finite=np.isfinite(original)&np.isfinite(candidate)
        if mask is not None: finite &= mask
        errors=np.where(finite,np.abs(candidate-original),np.nan)
        coordinate=tuple(int(v) for v in np.unravel_index(np.nanargmax(errors),candidate.shape))
        maximum=float(np.nanmax(errors))
        return {'status':'passed' if all(special.values()) and maximum<=budget else 'failed','max_abs_c':maximum,'worst_coordinate':list(coordinate),'candidate_at_worst':float(candidate[coordinate]),'original_at_worst':float(original[coordinate]),'special_masks_exact':special,'budget_c':budget,'finite_count':int(finite.sum())}
    with np.errstate(all='ignore'):
        serial=utci_calculator_compiled(*inputs)
        numpy=utci_calculator(*inputs)
    nominal=comparison(serial,fixture['UTCI'],domain)
    nominal['worst_inputs']={key:float(fixture[key][tuple(nominal['worst_coordinate'])]) for key in keys}
    numpy_nominal=comparison(numpy,fixture['UTCI'],domain)
    direct=np.array([polynomial_scalar(np.float32(t-a),a,w,p) for a,t,w,p in zip(ta[domain],tmrt[domain],wind[domain],fixture['Pa'][domain])],np.float32)
    polynomial=comparison(direct,fixture['polynomial'][domain])
    saturation=np.array([saturation_scalar(t) for t in ta[domain]],np.float32)
    errors=np.abs(saturation-fixture['saturation'][domain])
    relative=errors/fixture['saturation'][domain]
    saturation_result={'evidence':'diagnostic; full UTCI .02 C budget remains authoritative','max_abs_hpa':float(errors.max()),'max_relative':float(relative.max()),'worst_original_index':int(np.flatnonzero(domain)[errors.argmax()])}
    thread_results={}
    previous=numba.get_num_threads()
    try:
        for count in [1,4,10]:
            numba.set_num_threads(count)
            with np.errstate(all='ignore'):parallel=utci_calculator_compiled(*inputs,parallel=True)
            thread_results[str(count)]={'bitwise_equal_serial':bool(np.array_equal(parallel,serial,equal_nan=True)),**comparison(parallel,fixture['UTCI'],domain)}
    finally:numba.set_num_threads(previous)
    typed={}
    for name,outcome in manifest['typed_outcomes'].items():
        with np.load(directory/(name+'.npz')) as f:
            arrays=[f[key] for key in keys]
            try:utci_calculator_compiled(*arrays)
            except RuntimeError as error:matched=str(error) in outcome['traceback']
            else:matched=False
            valid=(arrays[0]>=-50)&(arrays[0]<=50)&(arrays[1]>=0)&(arrays[1]<=100)&(arrays[2]>=-50)&(arrays[2]<=100)&(arrays[3]>=0)&(arrays[3]<=17)
            poly=utci_polynomial(arrays[2][valid]-arrays[0][valid],arrays[0][valid],arrays[3][valid],f['Pa'][valid])
            typed[name]={'original_calculator_status':outcome['status'],'candidate_reproduces_original_exception':matched,'direct_polynomial':comparison(poly,f['polynomial'][valid])}
    gdal.UseExceptions()
    def raster(path):
        ds=gdal.Open(str(path));a=ds.ReadAsArray();ds=None;return a
    cases={}
    uniform_cases={}
    for name in ['base','uhi','landcover','landcover_normalized','directional','legacy_direct','legacy_wrapper']:
        base=ROOT/'tests/reference/optional_original_cpu'/name;out=base/'output_folder/0_0'
        forcing=np.loadtxt(base/'processed_inputs/metfiles/metfile_0_0_2020-07-18.txt',skiprows=1)
        temperature=raster(out/'Ta_0_0.tif');radiant=raster(out/'TMRT_0_0.tif');speed=raster(out/'Wind_0_0.tif')
        humidity=np.broadcast_to(forcing[:,10].astype(np.float32)[:,None,None],temperature.shape)
        original=raster(out/'UTCI_0_0.tif')
        candidate=utci_calculator_compiled(temperature,humidity,radiant,speed)
        buildings=raster(base/'processed_inputs/Building_DSM/Building_DSM_0_0.tif')-raster(base/'processed_inputs/DEM/DEM_0_0.tif')
        buildings[buildings<2]=1;buildings[buildings>=2]=0;candidate[:,buildings!=1]=np.nan
        cases[name]=comparison(candidate,original)
        uniform=np.empty_like(radiant)
        uniform_parallel=np.empty_like(radiant)
        for i in range(len(radiant)):
            uniform[i]=utci_calculator_uniform(np.float32(temperature[i,0,0]),np.float32(forcing[i,10]),radiant[i],speed[i])
            uniform_parallel[i]=utci_calculator_uniform(np.float32(temperature[i,0,0]),np.float32(forcing[i,10]),radiant[i],speed[i],parallel=True)
        equivalent=bool(np.array_equal(uniform_parallel,uniform,equal_nan=True))
        variable=utci_calculator_compiled(temperature,humidity,radiant,speed)
        uniform_equal_variable=bool(np.array_equal(uniform,variable,equal_nan=True))
        uniform[:,buildings!=1]=np.nan
        uniform_cases[name]={'bitwise_equal_variable_loop':uniform_equal_variable,'parallel_equal_serial':equivalent,**comparison(uniform,original)}
    diagnostics=report.parent/'characterization/utci_compiled'
    diagnostics.mkdir(parents=True,exist_ok=True)
    for name,kernel in [('serial',_utci_points_serial),('parallel',_utci_points_parallel),('uniform_serial',_utci_uniform_serial),('uniform_parallel',_utci_uniform_parallel)]:
        with (diagnostics/(name+'_types.txt')).open('w') as stream:kernel.inspect_types(file=stream)
    _utci_points_parallel.recompile()
    with (diagnostics/'parallel_diagnostics.txt').open('w') as stream,contextlib.redirect_stdout(stream):_utci_points_parallel.parallel_diagnostics(level=4)
    _utci_uniform_parallel.recompile()
    with (diagnostics/'uniform_parallel_diagnostics.txt').open('w') as stream,contextlib.redirect_stdout(stream):_utci_uniform_parallel.parallel_diagnostics(level=4)
    outside=(~domain)&(~fixture['invalid'])
    outside_result=comparison(serial,fixture['UTCI'],outside)
    outside_result['scope']='exploratory accepted values outside finite characterized profile; unchanged NumPy fallback, not nominal-domain gate; failures remain visible'
    result={'source_commit':COMMIT,'evidence_class':'candidate_compiled_vs_original_upstream_cpu','oracle_patch_hash':None,'reference_manifest':str(directory/'manifest.json'),'reference_fixture_sha256':manifest['fixture_sha256'],'invocation':sys.argv,'candidate_sources':{str(path.relative_to(ROOT)):sha(path) for path in (ROOT/'src/solweig_light/comfort/utci.py',ROOT/'src/solweig_light/comfort/_utci_scalar.py',ROOT/'src/solweig_light/comfort/utci_coefficients.json',ROOT/'src/solweig_light/pipeline.py')},'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},'compiled_domain':manifest['compiled_profile'],'nominal_serial':nominal,'nominal_numpy':numpy_nominal,'scalar_direct_polynomial':polynomial,'scalar_saturation':saturation_result,'parallel_threads':thread_results,'typed_profiles':typed,'packaged_rasters':cases,'uniform_packaged_rasters':uniform_cases,'uniform_policy':'one immutable float32 vapour pressure per timestep; identical scalar polynomial and validity; Python boxing boundary explicitly restores float32 pressure; no forcing-array allocation in compiled profile','outside_profile_exploratory':outside_result,'exploratory_failures':[{'check':'direct NumPy mixed-type polynomial with initially invented rtol1e-9/atol1e-6','status':'failed','interpretation':'Unapproved exploratory threshold had no protocol/accuracy justification. Lead instructed use existing frozen .02 C UTCI budget, which passes. Mixed-power implementation differences remain reported; expression checksum proves literals/order reconstruction, not evaluation identity.'}],'numba':{'serial_nopython_signatures':[str(x) for x in _utci_points_serial.nopython_signatures],'parallel_nopython_signatures':[str(x) for x in _utci_points_parallel.nopython_signatures],'fastmath':False,'diagnostics_directory':str(diagnostics)},'timings':'not measured; root sequences frozen kernel benchmarks separately','pipeline_default':'serial compiled uniform forcing with full NumPy fallback outside characterized profile'}
    write_json(report,result)
    print(json.dumps({'nominal_serial':nominal,'scalar_direct_polynomial':polynomial,'scalar_saturation':saturation_result,'outside_profile_exploratory':outside_result,'report':str(report)},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/utci_compiled_characterization.json')
    args=parser.parse_args()
    if args.generate:generate()
    if args.reference:reference(args.reference.resolve())
    if args.verify:verify(args.verify.resolve(),args.report.resolve())
    if not args.generate and not args.reference and not args.verify:parser.error('Specify --generate, --reference or --verify')

if __name__=='__main__':main()
