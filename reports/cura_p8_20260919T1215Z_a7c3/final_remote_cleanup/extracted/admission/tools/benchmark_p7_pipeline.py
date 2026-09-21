#!/usr/bin/env python3
"""Explicit-protocol P6-versus-candidate paired full-workload instrument."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback

LIMIT = 12 * 1024**3
FLAGS = ('tmrt', 'svf', 'kup', 'kdown', 'lup', 'ldown', 'shadow', 'wbgt', 'ta', 'wind')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMBA_NUM_THREADS')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc', '.nbi', '.pyc')}

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')

def expand(value, scene):
    if isinstance(value, str): return value.replace('{scene}', str(scene))
    if isinstance(value, dict): return {k: expand(v, scene) for k, v in value.items()}
    if isinstance(value, list): return [expand(v, scene) for v in value]
    return value

def validate_baseline(source):
    accepted = Path(__file__).resolve().parents[1] / 'reports/characterization/p7_p6_baseline'
    manifest_path = accepted / 'manifest.json'
    verification_path = accepted / 'p6_installed_verification.json'
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(verification_path.read_text())
    for relative, expected in manifest['files'].items():
        if digest(accepted / relative) != expected: raise ValueError(f'Accepted P6 snapshot integrity failure: {relative}')
    expected = {k.removeprefix('src/'):v for k,v in manifest['files'].items() if k.startswith('src/')}
    verified = {k.removeprefix('src/'):v for k,v in verification['source_sha256'].items() if k.startswith('src/')}
    if any(verified.get(k) != v for k,v in expected.items() if k.endswith('.py')) or any(expected.get(k) != v for k,v in verified.items()) or hashes(source) != expected:
        raise ValueError('Supplied baseline differs from accepted installed P6 source')
    return dict(passed=True, accepted_manifest=str(manifest_path), accepted_manifest_sha256=digest(manifest_path), installed_verification=str(verification_path), installed_verification_sha256=digest(verification_path), source_hashes=expected)

def validate_correctness_report(path, expected_hash, baseline_hashes, candidate_hashes):
    if digest(path) != expected_hash: raise ValueError('Correctness report hash mismatch')
    report=json.loads(Path(path).read_text())
    if report.get('status')!='passed': raise ValueError('Correctness report not passed')
    if report.get('source_hashes') != {'baseline':baseline_hashes,'candidate':candidate_hashes}: raise ValueError('Correctness report source hashes mismatch')
    ids=report.get('fixture_case_ids',[])
    if len(ids)!=7 or len(set(ids))!=7: raise ValueError('Correctness report requires seven unique fixtures')
    def evidence(item):
        ref=item.get('evidence',{})
        if item.get('passed') is not True or not ref.get('path') or digest(ref['path'])!=ref.get('sha256'): raise ValueError('Invalid correctness evidence reference')
    cases=report.get('cases',[])
    if len(cases)!=14 or {(z.get('case_id'),z.get('variant')) for z in cases}!={(case,variant) for case in ids for variant in ('baseline','candidate')}: raise ValueError('Correctness report requires fourteen variant cases')
    for item in cases:
        evidence(item)
        geometry=item.get('geometry_evidence',{})
        if not geometry.get('path') or digest(geometry['path'])!=geometry.get('sha256'):
            raise ValueError('Missing or changed geometry evidence')
        result=json.loads(Path(geometry['path']).read_text())
        records=result.get('records',[])
        if result.get('status')!='passed' or len(records)!=18 or len({(z.get('artifact'),z.get('field')) for z in records})!=18:
            raise ValueError('Geometry evidence requires eighteen distinct passed fields')
    for variant in ('baseline','candidate'):
        original=report.get('original_regression',{}).get(variant,{})
        angular=report.get('angular_ground_view',{}).get(variant,{})
        if original.get('cases')!=9 or angular.get('field_count')!=17 or angular.get('boundary_passed') is not True: raise ValueError('Original/angular correctness coverage missing')
        evidence(original); evidence(angular)
    return dict(path=str(path),sha256=expected_hash,fixture_case_ids=ids,status='validated_external_gate',evidence=report)

def validate_repaired_baseline(source, protocol, candidate_source):
    import tempfile
    parent=Path(__file__).resolve().parents[1]/'reports/characterization/p7_p6_baseline'
    parent_validation=validate_baseline(parent/'src')
    manifest_path=Path(protocol['baseline_manifest'])
    if digest(manifest_path)!=protocol['baseline_manifest_sha256']: raise ValueError('Repaired baseline manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest['parent_manifest_sha256']!=digest(parent/'manifest.json') or Path(manifest['parent_snapshot']).resolve()!=parent.resolve(): raise ValueError('Repaired baseline parent mismatch')
    if protocol['baseline_label']!=manifest['label']: raise ValueError('Repaired baseline label mismatch')
    policy=manifest.get('repair_policy','angular_v1')
    if protocol.get('repair_policy','angular_v1')!=policy:
        raise ValueError('Repair policy mismatch')
    allowed={'src/solweig_light/radiation/engine.py','src/solweig_light/radiation/ground_view.py'}
    if policy=='angular_and_svf_weights_v1':
        allowed.add('src/solweig_light/geometry/svf.py')
    elif policy!='angular_v1':
        raise ValueError('Unknown repair policy')
    parent_files=json.loads((parent/'manifest.json').read_text())['files']
    expected_parent={k:v for k,v in parent_files.items() if k.startswith('src/')}
    repaired=manifest['files']
    if repaired.keys()!=expected_parent.keys() or {k for k in repaired if repaired[k]!=expected_parent[k]}!=allowed or set(manifest['changed_files'])!=allowed: raise ValueError('Unallowed repaired baseline changed files')
    for name in allowed:
        if manifest['changed_files'][name]!={'before_sha256':expected_parent[name],'after_sha256':repaired[name]}: raise ValueError('Repaired file derivation hashes mismatch')
    expected={k.removeprefix('src/'):v for k,v in repaired.items()}
    if hashes(source)!=expected: raise ValueError('Repaired baseline source mismatch')
    patch=manifest_path.parent/('compatibility_repair.patch' if policy=='angular_and_svf_weights_v1' else 'angular_compatibility_repair.patch')
    if not protocol.get('repair_patch_sha256') or digest(patch)!=protocol['repair_patch_sha256'] or manifest['repair_patch_sha256']!=protocol['repair_patch_sha256']: raise ValueError('Missing or mismatched reviewed repair patch hash')
    # Apply the complete protocol-bound patch to parent copies, then compare the
    # entire resulting source inventory. File-name allowlisting alone is insufficient.
    with tempfile.TemporaryDirectory(prefix='p7-repair-validation-') as tmp:
        work=Path(tmp); shutil.copytree(parent/'src',work/'src')
        applied=subprocess.run(['patch','--batch','-p1','-i',str(patch.resolve())],cwd=work,capture_output=True,text=True)
        if applied.returncode or hashes(work/'src')!=expected: raise ValueError('Reviewed repair patch does not derive supplied baseline')
    numerical=validate_correctness_report(protocol['correctness_report'],protocol['correctness_report_sha256'],expected,hashes(candidate_source))
    if policy=='angular_and_svf_weights_v1':
        for variant in ('baseline','candidate'):
            weights=numerical['evidence'].get('svf_weights',{}).get(variant,{})
            ref=weights.get('evidence',{})
            if (weights.get('passed') is not True or weights.get('weight_count')!=180
                    or weights.get('boundary_pixel_count')!=2
                    or weights.get('maximum_weight_ulp')!=1
                    or not ref.get('path') or digest(ref['path'])!=ref.get('sha256')):
                raise ValueError('SVF weight repair correctness coverage missing')
    return dict(passed=True,label=protocol['baseline_label'],source_hashes=expected,accepted_parent_validation=parent_validation,parent_manifest_sha256=digest(parent/'manifest.json'),baseline_manifest=str(manifest_path),baseline_manifest_sha256=digest(manifest_path),full_repair_diff_sha256=digest(patch),numerical_gate=numerical)

def validate_protocol_baseline(source, protocol, candidate_source):
    if 'baseline_manifest' not in protocol: return validate_baseline(source)
    accepted=Path(__file__).resolve().parents[1]/'reports/characterization/p7_p6_baseline/manifest.json'
    if Path(protocol['baseline_manifest']).resolve()==accepted.resolve():
        if digest(accepted)!=protocol['baseline_manifest_sha256']: raise ValueError('Accepted baseline protocol manifest mismatch')
        if protocol['baseline_label']!=json.loads(accepted.read_text())['label']: raise ValueError('Accepted baseline label mismatch')
        result=validate_baseline(source); result['label']=protocol['baseline_label']; return result
    return validate_repaired_baseline(source,protocol,candidate_source)

def hardware_inventory():
    import psutil
    model = platform.processor()
    physical_ram = psutil.virtual_memory().total
    if sys.platform == 'darwin':
        model = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
        physical_ram = int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    return dict(platform=platform.platform(),cpu_model=model,logical_cpus=os.cpu_count(),physical_ram_bytes=physical_ram,affinity_policy='No affinity pinning; inherited scheduler policy',frozen_host_matches=(model=='Apple M1 Pro' and os.cpu_count()==10 and physical_ram==17179869184))

def cache_snapshot(root):
    manifests={}
    for path in sorted(root.rglob('manifest.json')) if root.exists() else []:
        payload=json.loads(path.read_text())
        identity=payload.get('identity',{})
        if 'construction' not in identity and identity.get('policy')=='legacy-logical-domain-geometry-v1':
            manifests[str(path.relative_to(root))]=dict(sha256=digest(path),identity=identity,key=payload.get('key'))
    generations=sorted(str(p.relative_to(root)) for p in root.rglob('generation-*') if p.is_dir()) if root.exists() else []
    return dict(pipeline_manifests=manifests,generations=generations)

def cache_reuse_proof(before, after, expected_keys):
    observed={item['key'] for item in before['pipeline_manifests'].values()}
    passed=bool(expected_keys) and observed==set(expected_keys) and before==after
    return dict(passed=passed,expected_pipeline_keys=expected_keys,before=before,after=after,proof='Backend integrity/identity preflight plus unchanged pipeline manifest hashes and UUID generation set; recompute/cache-enabled pipeline calls store, whose misses create new generations')

def apply_global_eligibility(summaries, frozen_guard, pairs):
    eligible=bool(frozen_guard and pairs and all(z['comparison']['passed'] and z.get('cache_conditions_passed',False) for z in pairs) and all(z['valid_pairs']==5 for z in summaries))
    for summary in summaries:
        summary['subgroup_comparison_passed']=summary['valid_pairs']==5
        summary['performance_claim_eligible']=eligible
    return eligible

def cache_preflight(spec, run):
    from solweig_light.identities import geometry_identity
    from solweig_light.pipeline import files_by_key
    from solweig_light.cache import GeometryStore
    prepared=Path(spec['kwargs']['preprocess_dir'])
    maps={name:files_by_key(prepared/name) for name in ('Building_DSM','Trees','DEM','metfiles','walls','aspect')}
    keys=set.intersection(*(set(mapping) for mapping in maps.values()))
    if spec['kwargs'].get('tile_keys') is not None: keys &= set(spec['kwargs']['tile_keys'])
    if not keys: raise ValueError('No expected pipeline cache tiles')
    store=GeometryStore(spec['runtime']['cache_dir'])
    records=[]
    for tile in sorted(keys):
        identity=geometry_identity({name:mapping[tile] for name,mapping in maps.items()},2)
        key=store.key_for(identity)
        # Integrity validation only; _open cannot create or repair an entry.
        with store._open(store.root/key,identity,key,True): pass
        records.append(dict(tile=tile,key=key,identity=identity))
    dump(run/'cache_preflight.json',dict(passed=True,records=records))

def child(spec_path):
    spec = json.loads(spec_path.read_text())
    run = spec_path.parent
    try:
        import solweig_light
        source = Path(spec['source']).resolve()
        package = Path(solweig_light.__file__).resolve()
        if not package.is_relative_to(source): raise RuntimeError(f'Wrong backend imported: {package}')
        dump(run / 'environment.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(), package=str(package), dependencies={d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, thread_environment={k: os.environ.get(k) for k in THREAD_ENV}, invocation=spec))
        if spec['entrypoint']=='__cache_preflight':
            cache_preflight(spec,run)
            dump(run/'outcome.json',{'status':'cache_integrity_validated'})
            return
        runtime = dict(spec['runtime'], cpu_budget=spec['budget'], workers=1, threads_per_worker=spec['budget'], memory_budget_bytes=LIMIT, resume=False)
        with solweig_light.runtime_options(solweig_light.RuntimeOptions(**runtime)):
            getattr(solweig_light, spec['entrypoint'])(**spec['kwargs'])
        dump(run / 'outcome.json', {'status': 'executed_not_yet_compared'})
    except BaseException:
        dump(run / 'outcome.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise

def measure(spec, run, python):
    import psutil
    run.mkdir(parents=True, exist_ok=False)
    dump(run / 'spec.json', spec)
    env = dict(os.environ, PYTHONPATH=spec['source'], PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='', NUMBA_CACHE_DIR=str(run / 'jit'))
    for key in THREAD_ENV: env[key] = str(spec['budget'])
    command = [python, str(Path(__file__).resolve()), '--child', str(run / 'spec.json')]
    samples, peak, aborted = [], 0, False
    start = time.perf_counter()
    with (run / 'stdout.log').open('w') as out, (run / 'stderr.log').open('w') as err:
        proc = subprocess.Popen(command, cwd=run, env=env, stdout=out, stderr=err, start_new_session=True)
        root = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                try: members = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess: members = []
                rss = 0
                for member in members:
                    try: rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                peak = max(peak, rss)
                samples.append([time.perf_counter() - start, rss])
                if rss > LIMIT:
                    aborted = True
                    break
                time.sleep(.02)
        finally:
            if aborted or proc.poll() is None:
                import signal
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            code = proc.wait()
    result = dict(command=command, exit_code=code, elapsed_seconds_including_imports_jit_io=time.perf_counter()-start, peak_process_tree_rss_bytes=peak, memory_limit_aborted=aborted, sample_interval_seconds=.02, jit_state='fresh private NUMBA_CACHE_DIR; uncached kernels compile each process; OS page cache uncontrolled', rss_caveat='shared pages may count twice; between-sample peaks may be missed')
    dump(run / 'rss_samples.json', samples)
    dump(run / 'measurement.json', result)
    return result

def nodata_equal(left, right):
    """Nodata identity, with paired NaNs equal and None distinct."""
    import math
    if left is None or right is None: return left is right
    return left == right or (math.isnan(left) and math.isnan(right))

def compare(left, right, rules, patterns):
    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()
    def inventory(root): return {str(p.relative_to(root)): p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    a, b = inventory(left), inventory(right)
    records = []
    passed = bool(a) and a.keys() == b.keys()
    for name in sorted(a.keys() & b.keys()):
        if a[name].suffix.lower() not in ('.tif', '.tiff'):
            records.append({'path':name, 'presence_matched':True, 'values_checked':False})
            continue
        x, y = gdal.Open(str(a[name])), gdal.Open(str(b[name]))
        field = Path(name).stem.split('_')[0]
        if field not in rules: raise ValueError(f'No frozen TIFF gate for {field}: {name}')
        rule = rules[field]
        schema = (x.RasterXSize,x.RasterYSize,x.RasterCount,x.GetGeoTransform(),x.GetProjection(),x.GetMetadata()) == (y.RasterXSize,y.RasterYSize,y.RasterCount,y.GetGeoTransform(),y.GetProjection(),y.GetMetadata())
        bands=[]
        if schema:
            for i in range(1,x.RasterCount+1):
                u,v=x.GetRasterBand(i),y.GetRasterBand(i)
                meta=(u.DataType,u.GetMetadata(),u.GetDescription()) == (v.DataType,v.GetMetadata(),v.GetDescription()) and nodata_equal(u.GetNoDataValue(),v.GetNoDataValue())
                valid=bool(meta)
                maximum=0.0
                worst=(0,0)
                saw_finite=False
                # Comparison is outside timing; bound scratch independently of scene size.
                for row in range(0,x.RasterYSize,256):
                    for col in range(0,x.RasterXSize,256):
                        width=min(256,x.RasterXSize-col); height=min(256,x.RasterYSize-row)
                        aa=u.ReadAsArray(col,row,width,height); bb=v.ReadAsArray(col,row,width,height)
                        mask=np.array_equal(u.GetMaskBand().ReadAsArray(col,row,width,height),v.GetMaskBand().ReadAsArray(col,row,width,height))
                        special=all(np.array_equal(f(aa),f(bb)) for f in (np.isnan,np.isposinf,np.isneginf))
                        finite=np.isfinite(aa)&np.isfinite(bb)
                        # Subtract only finite entries to avoid inf/NaN arithmetic warnings.
                        delta=np.zeros(aa.shape,dtype='float64')
                        av=aa[finite].astype('float64'); bv=bb[finite].astype('float64')
                        delta[finite]=np.abs(av-bv)
                        exact=rule.get('rule','').startswith('exact')
                        tolerance=rule.get('max_abs',rule.get('atol',0))+rule.get('rtol',0)*np.abs(av)
                        values=bool(np.all(delta[finite] <= (0 if exact else tolerance)))
                        valid &= bool(mask and special and values)
                        if finite.any():
                            local=float(delta[finite].max())
                            local_worst=np.unravel_index(np.argmax(np.where(finite,delta,-1)),delta.shape)
                            coordinate=(row+int(local_worst[0]),col+int(local_worst[1]))
                            if not saw_finite or local > maximum or (local == maximum and coordinate < worst):
                                maximum=local; worst=coordinate
                            saw_finite=True
                bands.append(dict(band=i,passed=valid,max_abs=maximum,worst_coordinate=list(worst)))
        valid=schema and all(z['passed'] for z in bands)
        passed &= valid
        records.append(dict(path=name, schema_equal=schema, passed=valid, bands=bands))
    return dict(evidence_class='P6_vs_candidate_not_upstream_golden',passed=bool(passed),missing_left=sorted(b.keys()-a.keys()),missing_right=sorted(a.keys()-b.keys()),artifacts=records,limitations=['Non-TIFF legacy artifacts checked for presence only; intermediate chronological state requires separate numerical gates'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path); p.add_argument('--baseline-source',type=Path); p.add_argument('--candidate-source',type=Path); p.add_argument('--run',type=Path)
    p.add_argument('--python',default=sys.executable); p.add_argument('--execute',action='store_true'); p.add_argument('--child',type=Path)
    args=p.parse_args()
    if args.child: child(args.child); return
    if not all((args.protocol,args.baseline_source,args.candidate_source,args.run)): p.error('protocol, both source roots and run required')
    protocol=json.loads(args.protocol.read_text())
    if protocol.get('instrument_sha256') and protocol['instrument_sha256']!=digest(__file__):
        raise ValueError('Instrument differs from protocol binding')
    if protocol.get('parent_protocol') and digest(protocol['parent_protocol'])!=protocol.get('parent_protocol_sha256'):
        raise ValueError('Parent protocol differs from amendment binding')
    if protocol['repetitions'] != 5 or protocol['seed'] != 20260918 or protocol['native_budgets'] not in ([1],[4],[1,4]): raise ValueError('Unsupported frozen pairing settings')
    fixture=Path(protocol['fixture']).resolve()
    kwargs=json.loads(Path(protocol['kwargs_manifest']).read_text())
    for flag in FLAGS:
        if kwargs.get('save_'+flag) is not True: raise ValueError('Full-workload harness requires all ten save flags explicitly true')
    if not protocol['regimes'] or len(set(protocol['regimes'])) != len(protocol['regimes']): raise ValueError('Regimes must be nonempty and unique')
    for key in ('preprocess_relative',):
        relative=Path(protocol[key])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Geometry path must stay within scene')
    for regime in protocol['regimes']:
        if regime not in ('cold','geometry_warm'): raise ValueError('Unsupported regime')
    sources={'baseline':args.baseline_source.resolve(),'candidate':args.candidate_source.resolve()}
    baseline_validation=validate_protocol_baseline(sources['baseline'],protocol,sources['candidate'])
    if 'geometry_warm' in protocol['regimes'] and (protocol['runtime_options'].get('cache_enabled') is not True or protocol['runtime_options'].get('legacy_cache_policy')!='recompute' or protocol['runtime_options'].get('cache_dir')!='{scene}/.runtime_cache'): raise ValueError('Warm proof requires enabled/recompute cache at {scene}/.runtime_cache')
    frozen=dict(baseline_label=protocol.get('baseline_label','unchanged accepted P6'),baseline_validation=baseline_validation,protocol_sha256=digest(args.protocol),kwargs_sha256=digest(protocol['kwargs_manifest']),fixture=hashes(fixture),sources={k:hashes(v) for k,v in sources.items()},harness_sha256=digest(__file__),protocol=protocol,hardware=hardware_inventory())
    if protocol['fixture_hashes'] != frozen['fixture']: raise ValueError('Fixture differs from frozen input inventory')
    order_rng=random.Random(protocol['seed'])
    schedule=[]
    for regime in protocol['regimes']:
        for budget in protocol['native_budgets']:
            for rep in range(5):
                order=['baseline','candidate']; order_rng.shuffle(order)
                schedule.append(dict(regime=regime,budget=budget,repetition=rep,order=order))
    frozen['trial_order']=schedule
    if not args.execute:
        print(json.dumps({'status':'validated_not_executed','provenance':frozen},indent=2)); return
    root=args.run.resolve(); root.mkdir(parents=True,exist_ok=False); dump(root/'frozen.json',frozen)
    trials=[]; pairs=[]
    def invoke(backend,regime,budget,run,scene):
        effective=expand(kwargs,scene)
        if regime=='geometry_warm': effective={k:v for k,v in effective.items() if k.startswith('save_') or k in ('base_path','selected_date_str','tile_keys')}; effective['preprocess_dir']=str(scene/protocol['preprocess_relative'])
        runtime=expand(protocol['runtime_options'],scene)
        return measure(dict(source=str(sources[backend]),budget=budget,entrypoint='thermal_comfort' if regime=='cold' else 'run_utci_tiles',kwargs=effective,runtime=runtime),run,args.python)
    for regime in protocol['regimes']:
        for budget in protocol['native_budgets']:
            for rep in range(5):
                order=next(z['order'] for z in schedule if z['regime']==regime and z['budget']==budget and z['repetition']==rep); pair={}
                for backend in order:
                    if hashes(fixture) != frozen['fixture'] or digest(protocol['kwargs_manifest']) != frozen['kwargs_sha256']: raise RuntimeError('Frozen input changed')
                    if hashes(sources[backend]) != frozen['sources'][backend] or digest(__file__) != frozen['harness_sha256'] or digest(args.protocol)!=frozen['protocol_sha256']: raise RuntimeError('Frozen source/harness/protocol changed')
                    name=f'{regime}_n{budget}_r{rep}_{backend}'; scene=root/(name+'_scene'); shutil.copytree(fixture,scene)
                    for relative in protocol['cold_remove_relative']:
                        target=scene/relative
                        if not target.resolve().is_relative_to(scene): raise ValueError('Cleanup escapes scene')
                        if target.is_dir(): shutil.rmtree(target)
                        elif target.exists(): target.unlink()
                    setup=None; before=None; expected_keys=[]; preflight=None
                    if regime=='geometry_warm':
                        setup=invoke(backend,'cold',budget,root/(name+'_setup'),scene)
                        if setup['exit_code'] == 0:
                            before=cache_snapshot(scene/'.runtime_cache')
                            effective=expand(kwargs,scene); effective['preprocess_dir']=str(scene/protocol['preprocess_relative'])
                            preflight=measure(dict(source=str(sources[backend]),budget=budget,entrypoint='__cache_preflight',kwargs=effective,runtime=expand(protocol['runtime_options'],scene)),root/(name+'_cache_preflight'),args.python)
                            if preflight['exit_code']==0:
                                expected_keys=[z['key'] for z in json.loads((root/(name+'_cache_preflight')/'cache_preflight.json').read_text())['records']]
                            dump(root/(name+'_geometry_hashes.json'),hashes(scene/protocol['preprocess_relative']))
                            for relative in protocol['warm_remove_relative']:
                                target=scene/relative
                                if not target.resolve().is_relative_to(scene): raise ValueError('Cleanup escapes scene')
                                if target.is_dir(): shutil.rmtree(target)
                                elif target.exists(): target.unlink()
                    measurement=invoke(backend,regime,budget,root/name,scene) if setup is None or (setup['exit_code']==0 and preflight is not None and preflight['exit_code']==0) else dict(exit_code=1,setup_failed=True)
                    cache_proof=cache_reuse_proof(before,cache_snapshot(scene/'.runtime_cache'),expected_keys) if before is not None else {'passed':regime=='cold'}
                    if regime=='geometry_warm' and expected_keys:
                        tile_records=json.loads((root/(name+'_cache_preflight')/'cache_preflight.json').read_text())['records']
                        required=('UTCI','TMRT','Kup','Kdown','Lup','Ldown','Shadow','WBGT','Ta','Wind')
                        complete=all((scene/'output_folder'/z['tile']/f'{field}_{z["tile"]}.tif').is_file() for z in tile_records for field in required)
                        cache_proof['complete_requested_outputs_present']=complete
                        cache_proof['passed'] &= complete
                    record=dict(cache_proof=cache_proof,cache_preflight=preflight,backend=backend,regime=regime,budget=budget,repetition=rep,order=order,scene=str(scene),setup=setup,measurement=measurement)
                    trials.append(record); pair[backend]=record; dump(root/'trials.json',trials)
                comparison=compare(Path(pair['baseline']['scene']),Path(pair['candidate']['scene']),protocol['tiff_field_rules'],protocol['artifact_globs']) if all(z['measurement']['exit_code']==0 for z in pair.values()) else {'passed':False,'execution_failed':True}
                dump(root/f'{regime}_n{budget}_r{rep}_comparison.json',comparison)
                pairs.append(dict(cache_conditions_passed=all(z['cache_proof']['passed'] for z in pair.values()),regime=regime,budget=budget,repetition=rep,comparison=comparison,baseline_seconds=pair['baseline']['measurement'].get('elapsed_seconds_including_imports_jit_io'),candidate_seconds=pair['candidate']['measurement'].get('elapsed_seconds_including_imports_jit_io')))
                dump(root/'pairs.json',pairs)
    summaries=[]
    for regime in protocol['regimes']:
        for budget in protocol['native_budgets']:
            group=[z for z in pairs if z['regime']==regime and z['budget']==budget]; valid=[z for z in group if z['comparison']['passed'] and z['cache_conditions_passed']]
            ratios=[z['baseline_seconds']/z['candidate_seconds'] for z in valid]
            differences=[z['baseline_seconds']-z['candidate_seconds'] for z in valid]
            summaries.append(dict(regime=regime,budget=budget,valid_pairs=len(valid),failed_pairs=len(group)-len(valid),raw_paired_ratios=ratios,raw_paired_seconds_saved=differences,median_ratio=statistics.median(ratios) if ratios else None,range_ratio=[min(ratios),max(ratios)] if ratios else None,sample_stdev_ratio=statistics.stdev(ratios) if len(ratios)>1 else None,paired_seconds_standard_error=statistics.stdev(differences)/len(differences)**.5 if len(differences)>1 else None,performance_claim_eligible=len(valid)==5))
    unchanged=all(hashes(v)==frozen['sources'][k] for k,v in sources.items()) and hashes(fixture)==frozen['fixture'] and digest(protocol['kwargs_manifest'])==frozen['kwargs_sha256'] and digest(args.protocol)==frozen['protocol_sha256'] and digest(__file__)==frozen['harness_sha256']
    baseline_guard=validate_protocol_baseline(sources['baseline'],protocol,sources['candidate'])==frozen['baseline_validation']
    global_eligible=apply_global_eligibility(summaries,unchanged and baseline_guard and hardware_inventory()==frozen['hardware'] and frozen['hardware']['frozen_host_matches'],pairs)
    dump(root/'summary.json',dict(baseline_label=frozen['baseline_label'],comparison_baseline='explicitly labeled candidate baseline; not upstream golden',baseline_validation=frozen['baseline_validation'],global_evidence_eligible=global_eligible,evidence_class='P6_vs_candidate_not_upstream_golden',frozen_inputs_unchanged=unchanged,summaries=summaries))
    if not unchanged or not all(z['cache_conditions_passed'] for z in pairs) or any(z['failed_pairs'] for z in summaries): raise SystemExit(1)

if __name__=='__main__': main()
