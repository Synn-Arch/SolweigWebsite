"""Artifact gates for P6/candidate comparisons; no benchmark execution."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

SPEC = importlib.util.spec_from_file_location('p7_pair_harness', Path(__file__).resolve().parents[2] / 'tools/benchmark_p7_pipeline.py')
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)
gdal.UseExceptions()


def raster(directory, values, nodata=None, metadata=None):
    directory.mkdir(exist_ok=True)
    values = np.asarray(values, dtype=np.float32)
    path = directory / 'UTCI_0_0.tif'
    ds = gdal.GetDriverByName('GTiff').Create(str(path), values.shape[1], values.shape[0], 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(values)
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(nodata)
    if metadata:
        ds.SetMetadata(metadata)
    ds = None
    return path


def compare(tmp_path, rule=None):
    return HARNESS.compare(tmp_path / 'a', tmp_path / 'b', {'UTCI': rule or {'max_abs': .02}}, ['*.tif', '*.npz'])


@pytest.mark.parametrize('left,right,equal', [(None,None,True), (None,float('nan'),False), (float('nan'),float('nan'),True), (float('nan'),-9999,False), (-9999,-9999,True), (-9999,-9998,False), (float('inf'),float('inf'),True), (float('inf'),float('-inf'),False)])
def test_nodata_identity(left, right, equal):
    assert HARNESS.nodata_equal(left, right) is equal


def test_nan_nodata_masks_and_special_locations(tmp_path):
    values = [[1, np.nan], [np.inf, -np.inf]]
    raster(tmp_path / 'a', values, np.nan)
    raster(tmp_path / 'b', values, np.nan)
    assert compare(tmp_path)['passed']
    raster(tmp_path / 'b', [[np.nan, 1], [np.inf, -np.inf]], np.nan)
    assert not compare(tmp_path)['passed']


def test_numeric_tolerance_across_windows_and_worst_coordinate(tmp_path):
    a = np.ones((258, 259), dtype=np.float32)
    b = a.copy(); b[257, 258] += .01
    raster(tmp_path / 'a', a); raster(tmp_path / 'b', b)
    result = compare(tmp_path)
    assert result['passed']
    band = result['artifacts'][0]['bands'][0]
    assert band['worst_coordinate'] == [257, 258]
    assert band['max_abs'] == pytest.approx(.01, abs=1e-6)
    assert not compare(tmp_path, {'max_abs': .001})['passed']
    assert not compare(tmp_path, {'rule': 'exact_value_and_masks'})['passed']


def test_relative_tolerance_uses_baseline(tmp_path):
    raster(tmp_path / 'a', [[100]])
    raster(tmp_path / 'b', [[100.5]])
    assert compare(tmp_path, {'atol': .01, 'rtol': .005})['passed']
    assert not compare(tmp_path, {'atol': .01, 'rtol': .001})['passed']


def test_metadata_and_nodata_mismatch(tmp_path):
    raster(tmp_path / 'a', [[1]], np.nan, {'timestamp': 'a'})
    raster(tmp_path / 'b', [[1]], np.nan, {'timestamp': 'b'})
    assert not compare(tmp_path)['passed']
    raster(tmp_path / 'b', [[1]], None, {'timestamp': 'a'})
    assert not compare(tmp_path)['passed']


def test_missing_artifact_and_rule(tmp_path):
    raster(tmp_path / 'a', [[1]])
    raster(tmp_path / 'b', [[1]])
    (tmp_path / 'a' / 'legacy.npz').write_bytes(b'presence')
    result = compare(tmp_path)
    assert not result['passed']
    assert result['missing_right'] == ['legacy.npz']
    with pytest.raises(ValueError, match='No frozen TIFF gate'):
        HARNESS.compare(tmp_path / 'a', tmp_path / 'b', {}, ['*.tif'])


def test_accepted_baseline_and_mismatch(tmp_path):
    accepted = Path(__file__).resolve().parents[2] / 'reports/characterization/p7_p6_baseline/src'
    assert HARNESS.validate_baseline(accepted)['passed']
    with pytest.raises(ValueError, match='Supplied baseline differs'):
        HARNESS.validate_baseline(tmp_path)


def test_warm_cache_change_or_missing_rejected():
    before = {'pipeline_manifests': {'k/manifest.json': {'key': 'k', 'sha256': 'a'}}, 'generations': ['k/generation-a']}
    assert HARNESS.cache_reuse_proof(before, before, ['k'])['passed']
    changed = {'pipeline_manifests': {'k/manifest.json': {'key': 'k', 'sha256': 'b'}}, 'generations': ['k/generation-a', 'k/generation-b']}
    assert not HARNESS.cache_reuse_proof(before, changed, ['k'])['passed']
    assert not HARNESS.cache_reuse_proof(before, before, ['missing'])['passed']
    assert not HARNESS.cache_reuse_proof(before, before, [])['passed']


@pytest.mark.parametrize('guard,comparison,cache,expected', [(True,True,True,True),(False,True,True,False),(True,False,True,False),(True,True,False,False)])
def test_global_eligibility_requires_all_evidence(guard, comparison, cache, expected):
    summaries = [{'valid_pairs': 5}, {'valid_pairs': 5}]
    pairs = [{'comparison': {'passed': comparison}, 'cache_conditions_passed': cache}]
    assert HARNESS.apply_global_eligibility(summaries, guard, pairs) is expected
    assert all(z['performance_claim_eligible'] is expected for z in summaries)
    assert all(z['subgroup_comparison_passed'] for z in summaries)


@pytest.fixture(params=['p7_repaired_p6_baseline', 'p7_repaired_p6_baseline_v2'])
def repaired_packet(tmp_path, request):
    import json
    import shutil
    root=Path(__file__).resolve().parents[2]
    snapshot=root/'reports/characterization'/request.param
    manifest=json.loads((snapshot/'manifest.json').read_text())
    source=snapshot/'src'; candidate=root/'src'
    ref=tmp_path/'evidence.json'; ref.write_text('{}')
    evidence={'path':str(ref),'sha256':HARNESS.digest(ref)}
    geometry=tmp_path/'geometry.json'
    geometry.write_text(json.dumps({'status':'passed','records':[{'artifact':'synthetic','field':str(i)} for i in range(18)]}))
    geometry_evidence={'path':str(geometry),'sha256':HARNESS.digest(geometry)}
    ids=[f'fixture_{i}' for i in range(7)]
    report={'status':'passed','fixture_case_ids':ids,'source_hashes':{'baseline':HARNESS.hashes(source),'candidate':HARNESS.hashes(candidate)},'cases':[{'case_id':case,'variant':variant,'passed':True,'evidence':evidence} for case in ids for variant in ('baseline','candidate')], 'original_regression':{variant:{'passed':True,'cases':9,'evidence':evidence} for variant in ('baseline','candidate')},'angular_ground_view':{variant:{'passed':True,'field_count':17,'boundary_passed':True,'evidence':evidence} for variant in ('baseline','candidate')}}
    for case in report['cases']:
        case['geometry_evidence']=geometry_evidence
    policy=manifest.get('repair_policy','angular_v1')
    if policy=='angular_and_svf_weights_v1':
        report['svf_weights']={variant:{'passed':True,'weight_count':180,'boundary_pixel_count':2,'maximum_weight_ulp':1,'evidence':evidence} for variant in ('baseline','candidate')}
    destination=tmp_path/'manifest.json'; destination.write_text(json.dumps(manifest))
    patch='compatibility_repair.patch' if policy=='angular_and_svf_weights_v1' else 'angular_compatibility_repair.patch'
    shutil.copyfile(snapshot/patch,tmp_path/patch)
    report_path=tmp_path/'correctness.json'; report_path.write_text(json.dumps(report))
    protocol={'baseline_manifest':str(destination),'baseline_manifest_sha256':HARNESS.digest(destination),'baseline_label':manifest['label'],'repair_patch_sha256':manifest['repair_patch_sha256'],'correctness_report':str(report_path),'correctness_report_sha256':HARNESS.digest(report_path)}
    protocol['repair_policy']=policy
    return source,candidate,protocol,manifest,report


def test_valid_repaired_derivation(repaired_packet):
    source,candidate,protocol,_,_=repaired_packet
    result=HARNESS.validate_protocol_baseline(source,protocol,candidate)
    assert result['passed']
    assert result['full_repair_diff_sha256']==protocol['repair_patch_sha256']
    assert result['numerical_gate']['status']=='validated_external_gate'


@pytest.mark.parametrize('mutation', ['parent','changed_file','repair_hash','missing_report','invalid_report','source_hashes','missing_case','missing_boundary','missing_original','bad_evidence','policy_mismatch','unknown_policy','missing_geometry','bad_geometry_hash','missing_geometry_field'])
def test_repaired_gate_rejects_invalid_packet(repaired_packet,mutation):
    import json
    source,candidate,protocol,manifest,report=repaired_packet
    if mutation=='parent': manifest['parent_manifest_sha256']='wrong'
    elif mutation=='changed_file': manifest['changed_files']['src/solweig_light/api.py']={}
    elif mutation=='repair_hash': protocol.pop('repair_patch_sha256')
    elif mutation=='missing_report': protocol['correctness_report']='/nonexistent/p7-report.json'
    elif mutation=='invalid_report': report['status']='failed'
    elif mutation=='source_hashes': report['source_hashes']['baseline']={}
    elif mutation=='missing_case': report['cases'].pop()
    elif mutation=='missing_boundary': report['angular_ground_view']['baseline']['boundary_passed']=False
    elif mutation=='missing_original': report['original_regression']['candidate']['cases']=8
    elif mutation=='bad_evidence': report['cases'][0]['evidence']['sha256']='wrong'
    elif mutation=='policy_mismatch': protocol['repair_policy']='mismatched'
    elif mutation=='unknown_policy': manifest['repair_policy']=protocol['repair_policy']='unknown'
    elif mutation=='missing_geometry': report['cases'][0].pop('geometry_evidence')
    elif mutation=='bad_geometry_hash': report['cases'][0]['geometry_evidence']['sha256']='wrong'
    elif mutation=='missing_geometry_field':
        ref=report['cases'][0]['geometry_evidence']; path=Path(ref['path'])
        geometry=json.loads(path.read_text()); geometry['records'].pop()
        path.write_text(json.dumps(geometry)); ref['sha256']=HARNESS.digest(path)
    manifest_path=Path(protocol['baseline_manifest']); manifest_path.write_text(json.dumps(manifest)); protocol['baseline_manifest_sha256']=HARNESS.digest(manifest_path)
    if mutation!='missing_report':
        report_path=Path(protocol['correctness_report']); report_path.write_text(json.dumps(report)); protocol['correctness_report_sha256']=HARNESS.digest(report_path)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        HARNESS.validate_protocol_baseline(source,protocol,candidate)


@pytest.mark.parametrize('repaired_packet', ['p7_repaired_p6_baseline_v2'], indirect=True)
@pytest.mark.parametrize('mutation', ['missing_weights', 'missing_receiver', 'bad_weight_evidence'])
def test_svf_repair_requires_its_own_evidence(repaired_packet, mutation):
    import json
    source,candidate,protocol,_,report=repaired_packet
    if mutation=='missing_weights':
        report.pop('svf_weights')
    elif mutation=='missing_receiver':
        report['svf_weights']['baseline']['boundary_pixel_count']=1
    else:
        # Replace instead of changing the shared test evidence dictionary.
        report['svf_weights']['candidate']['evidence']={'path':protocol['correctness_report'],'sha256':'wrong'}
    path=Path(protocol['correctness_report'])
    path.write_text(json.dumps(report))
    protocol['correctness_report_sha256']=HARNESS.digest(path)
    with pytest.raises(ValueError,match='SVF weight repair correctness coverage missing'):
        HARNESS.validate_protocol_baseline(source,protocol,candidate)
