"""Validate equal workloads and every P3 numerical benchmark output."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    entries=[(p.parent,json.loads(p.read_text())) for p in sorted(args.directory.glob('*/report.json'))]
    summaries=[]
    for directory,item in entries:
        reference_dir,reference=next((p,r) for p,r in entries if r['task']==item['task'] and r['bush']==item['bush'] and r['backend']=='upstream' and r['threads']==1)
        assert item['input_sha256']==reference['input_sha256'] and item['protocol_sha256']==reference['protocol_sha256']
        errors={}
        for filename,checksum in item['artifacts'].items():
            assert sha(directory/filename)==checksum
            assert sha(reference_dir/filename)==reference['artifacts'][filename]
            with np.load(directory/filename,allow_pickle=False) as actual, np.load(reference_dir/filename,allow_pickle=False) as expected:
                assert actual.files==expected.files
                for name in actual.files:
                    x,y=actual[name],expected[name]
                    assert x.shape==y.shape and x.dtype==y.dtype
                    for mask in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(x),mask(y))
                    exact=filename=='visibility.npz' or item['task']=='sky'
                    if item['task']=='wall13':exact=name in ['0','3','4']
                    if item['task']=='wall23':exact=name in ['0','1','2','6','7']
                    if exact:np.testing.assert_array_equal(x,y)
                    else:np.testing.assert_allclose(x,y,rtol=0,atol=1e-6)
                    delta=np.where(np.isfinite(y),abs(x-y),0);index=np.unravel_index(np.argmax(delta),delta.shape)
                    errors[f'{filename}/{name}']=dict(max_abs=float(delta[index]),worst_coordinate=list(map(int,index)))
        summaries.append(dict(report=str(directory/'report.json'),backend=item['backend'],task=item['task'],method=item['method'],parallel=item['parallel'],threads=item['threads'],bush=item['bush'],first_call_seconds=item['first_call_seconds'],warm_median_seconds=statistics.median(item['warm_seconds']),warm_min_seconds=min(item['warm_seconds']),warm_max_seconds=max(item['warm_seconds']),process_lifetime_peak_rss_bytes=item['process_lifetime_peak_rss_bytes'],retained_output_or_visibility_payload_bytes=item['retained_output_or_visibility_payload_bytes'],errors=errors))
    manifest=json.loads((args.directory/'execution_manifest.json').read_text())
    assert not manifest['failures'] and manifest['jobs']==len(entries)
    args.report.write_text(json.dumps(dict(scope='geometry kernels/SVF/visibility loading only, not full chronological simulation',comparisons_passed=len(entries),results=summaries),indent=2)+'\n')
    print(f'Passed all {len(entries)} equal-work configurations')


if __name__=='__main__':main()
