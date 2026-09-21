"""Validate equal P2 workloads/outputs and summarize raw kernel-only trials."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args=parser.parse_args()
    trials=[(p,json.loads(p.read_text())) for p in sorted(args.directory.glob('*.json'))]
    summaries=[]
    for path,trial in trials:
        reference=next(record for _,record in trials if record['backend']=='upstream' and record['kernel']==trial['kernel'] and record['threads']==1)
        assert trial['input_sha256']==reference['input_sha256']
        assert trial['protocol_sha256']==reference['protocol_sha256']
        outputs=[]
        for record in (trial,reference):
            file=args.directory/record['output_file']
            assert hashlib.sha256(file.read_bytes()).hexdigest()==record['output_sha256']
            with np.load(file,allow_pickle=False) as archive:outputs.append(archive['output'])
        actual,expected=outputs
        assert actual.shape==expected.shape and actual.dtype==expected.dtype
        for mask in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(mask(actual),mask(expected))
        if trial['kernel'].startswith('utci'):
            np.testing.assert_allclose(actual,expected,rtol=0,atol=.02,equal_nan=True)
        else:
            np.testing.assert_array_equal(actual,expected)
        difference=np.where(np.isfinite(expected),abs(actual-expected),0)
        index=np.unravel_index(np.argmax(difference),difference.shape)
        summaries.append(dict(file=str(path),backend=trial['backend'],kernel=trial['kernel'],threads=trial['threads'],
            callable=trial['callable'],max_abs=float(difference[index]),worst_coordinate=list(map(int,index)),
            first_call_seconds=trial['first_call_seconds'],warm_median_seconds=statistics.median(trial['warm_seconds']),
            warm_min_seconds=min(trial['warm_seconds']),warm_max_seconds=max(trial['warm_seconds']),
            process_lifetime_peak_rss_bytes=trial['process_lifetime_peak_rss_bytes']))
    args.report.write_text(json.dumps(dict(scope='kernel-only; no end-to-end speedup inference',
        comparisons_passed=len(summaries),results=summaries),indent=2)+'\n')
    print(f'Validated {len(summaries)} equal-work kernel trials')


if __name__=='__main__':main()
