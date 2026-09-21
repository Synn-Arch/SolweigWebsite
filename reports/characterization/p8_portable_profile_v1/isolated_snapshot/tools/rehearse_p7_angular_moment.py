#!/usr/bin/env python3
"""One real-worker smoke and one-pair diagnostic rehearsal; never benchmark evidence."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('angular_harness',ROOT/'tools/run_p7_angular_moment.py')
H=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(H)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--python',required=True);args=parser.parse_args()
    protocol_path=args.protocol.resolve();validation=H.validate_protocol(protocol_path);protocol=validation['protocol']
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sources,inventories=H.create_source_snapshots(args.output,protocol)
    case=protocol['fixtures']['small'];runtime_template=protocol['runtime'];records={}

    def run(label,source,root):
        scene,kwargs=H.prepare_scene(case,root/'scene');jit=root/'jit'
        runtime=H.runtime_options(protocol,scene,1)
        measurement=H.invoke_worker(source,'thermal_comfort',kwargs,runtime,root/'process',jit,protocol,args.python)
        status=H.execution_status(measurement,root/'process',scene)
        outputs=H.validate_outputs(scene,protocol['matrix']['output_fields'],protocol['matrix']['timesteps'])
        record={'measurement':measurement,'status':status,'outputs':outputs,'scene':str(scene)}
        H.write_json(root/'record.json',record)
        if not status['passed'] or not outputs['passed']:raise RuntimeError(f'{label} failed real worker rehearsal')
        return record

    smoke=run('candidate_smoke',sources['candidate'],args.output/'smoke_candidate')
    baseline=run('baseline_pair',sources['baseline'],args.output/'pair'/'baseline')
    candidate=run('candidate_pair',sources['candidate'],args.output/'pair'/'candidate')
    comparison=H.pair_comparison(Path(baseline['scene']),Path(candidate['scene']),protocol,validation['comparison_rules'])
    if not comparison['passed']:raise RuntimeError('one-pair output comparison failed')
    result={'schema':'p7_angular_real_subprocess_rehearsal_v1','status':'pass','benchmark_evidence':False,
        'protocol_sha256':H.digest(protocol_path),'harness_sha256':H.digest(ROOT/'tools/run_p7_angular_moment.py'),
        'source_inventories':inventories,'smoke':smoke,'pair':{'baseline':baseline,'candidate':candidate,'comparison':comparison,
        'diagnostic_ratio_candidate_over_baseline':candidate['measurement']['elapsed_seconds']/baseline['measurement']['elapsed_seconds']},
        'qualification':'single small first-use pair for execution correctness only; excluded from the frozen 40-pair matrix'}
    H.write_json(args.output/'result.json',result);print(json.dumps({'status':'pass','smoke_outputs':smoke['outputs']['passed'],'pair_comparison':comparison['passed']},indent=2))


if __name__=='__main__':main()
