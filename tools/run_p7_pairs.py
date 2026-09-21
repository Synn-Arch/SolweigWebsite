"""Run seven frozen paired protocols sequentially; retain every partial trial."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocols',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for case in ['small','repeated_block_256','dense_urban_256','vegetation_rich_256',
                 'real_dense_urban','real_vegetation_rich','real_sparse']:
        free=shutil.disk_usage(root).free
        if free<5*1024**3:
            records.append({'case':case,'status':'not_started_insufficient_disk_headroom','free_bytes':free})
            (args.output/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
            raise SystemExit(2)
        command=[sys.executable,str(root/'tools/benchmark_p7_pipeline.py'),
                 '--protocol',str(args.protocols/case/'protocol.json'),
                 '--baseline-source',str(root/'reports/characterization/p7_repaired_p6_baseline_v2/src'),
                 '--candidate-source',str(root/'src'),'--run',str(args.output/case),'--execute']
        print('Starting',case,flush=True)
        with (args.output/f'{case}.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        records.append({'case':case,'command':command,'exit_code':result.returncode,'free_bytes_before':free})
        (args.output/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
        print('Completed',case,result.returncode,flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
