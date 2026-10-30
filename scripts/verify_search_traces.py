"""Run the full trace verifier with exactly equivalent vectorized tie ranking.

The independent verifier's selections, seeds, stored scores, comparisons and
coverage are unchanged. Before execution, compare the optimized ranking with
the reference implementation on target columns and deterministic resamples.
"""
import argparse, gzip, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def ranks(values):
    values=np.asarray(values,dtype=float)
    result=np.full(len(values),np.nan)
    keep=np.isfinite(values)
    finite=values[keep]
    if len(finite)==0:return result
    order=np.argsort(finite,kind='mergesort')
    sorted_values=finite[order]
    left=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
    right=np.r_[left[1:],len(finite)]
    ranked=np.empty(len(finite),dtype=float)
    ranked[order]=np.repeat((left+right-1)/2.0,right-left)
    result[keep]=ranked
    return result

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--equivalence-report',type=Path,default=ROOT/'runs/search_rank_equivalence.json')
    args,remaining=parser.parse_known_args()
    spec=importlib.util.spec_from_file_location('independent_trace_verifier',ROOT/'src/engines/search_transfer/verify.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    reference=verifier.ranks
    arrays=[np.array([]),np.array([np.nan,np.inf,-np.inf]),np.array([1,1,2,0,np.nan,np.inf,-np.inf]),np.array([0,-0.0,1,2])]
    for file in sorted((args.output/'manifests').glob('*.json.gz')):
        with gzip.open(file,'rt',encoding='utf-8') as stream:record=json.load(stream)
        arrays.append(np.asarray(record['target_cached_accuracy'],float))
        arrays.extend(np.asarray([np.nan if x is None else x for x in values],float) for values in record['target_proxy_scores'].values())
    checked=0
    for array in arrays:
        for seed in range(11):
            sample=array if seed==0 or not len(array) else array[np.random.RandomState(seed).randint(0,len(array),len(array))]
            if not np.array_equal(reference(sample),ranks(sample),equal_nan=True):
                raise AssertionError('Tie ranking differs from the reference implementation')
            checked+=1
    report={'status':'PASS','arrays_checked':checked,'equality':'exact (including NaN mask)','scope':'Reference tie-rank equivalence on target columns, deterministic duplicate resamples and nonfinite edge cases; all other verifier operations unchanged.'}
    args.equivalence_report.parent.mkdir(parents=True,exist_ok=True)
    args.equivalence_report.write_text(json.dumps(report,indent=2)+'\n')
    verifier.ranks=ranks
    load=np.load
    def progress_load(path,*a,**kw):
        print('Checking '+Path(path).relative_to(args.output).as_posix(),file=sys.stderr,flush=True)
        return load(path,*a,**kw)
    verifier.np.load=progress_load
    sys.argv=[sys.argv[0],'--output',str(args.output),*remaining]
    verifier.main()

if __name__=='__main__':main()
