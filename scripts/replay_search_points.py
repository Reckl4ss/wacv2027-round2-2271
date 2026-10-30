"""Recompute all search point estimates from frozen target manifests.

Source correlations/coverage are cached, not regenerated from upstream files.
This does not recompute the 1,000-draw intervals or replace full trace verification.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('search_engine',ROOT/'src/engines/search_transfer/run.py')
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def main(output):
    ref=ROOT/'results/reference/search_transfer'
    with (ref/'method_summary.csv').open(newline='') as f:
        expected={(r['protocol'],r['pair_id'],int(r['K']),r['method']):float(r['point_mean_cached_accuracy']) for r in csv.DictReader(f)}
    observed=[]
    for path in sorted((ref/'manifests').glob('*.json.gz')):
        with gzip.open(path,'rt') as f:
            doc=json.load(f)
        acc=np.asarray(doc['target_cached_accuracy'],float)
        columns={key:np.asarray(value,float) for key,value in doc['target_proxy_scores'].items()}
        signs=doc['source_signs']
        rhos=doc['source_rhos']
        full=doc['full_coverage_proxies']
        best=max(full,key=lambda p:abs(rhos[p]))
        assert best==doc['source_best']
        samples={(k,method):[] for k in m.KS for method in m.METHODS}
        for s in range(m.N_SPLITS):
            rng=np.random.RandomState(m.SEED+1+s)
            perm=rng.permutation(len(acc))
            split=round(m.CALIB_FRAC*len(acc))
            calib,ev=perm[:split],perm[split:]
            assert not set(calib)&set(ev)
            survivors=[p for p in full if m.sign(m.spearman(columns[p][calib],acc[calib]))==signs[p]]
            screen=max(survivors,key=lambda p:abs(rhos[p])) if survivors else best
            scores={name:m.oriented_score(columns[p][ev],signs[p]) for name,p in [('source_best',best),('params','params'),('synflow','synflow'),('sign_stability_screen',screen)]}
            # Match the engine's frozen fixed-rank ensemble, including normalization.
            pair=m.Pair(doc['pair_id'],doc['role'],None,m.Axis(doc['target_axis'],np.asarray(doc['target_architecture_ids']),acc,columns,np.asarray(doc['target_input_rows'])),tuple(doc['proxy_order']))
            scores['rank_ensemble']=m.ensemble_score(pair,ev,signs)
            for k in m.KS:
                for method in m.DETERMINISTIC:
                    chosen=ev[m.top_positions(scores[method],k)]
                    samples[(k,method)].append(float(np.mean(acc[chosen])))
                random=[]
                for _ in range(m.N_RANDOM):
                    chosen=ev[rng.choice(len(ev),k,replace=False)]
                    random.append(float(np.mean(acc[chosen])))
                samples[(k,'random')].append(float(np.mean(random)))
        for protocol in m.PROTOCOLS:
            for (k,method),values in samples.items():
                value=float(np.mean(values))
                reference=expected[(protocol,doc['pair_id'],k,method)]
                if not math.isclose(value,reference,rel_tol=0,abs_tol=1e-10):
                    raise AssertionError((doc['pair_id'],k,method,value,reference))
                observed.append(dict(protocol=protocol,pair_id=doc['pair_id'],K=k,method=method,recomputed=value,reference=reference,abs_error=abs(value-reference)))
    output.mkdir(parents=True,exist_ok=True)
    (output/'point_checks.json').write_text(json.dumps(observed,indent=2)+'\n')
    report={'status':'PASS','point_estimates_checked':len(observed),'max_abs_error':max(r['abs_error'] for r in observed),'scope':'40-split point estimates from frozen target manifests; source-selection and bootstrap intervals not independently rerun'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
