# Numerical engine; see docs/protocols.md for interpretation and provenance.
import os
import sys
import math
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.dirname(ROOT)
RESULTS = os.path.join(THEORY, 'results')
sys.path.insert(0, ROOT)
from zoo_analysis import _spearman, _partial_spearman, _boot_ci, _seed_for

BASE_SSS = os.path.join(RESULTS, 'base_sss_cifar10.pkl')
COMP = os.path.join(RESULTS, 'exp_alpha_components.pkl')
OUT = os.path.join(RESULTS, 'width_partial_out.txt')
SEED = 1234








FLIPPERS = ['cond_sum', 'meco', 'jacob_cov', 'grasp']


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def build_sss():






    base = load(BASE_SSS)
    comp = {int(r['key']): r for r in load(COMP) if r.get('space') == 'sss'}
    recs = []
    for r in base:
        idx = int(r['i'])
        rec = {'index': idx}
        rec['acc'] = float(r.get('testacc', np.nan))           
        rec['valacc'] = float(r.get('valacc', np.nan))         
        rec['params'] = float(r.get('params', np.nan))         
        rec['param_count'] = float(r.get('param_count', np.nan))
        for q in ('grasp', 'meco', 'jacob_cov'):
            rec[q] = float(r.get(q, np.nan))
        c = comp.get(idx)
        rec['cond_sum'] = float(c['cond_sum']) if c else np.nan
        rec['erank'] = float(c['Ralpha_sum'][1.0]) if c else np.nan   
        recs.append(rec)
    return recs


def main():
    lines = []

    def L(s=''):
        lines.append(s)
        print(s)

    recs = build_sss()
    acc = np.array([r['acc'] for r in recs], float)
    valacc = np.array([r['valacc'] for r in recs], float)
    params = np.array([r['params'] for r in recs], float)
    erank = np.array([r['erank'] for r in recs], float)
    n_all = len(recs)
    n_cond = int(np.sum([np.isfinite(r['cond_sum']) for r in recs]))
    n_er = int(np.isfinite(erank).sum())
    rho_va_ta = _spearman(valacc, acc)   

    L('=' * 92)
    L('WIDTH-ADJUSTED PARTIAL ASSOCIATIONS -- NB201-SSS (channel-width varies; params monotone in width)')
    L('python: %s   numpy %s   SEED=%d' % (sys.executable, np.__version__, SEED))
    L('NO TRAINING / NO net build -- partial-correlation re-analysis of existing pkls only.')
    L('=' * 92)
    L('  n(base SSS)=%d ; erank/cond_sum available for n=%d (exp_alpha join).' % (n_all, n_cond))
    L('  acc field = testacc (matches zoo_analysis STDONLY, zoo_analysis.py:469); '
      'Spearman(testacc,valacc)=%+.3f.' % rho_va_ta)
    L('  Spearman(param_count, params) on SSS = %+.5f  (params IS the width surrogate).'
      % _spearman(np.array([r['param_count'] for r in recs], float), params))
    L('  W controls: [params]=raw width ; [erank]=representation effective rank ; [params,erank]=BOTH.')
    L('  LIKE-FOR-LIKE: for each proxy ALL controls (and the raw corr) are computed on the SAME')
    L('  per-proxy complete-case set (finite Q, acc, params AND erank), so params-only / erank-only /')
    L('  params+erank form a nested within-sample progression on identical rows (no cross-row contrast).')
    L('  VERDICT per cell: partial CI excludes 0 -> residual association (CI excludes zero);')
    L('                    partial CI spans 0    -> residual association is not resolved by this interval.')

    summary = {}
    for q in FLIPPERS:
        Qfull = np.array([r.get(q, np.nan) for r in recs], float)
        
        
        
        cc = (np.isfinite(Qfull) & np.isfinite(acc) & np.isfinite(params) & np.isfinite(erank))
        ncc = int(cc.sum())
        Q = Qfull[cc]; accc = acc[cc]; paramsc = params[cc]; erankc = erank[cc]
        raw = _spearman(Q, accc)                  
        raw_full = _spearman(Qfull, acc)          
        L('\n' + '-' * 92)
        L('PROXY %-10s  (complete-case n=%d; raw on all-finite n=%d)   '
          'raw corr(Q, acc_SSS)[cc] = %+.3f  (all-finite = %+.3f)'
          % (q, ncc, int(np.isfinite(Qfull).sum()), raw, raw_full))
        L('-' * 92)
        cell = {'raw': raw, 'raw_allfinite': raw_full, 'n': ncc}
        if ncc < 10:
            L('   <10 complete-case rows (finite Q,acc,params,erank); skipping partials for %s.' % q)
            cell['params+erank'] = {'partial': float('nan'), 'ci': [float('nan')] * 2, 'n': ncc}
            summary[q] = cell
            continue
        
        controls = [
            ('params',        paramsc[:, None]),
            ('erank',         erankc[:, None]),
            ('params+erank',  np.column_stack([paramsc, erankc])),
        ]
        for cname, W in controls:
            pr = _partial_spearman(Q, accc, W)
            lo, hi, nb = _boot_ci(
                lambda ix, Q=Q, accc=accc, W=W: _partial_spearman(Q[ix], accc[ix], W[ix]),
                ncc, seed=_seed_for('SSS|%s|%s' % (q, cname)))
            ci_excl = (math.isfinite(lo) and math.isfinite(hi) and
                       ((lo > 0 and hi > 0) or (lo < 0 and hi < 0)))
            same_sign = math.isfinite(pr) and math.isfinite(raw) and (np.sign(pr) == np.sign(raw))
            verdict = ('residual association (CI excludes 0)' if ci_excl
                       else 'CI includes zero')
            L('   partial(Q, acc | %-12s) = %+.3f  95%%CI[%+.3f,%+.3f]  n=%d  -> %s%s'
              % (cname, pr, lo, hi, ncc, verdict,
                 '' if same_sign else '  [sign changes vs raw]'))
            cell[cname] = {'partial': pr, 'ci': [lo, hi], 'n': ncc,
                           'ci_excludes_0': ci_excl}
        
        both = cell.get('params+erank', {})
        if both.get('ci_excludes_0'):
            L('   => %s: residual association excludes zero after params/effective-rank adjustment.' % q)
        elif math.isfinite(both.get('partial', float('nan'))):
            L('   => %s: residual association is not resolved after params/effective-rank adjustment.' % q)
        summary[q] = cell

    L('\n' + '=' * 92)
    L('SUMMARY (control = params+erank, the strongest width control; per-proxy complete-case rows,')
    L('  finite Q/acc/params/erank -- interpretation uses these erank-controlled cells)')
    L('=' * 92)
    L('  %-10s %9s %22s %s' % ('proxy', 'raw[cc]', 'partial|params+erank', 'verdict'))
    for q in FLIPPERS:
        both = summary[q].get('params+erank', {})
        pr = both.get('partial', float('nan'))
        ci = both.get('ci', [float('nan')] * 2)
        v = ('residual association' if both.get('ci_excludes_0')
             else ('CI includes zero' if math.isfinite(pr) else 'n/a'))
        L('  %-10s %+9.3f   %+9.3f [%+.3f,%+.3f]   %s'
          % (q, summary[q]['raw'], pr, ci[0], ci[1], v))

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n[width_partial] wrote %s' % OUT)


if __name__ == '__main__':
    main()
