# Numerical engine; see docs/protocols.md for interpretation and provenance.
import argparse
import math
import os
import pickle
import sys

import numpy as np


import zoo_analysis as za
import crossspace as cs

ROOT = os.path.dirname(os.path.abspath(__file__))          
THEORY = os.path.dirname(ROOT)                              
RESULTS = os.path.join(THEORY, 'results')
PROJ = os.path.dirname(THEORY)

DATA = os.environ.get('NASLIB_DATA_DIR', os.path.join(PROJ, 'data', 'external', 'NASLib', 'naslib', 'data'))
ZCB = os.environ.get('NASLIB_BASELINES_DIR', os.path.join(PROJ, 'data', 'external', 'NASLib', 'zc_baseline_out'))

SEED = 1234
SMOKE_N = 60


STD_PROXIES = za.STD_PROXIES

SPEC_NAMES = ['cond_sum', 'stable_sum'] + ['Ralpha_%s' % a for a in (0.5, 2.0, 4.0, 8.0, 16.0)]

T_K = cs.T_K   





def _tee_to(path):
    f = open(path, 'w', encoding='utf-8')
    real = sys.stdout
    sys.stdout = za._Tee(real, f)
    return real, f


def _untee(real, f):
    sys.stdout = real
    f.close()











DSKEY = {'cf10': ('cifar10-valid', 'nb201_cifar10_full_training.pickle'),
         'cf100': ('cifar100', 'nb201_cifar100_full_training.pickle'),
         'IN16': ('ImageNet16-120', 'nb201_ImageNet16_full_training.pickle')}


def _load_pickle(p):
    with open(p, 'rb') as fh:
        return pickle.load(fh)


def build_nb201_records(verbose=True):



    base = _load_pickle(os.path.join(RESULTS, 'base_tss_cifar10.pkl'))
    base_by_idx = {e['i']: e for e in base}
    prem = [e for e in _load_pickle(os.path.join(RESULTS, 'premise_data.pkl'))
            if e.get('space') == 'tss']
    prem_by_idx = {int(e['key']): e for e in prem}

    curves = {}
    arch_list = None
    for dk, (innerkey, fname) in DSKEY.items():
        cd = _load_pickle(os.path.join(DATA, fname))
        curves[dk] = (innerkey, cd)
        if arch_list is None:
            arch_list = list(cd.keys())

    idxs_std = sorted(base_by_idx.keys())                       
    idxs_spec = sorted(set(base_by_idx) & set(prem_by_idx))     

    
    cf10ik, cf10cd = curves['cf10']
    bv, cf = [], []
    for ix in idxs_std:
        b = base_by_idx[ix]
        vv = b.get('valacc', b.get('val_acc', b.get('testacc', np.nan)))
        ea = cf10cd[arch_list[ix]][cf10ik].get('eval_acc1es', [])
        bv.append(vv); cf.append(ea[-1] if len(ea) else np.nan)
    bv = np.asarray(bv, float); cf = np.asarray(cf, float)
    mm = np.isfinite(bv) & np.isfinite(cf)
    align_meanabs = float(np.mean(np.abs(bv[mm] - cf[mm]))) if mm.sum() else float('nan')
    align_frac = float(np.mean(np.abs(bv[mm] - cf[mm]) <= 1.0)) if mm.sum() else float('nan')
    assert (not math.isfinite(align_meanabs)) or align_meanabs < 1.5, \
        'NB201 index->arch alignment WRONG: mean|d|=%.2fpp' % align_meanabs

    
    ro, rp = [], []
    for ix in idxs_spec:
        e = prem_by_idx.get(ix)
        if e is None or 'T_slope' not in e:
            continue
        ours = cs.T_slope(cf10cd[arch_list[ix]][cf10ik].get('train_acc1es', []))
        ro.append(ours); rp.append(e['T_slope'])
    ro = np.asarray(ro, float); rp = np.asarray(rp, float)
    rmask = np.isfinite(ro) & np.isfinite(rp)
    from scipy import stats as _st
    T_repro_rho = float(_st.spearmanr(ro[rmask], rp[rmask]).correlation) if rmask.sum() > 4 else float('nan')
    T_repro_n = int(rmask.sum())

    def make_rec(ix, dk, with_spec):
        innerkey, cd = curves[dk]
        inner = cd[arch_list[ix]][innerkey]
        b = base_by_idx[ix]
        rec = {'space': 'tss', 'index': ix}
        ea = inner.get('eval_acc1es', [])
        rec['acc'] = float(ea[-1]) if len(ea) else float('nan')
        rec['params'] = float(b.get('params', np.nan))
        rec['flops'] = float(b.get('flops', np.nan))
        ts = cs.T_slope(inner.get('train_acc1es', []))
        rec['T_slope'] = float(ts)
        
        tl = inner.get('train_losses', [])
        if len(tl) >= 2 and math.isfinite(tl[0]) and tl[0] > 0:
            rec['T_loss'] = float((tl[0] - tl[min(T_K, len(tl)) - 1]) / (tl[0] + 1e-8))
        else:
            rec['T_loss'] = float('nan')
        for pr in STD_PROXIES:
            rec[pr] = float(b.get(pr, np.nan))
        if with_spec:
            e = prem_by_idx[ix]
            rec['erank'] = float(e.get('erank', np.nan))
            rec['cond_sum'] = float(e.get('cond_sum', np.nan))
            rec['stable_sum'] = float(e.get('stable_sum', np.nan))
            ra = e.get('Ralpha_sum', {})
            for a in (0.5, 2.0, 4.0, 8.0, 16.0):
                rec['Ralpha_%s' % a] = float(ra.get(a, np.nan)) if isinstance(ra, dict) else float('nan')
        else:
            
            
            rec['erank'] = float('nan')
        return rec

    out = {}
    for dk in DSKEY:
        out[dk] = {
            'std': [make_rec(ix, dk, with_spec=False) for ix in idxs_std],
            'spec': [make_rec(ix, dk, with_spec=True) for ix in idxs_spec],
        }
    diag = {'align_meanabs': align_meanabs, 'align_frac1pp': align_frac, 'align_n': int(mm.sum()),
            'T_repro_rho': T_repro_rho, 'T_repro_n': T_repro_n,
            'n_std': len(idxs_std), 'n_spec': len(idxs_spec)}
    if verbose:
        print('[NB201 build] DATA SOURCES:')
        print('   std proxies : base_tss_cifar10.pkl  (n=%d, DATASET-AGNOSTIC cf10 values)' % len(idxs_std))
        print('   accuracy    : nb201_{cifar10,cifar100,ImageNet16}_full_training.pickle '
              'eval_acc1es[-1] (PER-DATASET)')
        print('   T_slope     : OLS slope of train_acc1es[:%d] PER-DATASET '
              '(certified crossspace.T_slope)' % T_K)
        print('   spectral    : premise_data.pkl erank/cond_sum/stable_sum/Ralpha_* '
              '(base&premise join n=%d)' % len(idxs_spec))
        print('   alignment   : base valacc vs cf10 curve final-acc mean|d|=%.3fpp, %.1f%% within 1pp '
              '(n=%d; assert<1.5pp)' % (align_meanabs, 100 * align_frac, int(mm.sum())))
        print('   T reproduce : Spearman(our cf10 train-acc T, premise.T_slope)=%.3f over n=%d '
              '(same def; rank-agree).' % (T_repro_rho, T_repro_n))
    return out, diag


def run_nb201_multidataset(smoke):
    real, fh = _tee_to(os.path.join(RESULTS, 'scaleup_nb201_multidataset_out%s.txt'
                                    % ('_smoke' if smoke else '')))  
    try:
        print('=' * 80)
        print('SCALE-UP 1: NB201 MULTI-DATASET FULL DECOMPOSITION (R1/R2/R3)  -- NO GPU, NO TRAINING')
        print('PYTHON:', sys.executable, '| numpy', np.__version__, '| SEED', SEED)
        print('MODE:', 'SMOKE' if smoke else 'FULL')
        print('Stats imported verbatim from zoo_analysis.py (certified). Std proxies = dataset-')
        print('AGNOSTIC cf10 values vs PER-DATASET accuracy + PER-DATASET early-epoch T_slope.')
        print('=' * 80)
        recs, diag = build_nb201_records(verbose=True)
        zoo_names = STD_PROXIES + SPEC_NAMES   

        out = {'R1': {}, 'R2': {}, 'R3': {}}
        for dk in ('cf10', 'cf100', 'IN16'):
            std_recs = recs[dk]['std']
            spec_recs = recs[dk]['spec']
            if smoke:
                std_recs = std_recs[:SMOKE_N]
                spec_recs = spec_recs[:SMOKE_N]

            print('\n' + '#' * 80)
            print('# NB201 dataset=%s' % dk)
            print('#   STD-PROXY arm  n=%d (capacity control = params; erank not joined here)' % len(std_recs))
            print('#   SPECTRAL arm   n=%d (capacity control = erank from premise; spectral proxies in zoo)'
                  % len(spec_recs))
            print('#' * 80)

            
            tag = 'STDONLY-%s' % dk
            print('\n----- [%s] STANDARD-12-PROXY arm (capacity control = PARAMS) -----' % dk)
            if len(std_recs) >= 10:
                za.run_R1(std_recs, STD_PROXIES, tag, out['R1'])
                za.run_R2(std_recs, tag, out['R2'])
                bo = [r for r in std_recs if math.isfinite(r.get('T_slope', np.nan))]
                if len(bo) >= 10:
                    za.run_R3(bo, STD_PROXIES, tag, out['R3'], target='T_slope')
                else:
                    print('  [%s R3] <10 finite-T archs; skipping.' % tag)
            else:
                print('  <10 archs; skipping std arm.')

            
            tagf = 'FULLZOO-%s' % dk
            print('\n----- [%s] FULL-ZOO (std + spectral) arm (capacity anchor = ERANK) -----' % dk)
            if len(spec_recs) >= 10:
                za.run_R1(spec_recs, zoo_names, tagf, out['R1'])
                za.run_R2(spec_recs, tagf, out['R2'])
                bo = [r for r in spec_recs if math.isfinite(r.get('T_slope', np.nan))]
                if len(bo) >= 10:
                    za.run_R3(bo, zoo_names, tagf, out['R3'], target='T_slope')
                else:
                    print('  [%s R3] <10 finite-T archs; skipping.' % tagf)
            else:
                print('  <10 archs; skipping spectral arm.')

        
        print('\n' + '=' * 80)
        print('R3 CROSS-DATASET VERDICT (full-zoo arm): does ANY label-free proxy robustly access')
        print('the trainability residual (partial(Q,T|erank,params)>0 & CI>0) in cf100 AND IN16?')
        print('=' * 80)
        za.r3_verdict(out['R3'], spaces=('cf100', 'IN16'), tagfmt='FULLZOO-%s')
        za.r3_verdict(out['R3'], spaces=('cf100', 'IN16'), tagfmt='STDONLY-%s')

        if not smoke:
            with open(os.path.join(RESULTS, 'scaleup_nb201_multidataset_results.pkl'), 'wb') as f:
                pickle.dump({'out': out, 'diag': diag}, f, protocol=2)
            print('\n[saved] scaleup_nb201_multidataset_results.pkl')
        print('\n[done] NB201 multi-dataset decomposition complete.')
    finally:
        _untee(real, fh)










def build_nb101_records(verbose=True):
    import json
    p = os.path.join(ZCB, 'v2--nasbench101--cifar10--9000.json')
    d = json.load(open(p))
    
    
    
    
    NB101_MAP = {'grad_norm': 'grad_norm', 'snip': 'snip', 'fisher': 'fisher',
                 'synflow': 'synflow', 'jacob_cov': 'jacov', 'grasp': 'grasp',
                 'plain': 'plain', 'zen': 'zen'}
    EXTRA = {'epe_nas': 'epe_nas', 'l2_norm': 'l2_norm', 'nwot': 'nwot'}
    recs = []
    for k, rec in d.items():
        r = {'space': 'nb101', 'index': k}
        r['acc'] = float(rec.get('val_acc', np.nan))
        r['params'] = float(rec.get('params', np.nan))
        r['flops'] = float(rec.get('flops', np.nan))
        r['erank'] = float('nan')      
        r['T_slope'] = float('nan')    
        r['T_loss'] = float('nan')
        for canon, src in NB101_MAP.items():
            r[canon] = float(rec.get(src, np.nan))
        for canon, src in EXTRA.items():
            r[canon] = float(rec.get(src, np.nan))
        recs.append(r)
    proxy_names = list(NB101_MAP.keys()) + list(EXTRA.keys())
    
    for nm in EXTRA:
        za.SIGN_PRIOR.setdefault(nm, +1)   
        za.LOG_PROXIES.add(nm) if nm in ('l2_norm', 'epe_nas') else None
    if verbose:
        print('[NB101 build] DATA SOURCE: %s (n=%d, SMALL-N)' % (os.path.basename(p), len(recs)))
        print('   acc = val_acc (json). capacity control = PARAMS (json).')
        print('   proxies = %s' % proxy_names)
        print('   BLOCKED no-GPU (stated, NOT fabricated): erank/spectral (need net build),')
        print('   early-epoch T_slope (needs full nasbench_full.tfrecord). -> R3 T-arm SKIPPED;')
        print('   we report params-controlled partial(Q, acc | params) instead.')
    return recs, proxy_names


def run_nb101(smoke):
    real, fh = _tee_to(os.path.join(RESULTS, 'scaleup_nb101_out%s.txt'
                                    % ('_smoke' if smoke else '')))  
    try:
        print('=' * 80)
        print('SCALE-UP 2: NB101 STANDARD-PROXY + PARAMS DECOMPOSITION  -- NO GPU, NO TRAINING')
        print('PYTHON:', sys.executable, '| numpy', np.__version__, '| SEED', SEED)
        print('MODE:', 'SMOKE' if smoke else 'FULL')
        print('Stats imported verbatim from zoo_analysis.py. CAPACITY CONTROL = params.')
        print('SMALL-N (n=100): CIs wide -> ALL NB101 findings are SUGGESTIVE.')
        print('=' * 80)
        recs, proxy_names = build_nb101_records(verbose=True)
        if smoke:
            recs = recs[:SMOKE_N]

        out = {'R1': {}, 'R2': {}, 'R3': {}}
        tag = 'NB101-cf10'
        print('\n' + '#' * 80)
        print('# NB101  n=%d  (capacity control = PARAMS; erank=NaN -> run_R2/R3 use params fallback)'
              % len(recs))
        print('#' * 80)

        
        
        za.run_R1(recs, proxy_names, tag, out['R1'])

        
        
        za.run_R2(recs, tag, out['R2'])

        
        
        
        
        print('\n----- [NB101] per-proxy access (target=T_slope is all-NaN -> partial(Q,T|..) shows')
        print('      n/a HONESTLY; the load-bearing NB101 column is partial(Q, acc-residual | params)')
        print('      and the ceiling rankR2(acc|params) vs +best proxy). -----')
        za.run_R3(recs, proxy_names, tag, out['R3'], target='T_slope')

        print('\n[NB101 INTERPRETATION] No-GPU NB101 supports ONLY the capacity-vs-complementarity')
        print('  arm (R1 PCA + NBSZ residual gain + ceiling rankR2(acc|params)). The trainability')
        print('  (T) arm and the spectral (erank) arm are BLOCKED without GPU/tfrecord and are NOT')
        print('  estimated (no fabricated T/erank). All numbers are SMALL-N (n=100) -> suggestive.')

        if not smoke:
            with open(os.path.join(RESULTS, 'scaleup_nb101_results.pkl'), 'wb') as f:
                pickle.dump(out, f, protocol=2)
            print('\n[saved] scaleup_nb101_results.pkl')
        print('\n[done] NB101 decomposition complete.')
    finally:
        _untee(real, fh)












def run_ci_quarantine(smoke):
    real, fh = _tee_to(os.path.join(RESULTS, 'scaleup_ci_quarantine_out%s.txt'
                                    % ('_smoke' if smoke else '')))  
    try:
        print('=' * 80)
        print('SCALE-UP 3: CROSS-SPACE CI QUARANTINE  -- NO GPU, NO TRAINING')
        print('PYTHON:', sys.executable, '| numpy', np.__version__, '| SEED', SEED, '| N_BOOT', cs.N_BOOT)
        print('Loaders + spearman + boot_spearman_ci imported verbatim from crossspace.py.')
        print('=' * 80)

        spaces, nb201, nb301 = cs.collect_spaces(verbose=True, smoke_cap=200 if smoke else None)

        
        DEGENERATE = {}
        DEGENERATE['NB301-cf10'] = ('DISJOINT: proxy json arch set does not match the curve arch '
                                    'set (genotype matches=%d/%d of %d curve archs); proxy<->T '
                                    'cannot be joined.' % (nb301.get('curve_match', 0),
                                                           nb301.get('n', 0), nb301.get('curve_n', 0)))
        
        
        for cname, dd, acck, _ in spaces:
            if cname == 'TransNAS-object':
                present = [p for p in cs.STD_PROXIES if p in dd and np.isfinite(dd[p]).sum() >= 4]
                if len(present) < 5:
                    DEGENERATE['TransNAS-object'] = ('SPARSE: only proxies %s present (full zoo not '
                                                     'scored); column underpowered.' % present)

        
        cells = {}   
        for pp in cs.STD_PROXIES:
            cells[pp] = {}
            for cname, dd, acck, _ in spaces:
                if pp not in dd:
                    continue
                r, n = cs.spearman(dd[pp], dd[acck])
                if not math.isfinite(r):
                    continue
                if n >= 8:
                    lo, hi = cs.boot_spearman_ci(dd[pp], dd[acck])
                else:
                    lo, hi = float('nan'), float('nan')
                cells[pp][cname] = (r, lo, hi, n)

        
        print('\n' + '=' * 80)
        print('TABLE A: per-(proxy, space) corr(Q,acc) + bootstrap 95% CI')
        print('  [DEGEN] = degenerate/dropped column (NB301 disjoint, TransNAS-object sparse).')
        print('=' * 80)
        print('proxy'.ljust(11) + 'space'.ljust(16) + 'rho      CI_low   CI_high  n     status')
        for pp in cs.STD_PROXIES:
            for cname, (r, lo, hi, n) in cells[pp].items():
                degen = cname in DEGENERATE
                ci = ('[%+.3f,%+.3f]' % (lo, hi)) if math.isfinite(lo) else '[n<8: no CI]'
                status = 'DEGEN' if degen else ('CI' if math.isfinite(lo) else 'no-CI(n<8)')
                print('%-11s%-16s%+.3f   %-22s n%-4d %s' % (pp, cname, r, ci, n, status))

        
        def ci_entirely_pos(lo, hi):
            return math.isfinite(lo) and lo > 0

        def ci_entirely_neg(lo, hi):
            return math.isfinite(hi) and hi < 0

        def cis_disjoint(a, b):
            (_, alo, ahi, _) = a; (_, blo, bhi, _) = b
            if not (math.isfinite(alo) and math.isfinite(blo)):
                return False
            return (ahi < blo) or (bhi < alo)

        print('\n' + '=' * 80)
        print('TABLE B: CROSS-SPACE SIGN-FLIP QUARANTINE')
        print('  A flip = a pair of NON-DEGENERATE spaces where the proxy has rho>+0.02 in one and')
        print('  rho<-0.02 in the other. CI-SEPARATED (load-bearing): one CI entirely >0, the other')
        print('  entirely <0, AND the two CIs are disjoint. Else SUGGESTIVE.')
        print('=' * 80)
        any_flip = False
        loadbearing = []
        for pp in cs.STD_PROXIES:
            sp_items = [(cn, v) for cn, v in cells[pp].items() if cn not in DEGENERATE]
            pos = [(cn, v) for cn, v in sp_items if v[0] > 0.02]
            neg = [(cn, v) for cn, v in sp_items if v[0] < -0.02]
            if not (pos and neg):
                continue
            any_flip = True
            print('\n  proxy=%s  (non-degenerate spaces only)' % pp)
            for cn, v in sorted(sp_items, key=lambda kv: kv[1][0]):
                r, lo, hi, n = v
                ci = ('[%+.3f,%+.3f]' % (lo, hi)) if math.isfinite(lo) else '[n<8]'
                print('       %-16s rho=%+.3f  CI=%-20s n=%d' % (cn, r, ci, n))
            
            best_pos = max(pos, key=lambda kv: kv[1][0])
            best_neg = min(neg, key=lambda kv: kv[1][0])
            sep = (ci_entirely_pos(*best_pos[1][1:3]) and ci_entirely_neg(*best_neg[1][1:3])
                   and cis_disjoint(best_pos[1], best_neg[1]))
            if sep:
                print('       -> CI-SEPARATED (LOAD-BEARING): %s rho%+.3f CI%s  VS  %s rho%+.3f CI%s'
                      % (best_pos[0], best_pos[1][0], '[%+.3f,%+.3f]' % best_pos[1][1:3],
                         best_neg[0], best_neg[1][0], '[%+.3f,%+.3f]' % best_neg[1][1:3]))
                loadbearing.append((pp, best_pos[0], best_neg[0]))
            else:
                print('       -> SUGGESTIVE (CIs overlap / straddle 0 / small-n): not load-bearing.')
        if not any_flip:
            print('\n  No proxy shows a +/- sign-flip across the non-degenerate spaces.')

        print('\n' + '=' * 80)
        print('SUMMARY')
        print('=' * 80)
        print('  LOAD-BEARING (CI-separated) sign-flips:')
        if loadbearing:
            for pp, sp_p, sp_n in loadbearing:
                print('     %-12s  %s (+) vs %s (-)' % (pp, sp_p, sp_n))
        else:
            print('     none (no flip survives the CI-separation + disjointness test).')
        print('\n  DROPPED / FLAGGED degenerate columns (excluded from the flip test):')
        for cn, why in DEGENERATE.items():
            print('     %-16s %s' % (cn, why))
        print('\n  NOTE: small-n columns (NB101 n~100, NB301 n=50) still receive CIs when n>=8 but')
        print('  are WIDE -> any flip involving them lands SUGGESTIVE unless the CIs are disjoint.')

        if not smoke:
            with open(os.path.join(RESULTS, 'scaleup_ci_quarantine_results.pkl'), 'wb') as f:
                pickle.dump({'cells': cells, 'loadbearing': loadbearing,
                             'degenerate': DEGENERATE}, f, protocol=2)
            print('\n[saved] scaleup_ci_quarantine_results.pkl')
        print('\n[done] CI quarantine complete.')
    finally:
        _untee(real, fh)





def main():
    ap = argparse.ArgumentParser(description='No-training scale-up of the certified NB201 R1-R3 '
                                             'decomposition + cross-space CI quarantine.')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--smoke', action='store_true', help='small slice; verify it runs end-to-end.')
    g.add_argument('--full', action='store_true', help='all archs; full bootstrap; save outputs.')
    ap.add_argument('--only', choices=['nb201', 'nb101', 'ci'], default=None,
                    help='run a single part.')
    args = ap.parse_args()
    if not args.full:
        args.smoke = True
    smoke = args.smoke

    np.random.seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)
    print('PYTHON:', sys.executable)
    print('numpy', np.__version__, '| MODE', 'SMOKE' if smoke else 'FULL', '| SEED', SEED)

    if args.only in (None, 'nb201'):
        print('\n>>> [1/3] NB201 multi-dataset ...')
        run_nb201_multidataset(smoke)
        print('    -> theory/results/scaleup_nb201_multidataset_out.txt')
    if args.only in (None, 'nb101'):
        print('\n>>> [2/3] NB101 ...')
        run_nb101(smoke)
        print('    -> theory/results/scaleup_nb101_out.txt')
    if args.only in (None, 'ci'):
        print('\n>>> [3/3] cross-space CI quarantine ...')
        run_ci_quarantine(smoke)
        print('    -> theory/results/scaleup_ci_quarantine_out.txt')
    print('\n[ALL DONE]')


if __name__ == '__main__':
    main()
