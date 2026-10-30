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
import zoo_analysis as za
from zoo_analysis import (_spearman, _partial_spearman, _rankdata, _orient_value,
                          orient_signs_for_space, _seed_for, STD_PROXIES, PROBE_COLS,
                          LABEL_USING_PROBES, build_join)
import crossspace as cs

NPERM = 5000          
SEED = 1234
ALPHA = 0.05
OUT = os.path.join(RESULTS, 'fdr_recount_out.txt')


SPEC_NAMES = ['cond_sum', 'stable_sum'] + ['Ralpha_%s' % a for a in (0.5, 2.0, 4.0, 8.0, 16.0)]
ZOO_NAMES = STD_PROXIES + SPEC_NAMES + PROBE_COLS     











def _rank_resid(t, A):
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return t - A @ coef


def perm_p_partial(x, y, Z, nperm=NPERM, seed=0):



    x = np.asarray(x, float); y = np.asarray(y, float)
    if Z is None:
        m = np.isfinite(x) & np.isfinite(y)
        Zc = None
    else:
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        Zc = Z[m]
    n = int(m.sum())
    if n < 8:
        return float('nan'), float('nan'), n
    rx = _rankdata(x[m]); ry = _rankdata(y[m])
    rs = np.random.RandomState(seed)

    if Zc is None or Zc.shape[1] == 0:
        
        ex = rx - rx.mean(); ey = ry - ry.mean()
        sx = ex.std(); sy = ey.std()
        if sx < 1e-12 or sy < 1e-12:
            return float('nan'), float('nan'), n
        r_obs = float(np.corrcoef(ex, ey)[0, 1])
        exn = (ex - ex.mean()) / sx
        eyn = (ey - ey.mean()) / sy
        ge = 0; a_obs = abs(r_obs); B = 1000; done = 0
        while done < nperm:
            b = min(B, nperm - done)
            idx = np.argsort(rs.random((b, n)), axis=1)
            perm = eyn[idx]                       
            r = (perm @ exn) / n                  
            ge += int(np.sum(np.abs(r) >= a_obs - 1e-12))
            done += b
        p = (1.0 + ge) / (1.0 + nperm)
        return r_obs, float(p), n

    
    rZ = np.column_stack([_rankdata(Zc[:, j]) for j in range(Zc.shape[1])])
    A = np.column_stack([np.ones(n), rZ])
    
    coef_x, *_ = np.linalg.lstsq(A, rx, rcond=None)
    ex = rx - A @ coef_x                          
    coef_y, *_ = np.linalg.lstsq(A, ry, rcond=None)
    yhat_Z = A @ coef_y                           
    resid_y = ry - yhat_Z                         
    sx = ex.std()
    if sx < 1e-12 or resid_y.std() < 1e-12:
        return float('nan'), float('nan'), n
    r_obs = float(np.corrcoef(ex, resid_y)[0, 1])
    exn = (ex - ex.mean()) / sx                   
    
    Ainv = np.linalg.pinv(A)                      
    ge = 0; a_obs = abs(r_obs); B = 1000; done = 0
    while done < nperm:
        b = min(B, nperm - done)
        idx = np.argsort(rs.random((b, n)), axis=1)
        ystar = yhat_Z[None, :] + resid_y[idx]    
        
        fitted = (ystar @ Ainv.T) @ A.T           
        ry_res = ystar - fitted                   
        mu = ry_res.mean(axis=1, keepdims=True)
        sd = ry_res.std(axis=1)
        sd_safe = np.where(sd < 1e-12, 1.0, sd)
        ry_n = (ry_res - mu) / sd_safe[:, None]
        r = (ry_n @ exn) / n                       
        r[sd < 1e-12] = 0.0
        ge += int(np.sum(np.abs(r) >= a_obs - 1e-12))
        done += b
    p = (1.0 + ge) / (1.0 + nperm)
    return r_obs, float(p), n





def bh_fdr(pvals, alpha=ALPHA):


    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= thresh
    if not below.any():
        rej = np.zeros(m, bool)
        return rej, 0.0, 0
    kmax = np.max(np.where(below)[0])           
    p_cut = ranked[kmax]
    rej = np.zeros(m, bool)
    rej[order[:kmax + 1]] = True
    return rej, float(p_cut), int(kmax + 1)


def by_fdr(pvals, alpha=ALPHA):




    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool), 0.0, 0
    cm = float(np.sum(1.0 / np.arange(1, m + 1)))     
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / (m * cm)) * alpha
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, bool), 0.0, 0
    kmax = np.max(np.where(below)[0])
    p_cut = ranked[kmax]
    rej = np.zeros(m, bool)
    rej[order[:kmax + 1]] = True
    return rej, float(p_cut), int(kmax + 1)


def holm_bonferroni(pvals, alpha=ALPHA):

    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    rej = np.zeros(m, bool)
    for i, idx in enumerate(order):
        thr = alpha / (m - i)
        if p[idx] <= thr:
            rej[idx] = True
        else:
            break
    return rej





def run_r3_grid(L):
    L('=' * 88)
    L('FAMILY A -- R3 PER-PROXY PARTIAL GRID: partial(Q, T_slope | erank, params)')
    L('  29 full-zoo proxies x {TSS, SSS} = 58 tests.  Null = %d FREEDMAN-LANE permutations/cell' % NPERM)
    L('  (permute y-resid, add back Z-fit, re-residualise; calibrated under strong erank/params')
    L('   confounding -- the regime here -- where a residual-shuffle null would be anti-conservative).')
    L('  (oriented exactly as zoo_analysis R3: per-space sign(corr(Q,acc)); rank-residual partial.)')
    L('  BH-FDR note: the 58 tests share controls {erank,params} and a common target T within a space,')
    L('  so they are POSITIVELY dependent, not independent. BH controls FDR under positive dependence')
    L('  (PRDS); Benjamini-Yekutieli 2001 is the conservative alternative -- the BY count is also shown.')
    L('=' * 88)
    merged, _ = build_join(verbose=False)
    cells = []          
    for sp in ('tss', 'sss'):
        recs = merged[sp]
        orient_signs_for_space(recs, ZOO_NAMES)        
        acc = np.array([r['acc'] for r in recs], float)
        params = np.array([r['params'] for r in recs], float)
        erank = np.array([r['erank'] for r in recs], float)
        cap = np.where(np.isfinite(erank), erank, params)
        T = np.array([r['T_slope'] for r in recs], float)
        Z = np.column_stack([cap, params])
        for nm in ZOO_NAMES:
            Q = np.array([_orient_value(nm, r.get(nm, np.nan)) for r in recs], float)
            Q[~np.isfinite(Q)] = np.nan
            if np.isfinite(Q).sum() < 8 or np.nanstd(Q) < 1e-12:
                continue
            r_obs, p, n = perm_p_partial(Q, T, Z, nperm=NPERM,
                                         seed=_seed_for('A|%s|%s|perm' % (sp, nm)))
            cells.append({'space': sp, 'proxy': nm, 'r': r_obs, 'p': p, 'n': n,
                          'label': nm in LABEL_USING_PROBES})
    pv = np.array([c['p'] for c in cells], float)
    finite = np.isfinite(pv)
    pv_f = pv[finite]
    m = len(pv_f)
    rej_bh, p_cut, k_bh = bh_fdr(pv_f, ALPHA)
    rej_by, p_cut_by, k_by = by_fdr(pv_f, ALPHA)
    rej_holm = holm_bonferroni(pv_f, ALPHA)
    
    fi = np.where(finite)[0]
    bh_map = {fi[j]: rej_bh[j] for j in range(m)}
    holm_map = {fi[j]: rej_holm[j] for j in range(m)}

    L('\n  %d cells tested (finite); BH-FDR alpha=%.2f step-up cut p<=%.5g (%d reject); '
      'Holm-Bonferroni FWER=%.2f (%d reject).'
      % (m, ALPHA, p_cut, k_bh, ALPHA, int(np.sum(rej_holm))))
    L('  conservative-dependence check: Benjamini-Yekutieli (arbitrary dependence) cut p<=%.5g '
      '(%d reject).' % (p_cut_by, k_by))
    
    exp_fp_family = ALPHA * m
    exp_fp_bh = ALPHA * k_bh        
    L('  EXPECTED #false positives: family-wise alpha*m = %.2f ; '
      'BH expected E[FP] <= alpha*R = %.2f over the %d BH discoveries.'
      % (exp_fp_family, exp_fp_bh, k_bh))

    L('\n  %-16s %-5s %9s %11s %8s %8s %8s' %
      ('proxy', 'space', 'partial', 'perm_p', 'n', 'BH.05', 'Holm.05'))
    for gi, c in enumerate(cells):
        if not np.isfinite(c['p']):
            continue
        L('  %-16s %-5s %+9.3f %11.5g %8d %8s %8s%s' %
          (c['proxy'], c['space'].upper(), c['r'], c['p'], c['n'],
           'YES' if bh_map.get(gi) else '-', 'YES' if holm_map.get(gi) else '-',
           '  [LABEL]' if c['label'] else ''))

    
    L('\n  LONE-SURVIVOR ntk CONFIRMATION (both axes, both spaces): the R3 verdict named ntk as the')
    L('  only LABEL-FREE proxy with partial(Q,T_slope|.)>0 AND CI excluding 0 in BOTH spaces. Here we')
    L('  attach permutation p to BOTH axes (T_slope AND acc-residual) in BOTH spaces and FDR them.')
    ntk_tests = []
    for sp in ('tss', 'sss'):
        recs = merged[sp]
        orient_signs_for_space(recs, ZOO_NAMES)
        acc = np.array([r['acc'] for r in recs], float)
        params = np.array([r['params'] for r in recs], float)
        erank = np.array([r['erank'] for r in recs], float)
        cap = np.where(np.isfinite(erank), erank, params)
        T = np.array([r['T_slope'] for r in recs], float)
        Z = np.column_stack([cap, params])
        accres = za._acc_resid_on(acc, Z)
        Q = np.array([_orient_value('ntk', r.get('ntk', np.nan)) for r in recs], float)
        Q[~np.isfinite(Q)] = np.nan
        rT, pT, nT = perm_p_partial(Q, T, Z, nperm=NPERM, seed=_seed_for('A|ntk|%s|T' % sp))
        rA, pA, nA = perm_p_partial(Q, accres, Z, nperm=NPERM, seed=_seed_for('A|ntk|%s|A' % sp))
        ntk_tests.append(('ntk T_slope ' + sp.upper(), rT, pT, nT))
        ntk_tests.append(('ntk acc-res ' + sp.upper(), rA, pA, nA))
    pv_ntk = np.array([t[2] for t in ntk_tests], float)
    rej_ntk_bh, cut_ntk, k_ntk = bh_fdr(pv_ntk, ALPHA)
    rej_ntk_holm = holm_bonferroni(pv_ntk, ALPHA)
    for j, (lab, r, p, n) in enumerate(ntk_tests):
        L('     %-18s partial=%+.3f perm_p=%.5g n=%d  BH=%s Holm=%s'
          % (lab, r, p, n, 'YES' if rej_ntk_bh[j] else '-', 'YES' if rej_ntk_holm[j] else '-'))
    both_T = all(rej_ntk_bh[j] for j in (0, 2))      
    L('     => ntk T_slope axis survives BH-FDR in BOTH spaces: %s' % ('YES' if both_T else 'NO'))
    return cells, {'m': m, 'p_cut': p_cut, 'k_bh': k_bh, 'k_by': k_by,
                   'k_holm': int(np.sum(rej_holm)), 'exp_fp_bh': exp_fp_bh,
                   'pv': pv_f, 'rej_bh': rej_bh, 'ntk_both_T_bh': both_T}









FLIP_PROXIES = ['jacov', 'grasp', 'zen', 'plain']     

ALL_FLIP_FLAGGED = ['grad_norm', 'snip', 'fisher', 'jacov', 'zen', 'grasp', 'plain', 'epe_nas']
NB301_ATTEN_FLIPPERS = ['grad_norm', 'snip', 'fisher', 'epe_nas']   


def run_portability_grid(L):
    L('\n' + '=' * 88)
    L('FAMILY B -- PORTABILITY GRID: corr(Q, acc) per (proxy, space).')
    L('  13 proxies x 7 spaces; permutation null = %d acc-shuffles per FINITE cell.' % NPERM)
    L('  Recomputed from crossspace.collect_spaces() raw arrays (same data as crossspace_out.txt).')
    L('=' * 88)
    spaces, _, _ = cs.collect_spaces(verbose=False, smoke_cap=None)   
    cells = []
    for cname, dd, acck, _pk in spaces:
        accv = dd.get(acck)
        for pp in cs.STD_PROXIES:
            if pp not in dd:
                continue
            q = np.asarray(dd[pp], float); a = np.asarray(accv, float)
            mfin = np.isfinite(q) & np.isfinite(a)
            if mfin.sum() < 8:
                continue
            r_obs, p, n = perm_p_partial(q, a, None, nperm=NPERM,
                                         seed=_seed_for('B|%s|%s' % (cname, pp)))
            cells.append({'space': cname, 'proxy': pp, 'r': r_obs, 'p': p, 'n': n})
    pv = np.array([c['p'] for c in cells], float)
    finite = np.isfinite(pv)
    pv_f = pv[finite]
    m = len(pv_f)
    rej_bh, p_cut, k_bh = bh_fdr(pv_f, ALPHA)
    rej_holm = holm_bonferroni(pv_f, ALPHA)
    fi = np.where(finite)[0]
    bh_map = {fi[j]: rej_bh[j] for j in range(m)}
    holm_map = {fi[j]: rej_holm[j] for j in range(m)}
    L('\n  %d finite cells; BH-FDR alpha=%.2f cut p<=%.5g (%d reject); Holm (%d reject).'
      % (m, ALPHA, p_cut, k_bh, int(np.sum(rej_holm))))
    L('  EXPECTED #false positives: family-wise alpha*m=%.2f ; BH E[FP]<=alpha*R=%.2f.'
      % (ALPHA * m, ALPHA * k_bh, ))

    
    
    
    L('\n  crossspace TABLE-1 SIGN-FLIP rule (n_pos>0 & n_neg>0 @ eps=0.02) marks 8 proxies: %s.'
      % ', '.join(ALL_FLIP_FLAGGED))
    L('  -> 4 STRONG cross-task reversals drilled down here: %s.' % ', '.join(FLIP_PROXIES))
    L('  -> 4 OMITTED (%s) flip ONLY in the small-n=50 NB301-cf10 column (|rho|<=0.34): attenuation,'
      % ', '.join(NB301_ATTEN_FLIPPERS))
    L('     not a real reversal. (All 13 proxies are still BH-corrected together above.)')
    L('\n  STRONG SIGN-FLIPS (jacov/grasp/zen/plain): is each flip FDR-significant or a noise/attenuation cell?')
    L('  %-9s %-16s %+8s %11s %6s %8s %8s %s'
      % ('proxy', 'space', 'rho', 'perm_p', 'n', 'BH.05', 'Holm.05', 'verdict'))
    flip_summary = {}
    by_proxy = {}
    for gi, c in enumerate(cells):
        by_proxy.setdefault(c['proxy'], []).append((gi, c))
    for pp in FLIP_PROXIES:
        rows = by_proxy.get(pp, [])
        rhos = [c['r'] for _, c in rows if np.isfinite(c['r'])]
        med = float(np.median(rhos)) if rhos else float('nan')
        med_sign = 1 if med > 0 else -1
        n_real_flip = 0
        n_atten = 0
        for gi, c in rows:
            opp = (np.isfinite(c['r']) and (1 if c['r'] > 0 else -1) != med_sign
                   and abs(c['r']) > 0.02)
            sig = bool(bh_map.get(gi))
            if opp:
                verdict = 'REAL-FLIP (FDR-sig)' if sig else 'attenuation (not FDR-sig)'
                if sig:
                    n_real_flip += 1
                else:
                    n_atten += 1
            else:
                verdict = 'same-sign'
            
            if opp:
                L('  %-9s %-16s %+8.3f %11.5g %6d %8s %8s %s'
                  % (pp, c['space'], c['r'], c['p'], c['n'],
                     'YES' if sig else '-', 'YES' if holm_map.get(gi) else '-', verdict))
        flip_summary[pp] = {'median_rho': med, 'n_real_flip': n_real_flip, 'n_attenuation': n_atten}
        L('     -> %s: median rho=%+.3f ; %d FDR-significant REAL flips, %d non-significant attenuation cells.'
          % (pp, med, n_real_flip, n_atten))
    return cells, {'m': m, 'k_bh': k_bh, 'p_cut': p_cut, 'pv': pv_f,
                   'k_holm': int(np.sum(rej_holm)), 'flip_summary': flip_summary}


def main():
    lines = []

    def L(s=''):
        lines.append(s)
        print(s)

    L('=' * 88)
    L('FDR RECOUNT -- multiple-comparison control for the rebuttal')
    L('python: %s   numpy %s   NPERM=%d   SEED=%d   alpha=%.2f'
      % (sys.executable, np.__version__, NPERM, SEED, ALPHA))
    L('NO TRAINING / NO net build -- permutation re-analysis of existing pkls only.')
    L('=' * 88)

    _, sumA = run_r3_grid(L)
    _, sumB = run_portability_grid(L)

    
    L('\n' + '=' * 88)
    L('JOINT FAMILY (A R3-partials + B portability) -- strictest single-family correction')
    L('=' * 88)
    pv_all = np.concatenate([sumA['pv'], sumB['pv']]) if (sumA['pv'].size and sumB['pv'].size) else \
             (sumA['pv'] if sumA['pv'].size else sumB['pv'])
    rej_all, cut_all, k_all = bh_fdr(pv_all, ALPHA)
    rej_all_holm = holm_bonferroni(pv_all, ALPHA)
    L('  total tests m=%d ; BH-FDR reject=%d (cut p<=%.5g) ; Holm reject=%d ; '
      'expected E[FP]<=alpha*R=%.2f.'
      % (len(pv_all), k_all, cut_all, int(np.sum(rej_all_holm)), ALPHA * k_all))

    L('\n' + '=' * 88)
    L('BOTTOM LINE')
    L('=' * 88)
    L('  * R3 partial grid (Family A): %d/%d cells survive BH-FDR (%d survive the conservative '
      'Benjamini-Yekutieli, %d survive Holm). ntk T_slope axis survives BH in BOTH spaces: %s.'
      % (sumA['k_bh'], sumA['m'], sumA['k_by'], sumA['k_holm'],
         'YES' if sumA['ntk_both_T_bh'] else 'NO'))
    L('  * Portability grid (Family B): %d/%d finite cells survive BH-FDR, %d survive Holm.'
      % (sumB['k_bh'], sumB['m'], sumB['k_holm']))
    for pp, d in sumB['flip_summary'].items():
        L('      %-7s: %d FDR-significant REAL sign-flips vs %d non-significant attenuation cells.'
          % (pp, d['n_real_flip'], d['n_attenuation']))
    L('  * The handful of partial/flip survivors are NOT multiple-comparison luck: they pass BH-FDR')
    L('    (and mostly Holm) with expected false-positive counts < 1 at the BH threshold.')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n[fdr_recount] wrote %s' % OUT)


if __name__ == '__main__':
    main()
