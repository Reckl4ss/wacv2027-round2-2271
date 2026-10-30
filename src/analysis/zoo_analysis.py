# Numerical engine; see docs/protocols.md for interpretation and provenance.
import argparse
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))          
THEORY = os.path.dirname(ROOT)                              
RESULTS = os.path.join(THEORY, 'results')                  
PROJ = os.path.dirname(THEORY)                              


BASE_TSS = os.path.join(RESULTS, 'base_tss_cifar10.pkl')
BASE_SSS = os.path.join(RESULTS, 'base_sss_cifar10.pkl')
COMP_PKL = os.path.join(RESULTS, 'exp_alpha_components.pkl')
PREMISE_PKL = os.path.join(RESULTS, 'premise_data.pkl')
PROBES_PKL = os.path.join(RESULTS, 'trainability_probes.pkl')


TSS_PBZ2 = os.path.join(PROJ, 'natsdata', 'NATS-tss-v1_0-3ffb9.pickle.pbz2')
SSS_PBZ2 = os.path.join(PROJ, 'natsdata', 'NATS-sss-v1_0-50262.pickle.pbz2')

SEED = 1234
NBOOT = 1000
SMOKE_N = 40
FULLZOO_MIN = 150          
KFOLD = 5                  
RIDGE_LAMBDA = 1.0         
EMAX = 4                   
EPS = 1e-8


STD_PROXIES = ['fisher', 'grad_norm', 'grasp', 'snip', 'synflow', 'jacob_cov',
               'plain', 'zico', 'meco', 'zen', 'ntk', 'gradsign']
SIGN_PRIOR = {p: +1 for p in STD_PROXIES}
SIGN_PRIOR['ntk'] = -1          


LOG_PROXIES = {'synflow', 'fisher', 'gradsign', 'grasp', 'ntk', 'zico', 'snip',
               'grad_norm', 'meco', 'zen'}


SPEC_PROXIES = ['cond_sum', 'stable_sum', 'Ralpha_0.5', 'Ralpha_2.0', 'Ralpha_4.0',
                'Ralpha_8.0', 'Ralpha_16.0']
SPEC_SIGN_PRIOR = {p: +1 for p in SPEC_PROXIES}
SPEC_SIGN_PRIOR['cond_sum'] = -1     



PROBE_COLS = ['Q_ntk_logcond', 'Q_ntk_logtrace', 'Q_ntk_effrank', 'Q_ntk_loglmax',
              'Q_zico_rand', 'Q_zico_rot', 'Q_coh_rand', 'Q_coh_rot', 'Q_adapt_rot',
              'Q_zico_oracle']
LABEL_USING_PROBES = {'Q_zico_oracle'}












PROBE_SIGN_PRIOR = {
    'Q_ntk_logcond': -1, 'Q_ntk_logtrace': +1, 'Q_ntk_effrank': +1, 'Q_ntk_loglmax': +1,
    'Q_zico_rand': +1, 'Q_zico_rot': +1, 'Q_coh_rand': -1, 'Q_coh_rot': -1,
    'Q_adapt_rot': +1, 'Q_zico_oracle': +1,
}

ORIENT_EPS = 0.02          
SIGN = {}                  





class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()





def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    avg = sums / cnt
    return avg[inv]


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    rx = _rankdata(x[m]); ry = _rankdata(y[m])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _partial_spearman(x, y, Z):
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    if m.sum() < 5:
        return float('nan')
    rx = _rankdata(x[m]); ry = _rankdata(y[m])
    rZ = np.column_stack([_rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(m.sum()), rZ])

    def resid(t):
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        return t - A @ coef
    ex = resid(rx); ey = resid(ry)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float('nan')
    return float(np.corrcoef(ex, ey)[0, 1])


def _rank_r2(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < 5:
        return float('nan')
    ry = _rankdata(y[m])
    rX = np.column_stack([_rankdata(X[m, j]) for j in range(X.shape[1])])
    A = np.column_stack([np.ones(m.sum()), rX])
    coef, *_ = np.linalg.lstsq(A, ry, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ry - pred) ** 2)); ss_tot = float(np.sum((ry - ry.mean()) ** 2))
    if ss_tot <= 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _acc_resid_on(y, X):

    y = np.asarray(y, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    out = np.full(len(y), np.nan)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < 5:
        return out
    ry = _rankdata(y[m])
    rX = np.column_stack([_rankdata(X[m, j]) for j in range(X.shape[1])])
    A = np.column_stack([np.ones(m.sum()), rX])
    coef, *_ = np.linalg.lstsq(A, ry, rcond=None)
    out[m] = ry - A @ coef
    return out


def _seed_for(label):


    h = 1469598103934665603  
    for ch in str(label).encode('utf-8'):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (SEED + (h % 2_000_000_000)) & 0x7FFFFFFF


def _boot_ci(fn, n, seed=SEED, nboot=NBOOT, ci=0.95):
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(nboot):
        idx = rs.randint(0, n, n)
        v = fn(idx)
        if v is not None and math.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return (float('nan'), float('nan'), len(vals))
    lo = float(np.percentile(vals, 100 * (1 - ci) / 2))
    hi = float(np.percentile(vals, 100 * (1 + ci) / 2))
    return (lo, hi, len(vals))















def _kfold_indices(n, k, seed=SEED):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    folds = np.array_split(perm, k)
    return folds


def _train_rank_map(col_tr, col_full):



    s = np.sort(col_tr)
    lo = np.searchsorted(s, col_full, side='left')
    hi = np.searchsorted(s, col_full, side='right')
    return 0.5 * (lo + hi)  


def _resid_on_cap_train(y_tr, y_full, cap_tr, cap_full):


    if cap_tr is None:
        return y_full
    if cap_tr.ndim == 1:
        cap_tr = cap_tr[:, None]; cap_full = cap_full[:, None]
    rc_tr = np.column_stack([_train_rank_map(cap_tr[:, j], cap_tr[:, j])
                             for j in range(cap_tr.shape[1])])
    rc_full = np.column_stack([_train_rank_map(cap_tr[:, j], cap_full[:, j])
                              for j in range(cap_tr.shape[1])])
    A_tr = np.column_stack([np.ones(len(y_tr)), rc_tr])
    coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
    A_full = np.column_stack([np.ones(len(y_full)), rc_full])
    return y_full - A_full @ coef


def cv_rank_r2(target, feats, cap=None, k=KFOLD, lam=RIDGE_LAMBDA, seed=SEED, return_pred=False):




    target = np.asarray(target, float)
    feats = np.asarray(feats, float)
    if feats.ndim == 1:
        feats = feats[:, None]
    fin = np.isfinite(target) & np.all(np.isfinite(feats), axis=1)
    if cap is not None:
        cap = np.asarray(cap, float)
        if cap.ndim == 1:
            cap = cap[:, None]
        fin = fin & np.all(np.isfinite(cap), axis=1)
    m = fin
    if m.sum() < max(2 * k, 10):
        return (float('nan'), None) if return_pred else float('nan')
    y_full = target[m]
    X_full = feats[m]
    cap_full = cap[m] if cap is not None else None
    n = m.sum()
    folds = _kfold_indices(n, k, seed)
    pred = np.full(n, np.nan)          
    yres_oof = np.full(n, np.nan)      
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(k) if g != f])
        if len(tr) < 3 or len(te) < 1:
            continue
        
        rX_tr = np.column_stack([_train_rank_map(X_full[tr, j], X_full[tr, j])
                                 for j in range(X_full.shape[1])])
        rX_te = np.column_stack([_train_rank_map(X_full[tr, j], X_full[te, j])
                                 for j in range(X_full.shape[1])])
        ry_tr = _train_rank_map(y_full[tr], y_full[tr])
        ry_te = _train_rank_map(y_full[tr], y_full[te])
        
        if cap_full is not None:
            cap_tr = cap_full[tr]
            cap_cat = cap_full[np.concatenate([tr, te])]   
            ntr = len(tr)
            
            for j in range(rX_tr.shape[1]):
                col_cat = np.concatenate([rX_tr[:, j], rX_te[:, j]])
                rj = _resid_on_cap_train(rX_tr[:, j], col_cat, cap_tr, cap_cat)
                rX_tr[:, j] = rj[:ntr]; rX_te[:, j] = rj[ntr:]
            
            y_cat = np.concatenate([ry_tr, ry_te])
            yr_cat = _resid_on_cap_train(ry_tr, y_cat, cap_tr, cap_cat)
            ry_tr = yr_cat[:ntr]; ry_te = yr_cat[ntr:]
        yres_oof[te] = ry_te
        mu = rX_tr.mean(axis=0); sd = rX_tr.std(axis=0); sd[sd < 1e-12] = 1.0
        Ztr = (rX_tr - mu) / sd
        Zte = (rX_te - mu) / sd
        ymu = ry_tr.mean()
        A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
        coef = np.linalg.solve(A, Ztr.T @ (ry_tr - ymu))
        pred[te] = ymu + Zte @ coef
    valid = np.isfinite(pred) & np.isfinite(yres_oof)
    if valid.sum() < 5:
        return (float('nan'), None) if return_pred else float('nan')
    yy = yres_oof[valid]; pp = pred[valid]
    ss_res = float(np.sum((yy - pp) ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    if return_pred:
        full_pred = np.full(len(target), np.nan)
        idx_m = np.where(m)[0]
        full_pred[idx_m] = pred
        return r2, full_pred
    return r2





def _load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def _index_by(records, keyname):

    out = {'tss': {}, 'sss': {}}
    ndup = 0
    for r in records:
        sp = r['space']
        try:
            k = int(r[keyname])
        except (ValueError, TypeError):
            continue
        d = out.setdefault(sp, {})
        if k in d:
            ndup += 1
            print('  [_index_by WARNING] duplicate (space=%s, %s=%d) -> overwriting' % (sp, keyname, k))
        d[k] = r
    if ndup:
        print('  [_index_by] total duplicate keys for %r: %d' % (keyname, ndup))
    return out


def build_join(verbose=True):

    base = _load(BASE_TSS) + _load(BASE_SSS)
    comp = _load(COMP_PKL)
    premise = _load(PREMISE_PKL)
    probes = _load(PROBES_PKL)

    bi = _index_by(base, 'i')
    ci = _index_by(comp, 'key')
    pi = _index_by(premise, 'key')
    qi = _index_by(probes, 'key')

    if verbose:
        for sp in ('tss', 'sss'):
            print(f'[join] space={sp}: base={len(bi[sp])} comp={len(ci[sp])} '
                  f'premise={len(pi[sp])} probes={len(qi[sp])}')

    merged = {'tss': [], 'sss': []}
    for sp in ('tss', 'sss'):
        keys = set(bi[sp]) & set(ci[sp]) & set(pi[sp]) & set(qi[sp])
        for k in sorted(keys):
            b, c, p, q = bi[sp][k], ci[sp][k], pi[sp][k], qi[sp][k]
            rec = {'space': sp, 'index': k}
            rec['acc'] = float(p['acc'])
            rec['params'] = float(p['params'])
            rec['erank'] = float(p['erank'])
            rec['T_loss'] = float(p['T_loss'])
            rec['T_slope'] = float(p['T_slope'])
            rec['acc_match'] = bool(p.get('acc_match', True))
            for pr in STD_PROXIES:
                rec[pr] = float(b.get(pr, np.nan))
            rec['flops'] = float(b.get('flops', np.nan))
            rec['cond_sum'] = float(c['cond_sum'])
            rec['stable_sum'] = float(c['stable_sum'])
            for a in (0.5, 2.0, 4.0, 8.0, 16.0):
                rec['Ralpha_%s' % a] = float(c['Ralpha_sum'][a])
            for qc in PROBE_COLS:
                rec[qc] = float(q.get(qc, np.nan))
            merged[sp].append(rec)
        if verbose:
            print(f'[join] space={sp}: FULL-ZOO intersection (base&comp&premise&probes) '
                  f'= {len(merged[sp])}')

    base_only = {'tss': [], 'sss': []}
    for sp in ('tss', 'sss'):
        for k, b in sorted(bi[sp].items()):
            rec = {'space': sp, 'index': k}
            rec['acc'] = float(b.get('testacc', np.nan))
            rec['params'] = float(b.get('params', np.nan))
            rec['flops'] = float(b.get('flops', np.nan))
            cc = ci[sp].get(k)
            rec['erank'] = float(cc['Ralpha_sum'][1.0]) if cc else float('nan')
            for pr in STD_PROXIES:
                rec[pr] = float(b.get(pr, np.nan))
            
            pp = pi[sp].get(k)
            rec['T_loss'] = float(pp['T_loss']) if pp else float('nan')
            rec['T_slope'] = float(pp['T_slope']) if pp else float('nan')
            base_only[sp].append(rec)
        if verbose:
            n_er = sum(1 for r in base_only[sp] if math.isfinite(r['erank']))
            n_T = sum(1 for r in base_only[sp] if math.isfinite(r['T_slope']))
            print(f'[join] space={sp}: STANDARD-proxy base table n={len(base_only[sp])} '
                  f'(erank present {n_er}, T present {n_T})')
    return merged, base_only





def _T_variants(eps, losses, accs):
    if len(losses) < 2:
        return float('nan'), float('nan')
    l0 = losses[0]; lE = losses[-1]
    t_loss = (l0 - lE) / (l0 + EPS) if (math.isfinite(l0) and l0 > 0) else float('nan')
    x = np.asarray(eps, float); ya = np.asarray(accs, float)
    xb = x - x.mean(); denom = float((xb * xb).sum())
    t_slope = float((xb * (ya - ya.mean())).sum() / denom) if denom > 0 else float('nan')
    return float(t_loss), t_slope


def extract_base_T(base_only, space, max_n=None):


    cache = os.path.join(RESULTS, 'base_T_%s.pkl' % space)
    have = {}
    if os.path.exists(cache):
        have = _load(cache)
        print(f'[baseT] {space}: loaded {len(have)} cached T from {cache}', flush=True)
    need = [r['index'] for r in base_only[space]
            if r['index'] not in have and not math.isfinite(r.get('T_slope', np.nan))]
    if max_n is not None:
        need = need[:max_n]
    if need:
        try:
            from nats_bench import create
        except Exception as e:
            print(f'[baseT] {space}: nats_bench unavailable ({e}); skipping extraction.')
            need = []
    if need:
        pbz2 = TSS_PBZ2 if space == 'tss' else SSS_PBZ2
        print(f'[baseT] {space}: building NATS API (~2min) for {len(need)} archs...', flush=True)
        api = create(pbz2, space, fast_mode=False, verbose=False)
        for j, idx in enumerate(need):
            eps, losses, accs = [], [], []
            for ep in range(0, EMAX + 1):
                try:
                    info = api.get_more_info(idx, 'cifar10-valid', iepoch=ep, hp='12', is_random=False)
                    tl = float(info['train-loss']); ta = float(info['train-accuracy'])
                except Exception:
                    break
                if not (math.isfinite(tl) and math.isfinite(ta)):
                    break
                eps.append(ep); losses.append(tl); accs.append(ta)
            have[idx] = _T_variants(eps, losses, accs)
            if (j + 1) % 100 == 0:
                print(f'  [baseT {space}] {j+1}/{len(need)}', flush=True)
        del api
        with open(cache, 'wb') as f:
            pickle.dump(have, f, protocol=2)
        print(f'[baseT] {space}: saved {len(have)} T to {cache}', flush=True)
    for r in base_only[space]:
        if not math.isfinite(r.get('T_slope', np.nan)) and r['index'] in have:
            tl, ts = have[r['index']]
            r['T_loss'] = tl; r['T_slope'] = ts
    return base_only





def orient_signs_for_space(recs, names):



    global SIGN
    acc = np.array([r.get('acc', np.nan) for r in recs], float)
    SIGN = {}
    info = []
    for nm in names:
        raw = np.array([r.get(nm, np.nan) for r in recs], float)
        raw[~np.isfinite(raw)] = np.nan
        vt = np.sign(raw) * np.log1p(np.abs(raw)) if nm in LOG_PROXIES else raw
        rho = _spearman(vt, acc)
        
        
        documented = nm in SIGN_PRIOR or nm in SPEC_SIGN_PRIOR or nm in PROBE_SIGN_PRIOR
        prior = SIGN_PRIOR.get(nm, SPEC_SIGN_PRIOR.get(nm, PROBE_SIGN_PRIOR.get(nm, +1)))
        if math.isfinite(rho) and abs(rho) >= ORIENT_EPS:
            sgn = +1 if rho > 0 else -1
            src = 'data(rho=%+.2f)' % rho
        else:
            sgn = prior
            tag_doc = 'documented' if documented else 'DEFAULT+1 (undocumented)'
            src = 'prior=%+d %s (rho=%s)' % (
                prior, tag_doc, '%.2f' % rho if math.isfinite(rho) else 'nan')
        SIGN[nm] = sgn
        info.append((nm, sgn, src))
    missing = [nm for nm in names if nm not in SIGN]
    assert not missing, 'un-oriented proxies: %s' % missing
    print('  [orient] per-space sign(corr(Q,acc)) (data unless prior-fallback):')
    for nm, sgn, src in info:
        print('        %-16s %+d  %s' % (nm, sgn, src))
    return SIGN


def _orient_value(name, v):
    sgn = SIGN.get(name, SIGN_PRIOR.get(name,
          SPEC_SIGN_PRIOR.get(name, PROBE_SIGN_PRIOR.get(name, +1))))
    if name in LOG_PROXIES:
        v = np.sign(v) * np.log1p(np.abs(v))
    return sgn * v


def proxy_matrix(recs, names):


    cols = []; keep = []; nonfinite_cells = {}
    for nm in names:
        v = np.array([r.get(nm, np.nan) for r in recs], float)
        v[~np.isfinite(v)] = np.nan
        v = _orient_value(nm, v)
        v[~np.isfinite(v)] = np.nan
        nf = int((~np.isfinite(v)).sum())
        if np.isfinite(v).sum() < 5 or np.nanstd(v) < 1e-12:
            continue
        cols.append(v); keep.append(nm); nonfinite_cells[nm] = nf
    if not cols:
        return None, None, [], None, {}
    X = np.column_stack(cols)
    row_mask = np.all(np.isfinite(X), axis=1)
    if row_mask.sum() < 5:
        return None, None, [], None, nonfinite_cells
    Xcc = X[row_mask]
    mu = Xcc.mean(axis=0); sd = Xcc.std(axis=0); sd[sd < 1e-12] = 1.0
    Xz = (Xcc - mu) / sd
    Rcols = []
    for j in range(Xcc.shape[1]):
        rr = _rankdata(Xcc[:, j]); rr = (rr - rr.mean()) / (rr.std() + 1e-12)
        Rcols.append(rr)
    Xr = np.column_stack(Rcols)
    return Xz, Xr, keep, row_mask, nonfinite_cells


def pca_pc1(X, anchor=None):






    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2); vr = var / var.sum()
    scores = U * S; load = Vt.T
    flip = False
    if anchor is not None:
        a = np.asarray(anchor, float)
        if a.shape[0] == scores.shape[0] and np.isfinite(a).sum() >= 3:
            c = _spearman(scores[:, 0], a)
            if math.isfinite(c) and abs(c) > 1e-9:
                flip = (c < 0)
            else:
                flip = (load[:, 0].sum() < 0)
        else:
            flip = (load[:, 0].sum() < 0)
    else:
        flip = (load[:, 0].sum() < 0)
    if flip:
        load[:, 0] *= -1; scores[:, 0] *= -1
    return vr, scores[:, :2], load[:, :2]





def run_R1(recs, names, tag, out):
    print('\n' + '=' * 80)
    print(f'R1 ONE-AXIS + NBSZ REPLICATE-AND-DECOMPOSE -- {tag}  '
          f'(n={len(recs)}, proxies requested={len(names)})')
    print('=' * 80)
    orient_signs_for_space(recs, names)
    Xz, Xr, keep, row_mask, nf_cells = proxy_matrix(recs, names)
    if Xz is None or len(keep) < 2:
        print('  too few usable proxies / complete-case rows; skipping R1.')
        return
    n_drop = int((~row_mask).sum())
    print('  [R1a masking] complete-case PCA: kept %d/%d archs (dropped %d non-finite). '
          'Per-column non-finite cells (masked, NOT zeroed):'
          % (int(row_mask.sum()), len(recs), n_drop))
    nfshow = {k: v for k, v in nf_cells.items() if v > 0}
    print('        %s' % (nfshow if nfshow else 'none (all proxy cells finite)'))

    acc = np.array([r['acc'] for r in recs], float)[row_mask]
    erank = np.array([r['erank'] for r in recs], float)[row_mask]
    params = np.array([r['params'] for r in recs], float)[row_mask]
    res = {'names': keep, 'n': len(recs), 'n_pca': int(row_mask.sum()),
           'n_dropped_nonfinite': n_drop, 'nonfinite_cells': nf_cells}

    
    for mode, X in (('rank-HEADLINE', Xr), ('raw-z-SENSITIVITY', Xz)):
        vr, scores, load = pca_pc1(X, anchor=acc)  
        pc1 = scores[:, 0]
        print(f'\n  [R1a {mode}] PCA: var(PC1)={vr[0]*100:5.1f}%  '
              f'var(PC1+PC2)={(vr[0]+vr[1])*100:5.1f}%  (k_proxies={len(keep)})')
        print(f'  [R1a {mode}] corr(PC1,params)={_spearman(pc1, params):+.3f}  '
              f'corr(PC1,erank)={_spearman(pc1, erank):+.3f}  '
              f'corr(PC1,acc)={_spearman(pc1, acc):+.3f}')
        order = np.argsort(-np.abs(load[:, 0]))
        print(f'  [R1a {mode}] PC1 loadings (sorted |.|):')
        for j in order:
            print(f'        {keep[j]:16s} {load[j, 0]:+.3f}')
        res[mode] = {'var_ratio': vr[:5].tolist(),
                     'pc1_loadings': dict(zip(keep, load[:, 0].tolist())),
                     'corr_pc1_params': _spearman(pc1, params),
                     'corr_pc1_erank': _spearman(pc1, erank),
                     'corr_pc1_acc': _spearman(pc1, acc)}

    
    P = len(keep)
    M = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            M[i, j] = _spearman(Xr[:, i], Xr[:, j])
    offdiag = M[~np.eye(P, dtype=bool)]
    print(f'\n  [R1a] proxy x proxy Spearman (oriented): mean|rho|={np.nanmean(np.abs(offdiag)):.3f}  '
          f'median|rho|={np.nanmedian(np.abs(offdiag)):.3f}  (P={P})')
    res['spearman_matrix'] = {'names': keep, 'matrix': M.tolist(),
                              'mean_abs_offdiag': float(np.nanmean(np.abs(offdiag)))}

    
    
    feat_all = {}
    for nm in keep:
        v = np.array([_orient_value(nm, r.get(nm, np.nan)) for r in recs], float)
        v[~np.isfinite(v)] = np.nan
        feat_all[nm] = v
    acc_all = np.array([r['acc'] for r in recs], float)
    erank_all = np.array([r['erank'] for r in recs], float)
    params_all = np.array([r['params'] for r in recs], float)

    
    print(f'\n  [R1b NBSZ replicate] {KFOLD}-fold CV ridge on per-column RANKS predicting acc.')
    print('       strictly leakage-free: features ranked WITHIN the train fold (test rows mapped '
          'via searchsorted); ridge fit + standardization use TRAIN folds only.')
    single_r2 = {}
    for nm in keep:
        single_r2[nm] = cv_rank_r2(acc_all, feat_all[nm])
    best_single = max((nm for nm in keep if math.isfinite(single_r2[nm])),
                      key=lambda nm: single_r2[nm], default=None)
    best_single_r2 = single_r2.get(best_single, float('nan'))
    Xall = np.column_stack([feat_all[nm] for nm in keep])
    all_r2 = cv_rank_r2(acc_all, Xall)
    gain_abs = all_r2 - best_single_r2
    gain_rel = (gain_abs / best_single_r2 * 100.0) if (math.isfinite(best_single_r2) and best_single_r2 > 0) else float('nan')
    print(f'       best single proxy = {best_single}  CV-R2={best_single_r2:.4f}')
    print(f'       ALL {len(keep)} proxies        CV-R2={all_r2:.4f}')
    print(f'       complementarity GAIN: abs={gain_abs:+.4f}  rel={gain_rel:+.1f}%  '
          f'(their "+42%"-style number)')
    print('       NOTE: best-single baseline is in-sample-selected (argmax CV-R2 over proxies), '
          'so the reported gain is a LOWER BOUND (conservative for the one-axis thesis).')
    res['nbsz'] = {'best_single': best_single, 'best_single_r2': best_single_r2,
                   'all_r2': all_r2, 'gain_abs': gain_abs, 'gain_rel_pct': gain_rel,
                   'single_r2': single_r2}

    
    
    
    
    cap_all = np.where(np.isfinite(erank_all), erank_all, params_all)
    Zcap = np.column_stack([cap_all, params_all])
    single_res_r2 = {nm: cv_rank_r2(acc_all, feat_all[nm], cap=Zcap) for nm in keep}
    best_single_res = max((nm for nm in keep if math.isfinite(single_res_r2[nm])),
                          key=lambda nm: single_res_r2[nm], default=None)
    best_single_res_r2 = single_res_r2.get(best_single_res, float('nan'))
    Xall_res = np.column_stack([feat_all[nm] for nm in keep])
    all_res_r2, res_pred = cv_rank_r2(acc_all, Xall_res, cap=Zcap, return_pred=True)
    gain_res_abs = all_res_r2 - best_single_res_r2
    rel_meaningful = math.isfinite(best_single_res_r2) and best_single_res_r2 > 0
    gain_res_rel = (gain_res_abs / best_single_res_r2 * 100.0) if rel_meaningful else float('nan')
    print(f'\n  [R1c decompose] capacity {{erank,params}} regressed out of acc AND every proxy '
          'INSIDE each CV train fold (train-fit, applied to held-out rows -> no target-touch).')
    print(f'       best single residual proxy = {best_single_res}  CV-R2(res)={best_single_res_r2:.4f}')
    print(f'       ALL proxies residual          CV-R2(res)={all_res_r2:.4f}')
    if rel_meaningful:
        print(f'       SURVIVING complementarity GAIN: abs={gain_res_abs:+.4f}  rel={gain_res_rel:+.1f}%')
    else:
        
        
        print(f'       SURVIVING complementarity GAIN: abs={gain_res_abs:+.4f}  '
              f'rel=n/a (residual baseline R2<=0)')
    res['nbsz_residual'] = {'best_single': best_single_res, 'best_single_r2': best_single_res_r2,
                            'all_r2': all_res_r2, 'gain_abs': gain_res_abs,
                            'gain_rel_pct': gain_res_rel}

    
    T = np.array([r['T_slope'] for r in recs], float)
    transfer = _partial_spearman(res_pred, T, Zcap) if res_pred is not None else float('nan')
    lo = hi = float('nan'); nb = 0
    if res_pred is not None:
        n_full = len(res_pred)
        lo, hi, nb = _boot_ci(lambda ix: _partial_spearman(res_pred[ix], T[ix], Zcap[ix]),
                              n_full, seed=_seed_for('%s|R1d_transfer' % tag))
    print(f'\n  [R1d transfer] partial(residual-surrogate-pred, T_slope | erank,params) = '
          f'{transfer:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]')
    res['nbsz_transfer'] = {'partial_pred_T': transfer, 'ci': [lo, hi]}

    
    print('\n  [R1 INTERPRETATION]')
    small_gain = (math.isfinite(gain_res_rel) and gain_res_rel < 50.0) or \
                 (math.isfinite(gain_res_abs) and gain_res_abs < 0.05)
    no_transfer = not (math.isfinite(transfer) and math.isfinite(lo) and lo > 0)
    if small_gain and no_transfer:
        print('       Residual complementarity is SMALL and does NOT transfer to T '
              '-> the +42%-style gain was CAPACITY DENOISING. Supports one-axis + ceiling.')
    elif not small_gain and not no_transfer:
        print('       Residual complementarity is LARGE and TRANSFERS to T '
              '-> one-axis FAILS (the zoo carries trainability-relevant residual structure).')
    else:
        print('       MIXED: residual gain %s, transfer %s -> see R3 for the per-proxy verdict.'
              % ('large' if not small_gain else 'small',
                 'positive' if not no_transfer else 'absent'))
    out[tag] = res





def run_R2(recs, tag, out):
    print('\n' + '=' * 80)
    print(f'R2 CAPACITY-ANCHORED DECOMPOSITION -- {tag}  (n={len(recs)})')
    print('=' * 80)
    acc = np.array([r['acc'] for r in recs], float)
    params = np.array([r['params'] for r in recs], float)
    erank = np.array([r['erank'] for r in recs], float)
    T = np.array([r.get('T_slope', np.nan) for r in recs], float)
    cap = np.where(np.isfinite(erank), erank, params)
    n = len(recs)
    n_fallback = int(np.sum(~np.isfinite(erank) & np.isfinite(params)))
    if n_fallback:
        print(f'  [capacity] {n_fallback}/{n} rows lack erank -> capacity column uses the params '
              f'FALLBACK for them (heterogeneous: erank for the rest). See STDONLY note.')
    n_dead = int(np.sum(np.isfinite(acc) & (acc <= 10.0 + 1e-6)))
    if n_dead:
        print(f'  [credibility] {n_dead}/{n} archs at the CIFAR-10 chance floor (acc<=10.0): '
              f'tied chance-level block inflates ties in rank fits (kept as data; flagged).')
    Zcp = np.column_stack([cap, params])
    r2_cp = _rank_r2(acc, Zcp)
    lo, hi, _ = _boot_ci(lambda ix: _rank_r2(acc[ix], np.column_stack([cap[ix], params[ix]])), n,
                         seed=_seed_for('%s|R2_cp' % tag))
    print(f'  rankR2(acc | erank,params)      = {r2_cp:.4f}  95%CI[{lo:.3f},{hi:.3f}]')
    r2_cpt = float('nan'); ci2 = [float('nan'), float('nan')]
    if np.isfinite(T).sum() >= 5:
        Zcpt = np.column_stack([cap, params, T])
        r2_cpt = _rank_r2(acc, Zcpt)
        lo2, hi2, _ = _boot_ci(lambda ix: _rank_r2(acc[ix],
                               np.column_stack([cap[ix], params[ix], T[ix]])), n,
                               seed=_seed_for('%s|R2_cpt' % tag))
        ci2 = [lo2, hi2]
        print(f'  rankR2(acc | erank,params,T)    = {r2_cpt:.4f}  95%CI[{lo2:.3f},{hi2:.3f}]')
        
        
        rs_nf = np.random.RandomState(_seed_for('%s|R2_dffloor' % tag))
        nf = []
        for _ in range(300):
            Qn = rs_nf.standard_normal(n)
            inc = _rank_r2(acc, np.column_stack([cap, params, Qn])) - r2_cp
            if math.isfinite(inc):
                nf.append(inc)
        nf = np.array(nf, float)
        nf_mean = float(nf.mean()) if nf.size else float('nan')
        nf_p95 = float(np.percentile(nf, 95)) if nf.size else float('nan')
        share_raw = r2_cpt - r2_cp
        print(f'  TRAINABILITY SHARE (in-sample increment from adding T) = {share_raw:+.4f}')
        print(f'       1-df noise floor (random 3rd predictor): mean={nf_mean:+.5f}  p95={nf_p95:+.5f}')
        print(f'       share above noise floor = {share_raw - nf_mean:+.4f} '
              f'(in-sample; T is a single real predictor)')
    else:
        nf_mean = nf_p95 = share_raw = float('nan')
        print('  (T unavailable for >=5 archs -> trainability-share increment skipped here.)')
    resid = _acc_resid_on(acc, Zcp)
    print(f'  acc-residual on capacity: finite={int(np.isfinite(resid).sum())}/{n}  '
          f'std={np.nanstd(resid):.3f}')
    out[tag] = {'rankR2_acc_cap_params': r2_cp, 'ci_cp': [lo, hi],
                'rankR2_acc_cap_params_T': r2_cpt, 'ci_cpt': ci2,
                'trainability_share': r2_cpt - r2_cp if math.isfinite(r2_cpt) else float('nan'),
                'trainability_share_noise_floor_mean': nf_mean,
                'trainability_share_noise_floor_p95': nf_p95,
                'trainability_share_above_floor': (share_raw - nf_mean)
                    if math.isfinite(r2_cpt) else float('nan'),
                'n': n, 'n_dead': n_dead, 'n_cap_params_fallback': n_fallback}
    return cap, resid





def run_R3(recs, proxy_names, tag, out, target='T_slope'):
    print('\n' + '=' * 80)
    print(f'R3 FULL-ZOO TRAINABILITY-RESIDUAL ACCESS + CEILING -- {tag}  '
          f'(n={len(recs)}, target={target})')
    print('=' * 80)
    orient_signs_for_space(recs, proxy_names)
    acc = np.array([r['acc'] for r in recs], float)
    params = np.array([r['params'] for r in recs], float)
    erank = np.array([r['erank'] for r in recs], float)
    cap = np.where(np.isfinite(erank), erank, params)
    T = np.array([r[target] for r in recs], float)
    Z = np.column_stack([cap, params])
    accres = _acc_resid_on(acc, Z)
    n = len(recs)
    n_fallback = int(np.sum(~np.isfinite(erank) & np.isfinite(params)))
    if n_fallback:
        print(f'  [capacity] {n_fallback}/{n} rows lack erank -> params-fallback in the capacity '
              f'control (heterogeneous; partials only partially capacity-controlled for those rows).')
    n_dead = int(np.sum(np.isfinite(acc) & (acc <= 10.0 + 1e-6)))
    if n_dead:
        print(f'  [credibility] {n_dead}/{n} chance-floor archs (acc<=10.0) in the rank fits.')
    print(f'  finite T={int(np.isfinite(T).sum())}/{n}  '
          f'partial(T,acc|erank,params)={_partial_spearman(T, acc, Z):+.3f} '
          f'(premise sanity; expect ~+0.6)')
    print('  %-16s %9s %20s %20s %12s %6s' %
          ('proxy', 'corr(Q,T)', 'partial(Q,T|cap,pa)', 'partial(Q,accres|..)',
           'corr(Q,er)', 'flag'))
    rows = {}
    for nm in proxy_names:
        Q = np.array([_orient_value(nm, r.get(nm, np.nan)) for r in recs], float)
        Q[~np.isfinite(Q)] = np.nan
        if np.isfinite(Q).sum() < 5 or np.nanstd(Q) < 1e-12:
            continue
        rQT = _spearman(Q, T)
        pQT = _partial_spearman(Q, T, Z)
        plo, phi, _ = _boot_ci(lambda ix: _partial_spearman(Q[ix], T[ix], Z[ix]), n,
                               seed=_seed_for('%s|%s|QT' % (tag, nm)))
        pQR = _partial_spearman(Q, accres, Z)
        rlo, rhi, _ = _boot_ci(lambda ix: _partial_spearman(Q[ix],
                               _acc_resid_on(acc[ix], np.column_stack([cap[ix], params[ix]])),
                               np.column_stack([cap[ix], params[ix]])), n,
                               seed=_seed_for('%s|%s|Qaccres' % (tag, nm)))
        rQe = _spearman(Q, erank)
        flag = 'LABEL' if nm in LABEL_USING_PROBES else ''
        robust_T = math.isfinite(pQT) and pQT > 0 and math.isfinite(plo) and plo > 0
        mark = '*' if robust_T else ''
        print('  %-16s %+9.3f %+9.3f[%+.2f,%+.2f] %+9.3f[%+.2f,%+.2f] %+12.3f %6s%s' %
              (nm, rQT, pQT, plo, phi, pQR, rlo, rhi, rQe, flag, mark))
        rows[nm] = {'corr_QT': rQT, 'partial_QT': pQT, 'partial_QT_ci': [plo, phi],
                    'partial_Qaccres': pQR, 'partial_Qaccres_ci': [rlo, rhi],
                    'corr_Qerank': rQe, 'label_using': nm in LABEL_USING_PROBES,
                    'robust_T_pos': robust_T}

    
    r2_base = _rank_r2(acc, Z)
    best_nm, best_r2 = None, -1.0
    for nm in proxy_names:
        if nm in LABEL_USING_PROBES:
            continue
        Q = np.array([_orient_value(nm, r.get(nm, np.nan)) for r in recs], float)
        Q[~np.isfinite(Q)] = np.nan
        if np.isfinite(Q).sum() < 5:
            continue
        r2q = _rank_r2(acc, np.column_stack([cap, params, Q]))
        if math.isfinite(r2q) and r2q > best_r2:
            best_r2, best_nm = r2q, nm
    r2_oracle = _rank_r2(acc, np.column_stack([cap, params, T]))
    
    K = sum(1 for nm in proxy_names if nm not in LABEL_USING_PROBES)
    rs = np.random.RandomState(SEED)
    floor = []
    for _ in range(200):
        bestn = -1.0
        for _k in range(max(K, 1)):
            Qn = rs.standard_normal(n)
            r2n = _rank_r2(acc, np.column_stack([cap, params, Qn]))
            if math.isfinite(r2n) and r2n > bestn:
                bestn = r2n
        if math.isfinite(bestn):
            floor.append(bestn - r2_base)
    floor = np.array(floor, float)
    fl_mean = float(floor.mean()) if floor.size else float('nan')
    fl_p95 = float(np.percentile(floor, 95)) if floor.size else float('nan')
    print(f'\n  CEILING rankR2(acc):  {{erank,params}}={r2_base:.4f}   '
          f'{{+best label-free proxy={best_nm}}}={best_r2:.4f}   {{+T oracle}}={r2_oracle:.4f}')
    print(f'  NOTE: best_r2 is an IN-SAMPLE MAX over K={K} label-free proxies '
          f'-> optimistically biased upward by selection.')
    print(f'  selection NOISE FLOOR (best-of-K random columns lift): '
          f'mean={fl_mean:+.4f}  p95={fl_p95:+.4f}')
    print(f'  best-proxy lift over capacity = {best_r2 - r2_base:+.4f}  '
          f'(real lift = {best_r2 - r2_base - fl_mean:+.4f} after noise floor)')
    print(f'  gap closed ONLY by ground-truth T = {r2_oracle - best_r2:+.4f} '
          f'(LOWER BOUND: best_r2 is selection-inflated, so true T-only gap is LARGER)')
    out[tag] = {'rows': rows, 'r2_base': r2_base, 'best_proxy': best_nm,
                'r2_best': best_r2, 'r2_oracle': r2_oracle, 'n': n,
                'ceiling_noise_floor_mean': fl_mean, 'ceiling_noise_floor_p95': fl_p95,
                'best_proxy_lift': best_r2 - r2_base, 'n_dead': n_dead}
    return rows





def r3_verdict(r3_out, spaces=('tss', 'sss'), tagfmt='FULLZOO-%s'):
    print('\n' + '=' * 80)
    print('R3 HEADLINE VERDICT: does ANY label-free proxy access the trainability residual')
    print('   robustly (partial(Q,T|erank,params)>0 AND CI excludes 0) in BOTH spaces?')
    print('=' * 80)
    per = {}
    avail = [sp for sp in spaces if (tagfmt % sp) in r3_out]
    for sp in avail:
        rows = r3_out.get(tagfmt % sp, {}).get('rows', {})
        for nm, d in rows.items():
            if d['label_using']:
                continue
            per.setdefault(nm, {})[sp] = d['robust_T_pos']
    winners = [nm for nm, dd in per.items() if avail and all(dd.get(sp, False) for sp in avail)]
    if winners:
        print('  ACCESSIBLE in BOTH spaces (robust +partial):', winners)
        print('  These proxies have positive residual-access intervals under this orientation protocol.')
    else:
        print('  NO label-free proxy is robustly positive in BOTH spaces (%s).' % ','.join(avail))
        print('  This interval pattern does not establish universal inaccessibility.')
    return winners





def run_all(merged, base_only, smoke, do_base_T):
    out = {'R1': {}, 'R2': {}, 'R3': {}}
    spaces = ('tss', 'sss')
    spec_names = ['cond_sum', 'stable_sum'] + ['Ralpha_%s' % a for a in (0.5, 2.0, 4.0, 8.0, 16.0)]
    zoo_names = STD_PROXIES + spec_names + PROBE_COLS

    for sp in spaces:
        recs = merged[sp]
        if smoke and len(recs) > SMOKE_N:
            recs = recs[:SMOKE_N]
        tag = 'FULLZOO-%s' % sp
        print('\n' + '#' * 80)
        print(f'# FULL ZOO  space={sp}  n={len(recs)}')
        print('#' * 80)
        if len(recs) < 10:
            print('  <10 archs; skipping analyses (expected under --smoke if tiny).')
            continue
        run_R1(recs, zoo_names, tag, out['R1'])
        run_R2(recs, tag, out['R2'])
        run_R3(recs, zoo_names, tag, out['R3'], target='T_slope')

    r3_verdict(out['R3'], spaces, 'FULLZOO-%s')

    
    small = [sp for sp in spaces if len(merged[sp]) < FULLZOO_MIN]
    print('\n' + '#' * 80)
    print(f'# STANDARD-13-PROXY-ONLY path at full base n (R1+R2 always; R3 where T available).')
    if small:
        print(f'# (triggered: spaces {small} have full-zoo intersection <{FULLZOO_MIN}.)')
    print('#' * 80)
    for sp in spaces:
        bo = base_only[sp]
        if smoke and len(bo) > SMOKE_N:
            bo = bo[:SMOKE_N]
        tag = 'STDONLY-%s' % sp
        print('\n' + '-' * 80)
        print(f'# STANDARD-ONLY  space={sp}  base n={len(bo)}')
        print('-' * 80)
        if len(bo) < 10:
            print('  <10 archs; skipping.')
            continue
        run_R1(bo, STD_PROXIES, tag, out['R1'])
        run_R2(bo, tag, out['R2'])
        if do_base_T:
            mx = SMOKE_N if smoke else None
            extract_base_T(base_only, sp, max_n=mx)
        bo2 = base_only[sp]
        if smoke and len(bo2) > SMOKE_N:
            bo2 = bo2[:SMOKE_N]
        bo2 = [r for r in bo2 if math.isfinite(r.get('T_slope', np.nan))]
        if len(bo2) >= 10:
            run_R3(bo2, STD_PROXIES, tag, out['R3'], target='T_slope')
        else:
            print(f'  [R3 STDONLY {sp}] only {len(bo2)} archs with finite T; skipping R3 '
                  '(pass --full to extract more via NATS, or rely on FULLZOO R3).')
    if all(('STDONLY-%s' % sp) in out['R3'] for sp in spaces):
        r3_verdict(out['R3'], spaces, 'STDONLY-%s')
    return out


def main():
    ap = argparse.ArgumentParser(description='R1-R3 zoo-dimensionality / trainability-access analysis.')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--smoke', action='store_true', help='Small slice; verify joins + analyses run.')
    g.add_argument('--full', action='store_true', help='All archs; save out.txt + results pkl.')
    ap.add_argument('--no-base-t', action='store_true',
                    help='Skip the NATS base-T extraction in the STDONLY R3 path.')
    args = ap.parse_args()
    if not args.full:
        args.smoke = True

    np.random.seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    txt_path = os.path.join(RESULTS, 'zoo_analysis_out.txt')
    tee = None
    real_stdout = sys.stdout
    if args.full:
        tee = open(txt_path, 'w', encoding='utf-8')
        sys.stdout = _Tee(real_stdout, tee)

    try:
        print('=' * 80)
        print('PYTHON:', sys.executable)
        print('numpy', np.__version__)
        print('MODE:', 'SMOKE' if args.smoke else 'FULL', '  SEED:', SEED)
        print('=' * 80)

        merged, base_only = build_join(verbose=True)
        
        do_base_T = args.full and (not args.no_base_t)
        out = run_all(merged, base_only, smoke=args.smoke, do_base_T=do_base_T)

        if args.full:
            outpkl = os.path.join(RESULTS, 'zoo_analysis_results.pkl')
            with open(outpkl, 'wb') as f:
                pickle.dump(out, f, protocol=2)
            print(f'\n[full] saved results pkl -> {outpkl}')
            print(f'[full] saved text report -> {txt_path}')
        else:
            print('\n[smoke] OK: joins verified + R1/R2/R3 ran on the small slice.')
    finally:
        if tee is not None:
            sys.stdout = real_stdout
            tee.close()


if __name__ == '__main__':
    main()
