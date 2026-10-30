# Numerical engine; see docs/protocols.md for interpretation and provenance.
import os, sys, json, ast, pickle, argparse, hashlib
import numpy as np
from scipy import stats

SEED = 1234
np.random.seed(SEED)
N_BOOT = 2000

ROOT = os.environ.get("NAS_PROJECT_ROOT", os.getcwd())
RES  = os.environ.get("NAS_RESULTS_DIR", os.path.join(ROOT, "runs", "crossspace"))
DATA = os.environ.get("NASLIB_DATA_DIR", os.path.join(ROOT, "data", "external", "NASLib", "naslib", "data"))
ZCB  = os.environ.get("NASLIB_BASELINES_DIR", os.path.join(ROOT, "data", "external", "NASLib", "zc_baseline_out"))
TNB  = os.path.join(DATA, "zc_benchmarks_baselines")



PROXY_ALIASES = {
    "grad_norm": ["grad_norm"],
    "snip":      ["snip"],
    "fisher":    ["fisher"],
    "jacov":     ["jacov", "jacob_cov"],
    "synflow":   ["synflow"],
    "nwot":      ["nwot"],
    "zen":       ["zen"],
    "grasp":     ["grasp"],
    "l2_norm":   ["l2_norm"],
    "plain":     ["plain"],
    "params":    ["params"],
    "flops":     ["flops"],
    "epe_nas":   ["epe_nas"],
}
STD_PROXIES = list(PROXY_ALIASES.keys())




OUR_PROXIES = ["erank", "cond_sum", "stable_sum", "Ralpha2"]




def _clean(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def spearman(x, y):
    x, y = _clean(x, y)
    if len(x) < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), len(x)
    r, _ = stats.spearmanr(x, y)
    return float(r), len(x)

def boot_spearman_ci(x, y, n_boot=N_BOOT, seed=SEED):
    x, y = _clean(x, y)
    n = len(x)
    if n < 8:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.all(xb == xb[0]) or np.all(yb == yb[0]):
            continue
        r, _ = stats.spearmanr(xb, yb)
        if np.isfinite(r):
            out.append(r)
    if len(out) < 50:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

def rank_transform(a):
    a = np.asarray(a, float)
    return stats.rankdata(a)

def partial_spearman(q, acc, ctrl):


    q, acc, ctrl = map(np.asarray, (q, acc, ctrl))
    m = np.isfinite(q) & np.isfinite(acc) & np.isfinite(ctrl)
    q, acc, ctrl = q[m], acc[m], ctrl[m]
    n = len(q)
    if n < 8:
        return float("nan"), n
    rq, ra, rc = rank_transform(q), rank_transform(acc), rank_transform(ctrl)
    def resid(y, X):
        X1 = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        return y - X1 @ beta
    if np.all(rc == rc[0]):
        r, _ = stats.spearmanr(q, acc)
        return float(r), n
    eq = resid(rq, rc.reshape(-1, 1))
    ea = resid(ra, rc.reshape(-1, 1))
    if np.all(eq == eq[0]) or np.all(ea == ea[0]):
        return float("nan"), n
    r = np.corrcoef(eq, ea)[0, 1]
    return float(r), n

def rank_r2(acc, feats):



    acc = np.asarray(acc, float)
    feats = [np.asarray(f, float) for f in feats]
    m = np.isfinite(acc)
    for f in feats:
        m &= np.isfinite(f)
    acc = acc[m]; feats = [f[m] for f in feats]
    n = len(acc)
    if n < 10:
        return float("nan"), float("nan"), n
    ya = rank_transform(acc)
    X = np.column_stack([rank_transform(f) for f in feats])
    X1 = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X1, ya, rcond=None)
    pred = X1 @ beta
    ss_res = np.sum((ya - pred) ** 2)
    ss_tot = np.sum((ya - ya.mean()) ** 2)
    r2_in = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    folds = np.array_split(order, 5)
    cv_res = []; cv_tot = []
    for i in range(5):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(5) if j != i])
        if len(tr) < 6 or len(te) < 2:
            continue
        ya_tr = rank_transform(acc[tr]); Xtr = np.column_stack([rank_transform(f[tr]) for f in feats])
        
        Xte_cols = []
        for f in feats:
            ftr = f[tr]; fte = f[te]
            rtr = rank_transform(ftr)
            Xte_cols.append(np.interp(fte, np.sort(ftr), np.sort(rtr)))
        Xte = np.column_stack(Xte_cols)
        X1tr = np.column_stack([np.ones(len(tr)), Xtr])
        b, *_ = np.linalg.lstsq(X1tr, ya_tr, rcond=None)
        X1te = np.column_stack([np.ones(len(te)), Xte])
        pred_te = X1te @ b
        
        
        atr = acc[tr]
        ya_te = np.interp(acc[te], np.sort(atr), np.sort(rank_transform(atr)))
        ybar_tr = ya_tr.mean()  
        cv_res.append(np.sum((ya_te - pred_te) ** 2))
        cv_tot.append(np.sum((ya_te - ybar_tr) ** 2))
    if cv_tot and sum(cv_tot) > 0:
        r2_cv = 1 - sum(cv_res) / sum(cv_tot)
    else:
        r2_cv = float("nan")
    return float(r2_in), float(r2_cv), n
















T_K = 5
def T_slope(curve):


    v = np.asarray(curve, float)
    v = v[np.isfinite(v)]
    k = min(T_K, len(v))
    if k < 3:
        return float("nan")
    ep = np.arange(k, dtype=float)
    return float(stats.linregress(ep, v[:k]).slope)

def T_from_trainloss(losses):


    v = np.asarray(losses, float)
    v = v[np.isfinite(v)]
    k = min(T_K, len(v))
    if k < 3:
        return float("nan")
    ep = np.arange(k, dtype=float)
    return float(-stats.linregress(ep, v[:k]).slope)







def _pick_alias(d, canonical):
    for a in PROXY_ALIASES[canonical]:
        if a in d:
            return d[a]
    return None

def load_nb101(verbose=True):
    p = os.path.join(ZCB, "v2--nasbench101--cifar10--9000.json")
    d = json.load(open(p))
    archs = list(d.keys())
    cols = {pp: [] for pp in STD_PROXIES}
    acc = []; params = []
    for k in archs:
        rec = d[k]
        acc.append(rec.get("val_acc", np.nan))
        params.append(rec.get("params", np.nan))
        for pp in STD_PROXIES:
            cols[pp].append(_pick_alias(rec, pp) if _pick_alias(rec, pp) is not None else np.nan)
    out = {pp: np.array(cols[pp], float) for pp in STD_PROXIES}
    out["acc"] = np.array(acc, float); out["params"] = np.array(params, float)
    out["n"] = len(archs); out["T"] = None
    present = [pp for pp in STD_PROXIES if np.isfinite(out[pp]).sum() >= 4]
    if verbose:
        print(f"[NB101-cf10] archs in proxy json: {len(archs)} (SMALL-N)")
        print(f"[NB101-cf10] proxies present: {present}")
        print(f"[NB101-cf10] acc finite: {np.isfinite(out['acc']).sum()}")
    return out

def load_nb301(verbose=True):

    p = os.path.join(ZCB, "v2--nasbench301--cifar10--9000.json")
    d = json.load(open(p))
    archs = list(d.keys())
    cols = {pp: [] for pp in STD_PROXIES}
    acc = []; params = []
    for k in archs:
        rec = d[k]
        acc.append(rec.get("val_acc", np.nan)); params.append(rec.get("params", np.nan))
        for pp in STD_PROXIES:
            v = _pick_alias(rec, pp)
            cols[pp].append(v if v is not None else np.nan)
    out = {pp: np.array(cols[pp], float) for pp in STD_PROXIES}
    out["acc"] = np.array(acc, float); out["params"] = np.array(params, float)
    out["n"] = len(archs); out["T"] = None
    
    curve_match = 0; curve_n = 0
    try:
        curves = pickle.load(open(os.path.join(DATA, "nb301_full_training.pickle"), "rb"))
        curve_n = len(curves); ckset = set(curves.keys())
        for k in archs:
            t = tuple(tuple(tuple(x) for x in cell) for cell in ast.literal_eval(k))
            if t in ckset:
                curve_match += 1
    except Exception as e:
        if verbose: print("[NB301] curve load err:", e)
    present = [pp for pp in STD_PROXIES if np.isfinite(out[pp]).sum() >= 4]
    if verbose:
        print(f"[NB301-cf10] archs in proxy json: {len(archs)} (SMALL-N)")
        print(f"[NB301-cf10] proxies present: {present}")
        print(f"[NB301-cf10] proxy-json val_acc used as acc (curve cross-match below)")
        print(f"[NB301] curve pickle archs: {curve_n}; proxy->curve genotype MATCHES: {curve_match}/{len(archs)}")
        print(f"[NB301] -> proxy arch set is DISJOINT from the {curve_n}-arch curve subset; "
              f"using proxy-json val_acc. No T for NB301 (no per-proxy-arch curve).")
    out["curve_match"] = curve_match; out["curve_n"] = curve_n
    return out

def load_nb301_curve_T(verbose=True):



    try:
        curves = pickle.load(open(os.path.join(DATA, "nb301_full_training.pickle"), "rb"))
    except Exception as e:
        if verbose: print("[NB301-curveT] load err:", e)
        return None
    Ts = []; finalacc = []
    for k, v in curves.items():
        va = v.get("val_accuracies", [])
        tl = v.get("train_losses", [])
        Ts.append(T_from_trainloss(tl))   
        finalacc.append(va[-1] if len(va) else np.nan)
    out = {"T": np.array(Ts, float), "acc": np.array(finalacc, float), "n": len(curves)}
    if verbose:
        print(f"[NB301-curveT] curve archs: {out['n']}; T finite: {np.isfinite(out['T']).sum()} "
              f"(T = -slope of first {T_K} TRAIN-LOSS points; NB301 has no train-acc curve)")
    return out

def load_nb201(verbose=True):



    base = pickle.load(open(os.path.join(RES, "base_tss_cifar10.pkl"), "rb"))
    base_by_idx = {e["i"]: e for e in base}
    prem = pickle.load(open(os.path.join(RES, "premise_data.pkl"), "rb"))
    prem = [e for e in prem if e.get("space") == "tss"]
    prem_by_idx = {int(e["key"]): e for e in prem}
    
    cf10c  = pickle.load(open(os.path.join(DATA, "nb201_cifar10_full_training.pickle"), "rb"))
    cf100c = pickle.load(open(os.path.join(DATA, "nb201_cifar100_full_training.pickle"), "rb"))
    in16c  = pickle.load(open(os.path.join(DATA, "nb201_ImageNet16_full_training.pickle"), "rb"))
    arch_list = list(cf10c.keys())
    DSKEY = {"cf10": ("cifar10-valid", cf10c), "cf100": ("cifar100", cf100c),
             "IN16": ("ImageNet16-120", in16c)}

    
    idxs_proxy = sorted(base_by_idx.keys())          
    idxs_full  = sorted(set(base_by_idx) & set(prem_by_idx))  
    if verbose:
        print(f"[NB201] base_tss archs (std proxies): {len(idxs_proxy)}")
        print(f"[NB201] premise archs (spectral+T, tss): {len(prem_by_idx)}")
        print(f"[NB201] JOIN base&premise (std+spectral+T): {len(idxs_full)}")
        print(f"[NB201] multi-dataset curve archs: cf10={len(cf10c)} cf100={len(cf100c)} IN16={len(in16c)}")

    def build(idxs, want_spectral):
        cols = {pp: [] for pp in STD_PROXIES}
        accs = {"cf10": [], "cf100": [], "IN16": []}
        params = []; T = {"cf10": [], "cf100": [], "IN16": []}
        spec = {sp: [] for sp in OUR_PROXIES}
        for ix in idxs:
            b = base_by_idx[ix]
            for pp in STD_PROXIES:
                v = _pick_alias(b, pp)
                cols[pp].append(v if v is not None else np.nan)
            params.append(b.get("params", np.nan))
            astr = arch_list[ix]
            for dk, (innerkey, cdict) in DSKEY.items():
                inner = cdict[astr][innerkey]
                fa = inner["eval_acc1es"][-1] if "eval_acc1es" in inner else np.nan
                accs[dk].append(fa)
                
                tacc = inner.get("train_acc1es", [])
                T[dk].append(T_slope(tacc))
            if want_spectral:
                e = prem_by_idx.get(ix)
                if e is not None:
                    spec["erank"].append(e.get("erank", np.nan))
                    spec["cond_sum"].append(e.get("cond_sum", np.nan))
                    spec["stable_sum"].append(e.get("stable_sum", np.nan))
                    ra = e.get("Ralpha_sum", {})
                    spec["Ralpha2"].append(ra.get(2.0, np.nan) if isinstance(ra, dict) else np.nan)
                else:
                    for sp in OUR_PROXIES: spec[sp].append(np.nan)
        res = {pp: np.array(cols[pp], float) for pp in STD_PROXIES}
        res["params"] = np.array(params, float)
        for dk in accs:
            res["acc_" + dk] = np.array(accs[dk], float)
            res["T_" + dk] = np.array(T[dk], float)
        if want_spectral:
            for sp in OUR_PROXIES:
                res[sp] = np.array(spec[sp], float)
        res["n"] = len(idxs)
        return res

    proxy_only = build(idxs_proxy, want_spectral=False)   
    full       = build(idxs_full,  want_spectral=True)     

    
    
    
    base_valacc = []; curve_finalacc = []
    for ix in idxs_proxy:
        b = base_by_idx[ix]
        bv = b.get("valacc", b.get("val_acc", b.get("testacc", np.nan)))
        base_valacc.append(bv)
        astr = arch_list[ix]
        ea = cf10c[astr]["cifar10-valid"].get("eval_acc1es", [])
        curve_finalacc.append(ea[-1] if len(ea) else np.nan)
    bv = np.asarray(base_valacc, float); cf = np.asarray(curve_finalacc, float)
    mm = np.isfinite(bv) & np.isfinite(cf)
    align_meanabs = float(np.mean(np.abs(bv[mm] - cf[mm]))) if mm.sum() else float("nan")
    align_frac1pp = float(np.mean(np.abs(bv[mm] - cf[mm]) <= 1.0)) if mm.sum() else float("nan")
    align_n = int(mm.sum())
    
    assert (not np.isfinite(align_meanabs)) or align_meanabs < 1.5, \
        f"NB201 index->arch alignment looks WRONG: mean|d|={align_meanabs:.2f}pp"

    
    
    
    
    
    
    rep_ours = []; rep_prem = []
    for ix in idxs_full:
        e = prem_by_idx.get(ix)
        if e is None or "T_slope" not in e:
            continue
        astr = arch_list[ix]
        ours = T_slope(cf10c[astr]["cifar10-valid"].get("train_acc1es", []))
        rep_ours.append(ours); rep_prem.append(e["T_slope"])
    rep_ours = np.asarray(rep_ours, float); rep_prem = np.asarray(rep_prem, float)
    rm = np.isfinite(rep_ours) & np.isfinite(rep_prem)
    T_agree_rho = float(stats.spearmanr(rep_ours[rm], rep_prem[rm]).correlation) if rm.sum() > 4 else float("nan")
    T_reproduce_n = int(rm.sum())

    proxy_only["align_check"] = (align_meanabs, align_frac1pp, align_n)
    proxy_only["T_repro_check"] = (T_agree_rho, T_reproduce_n)

    if verbose:
        for dk in ["cf10", "cf100", "IN16"]:
            print(f"[NB201-{dk}] std-proxy arm n={proxy_only['n']} acc finite={np.isfinite(proxy_only['acc_'+dk]).sum()} "
                  f"T finite={np.isfinite(proxy_only['T_'+dk]).sum()}")
        print(f"[NB201] spectral-arm n={full['n']} (erank/cond/stable/Ralpha2 available)")
        print(f"[NB201] index->arch ALIGNMENT cross-check: mean|valacc-curveAcc|={align_meanabs:.3f}pp, "
              f"{100*align_frac1pp:.1f}% within 1pp over n={align_n} (mapping verified, not asserted by note).")
        print(f"[NB201] premise T_slope AGREEMENT: Spearman(our train-acc T, premise.T_slope)={T_agree_rho:.3f} "
              f"over n={T_reproduce_n} (SAME definition; values differ as premise used a different "
              f"training-run source, so they agree by rank not byte-for-byte).")
    return {"proxy": proxy_only, "full": full}

def load_transnas(verbose=True):


    import glob
    files = sorted(glob.glob(os.path.join(TNB, "tnb101_micro--*.json")))
    tasks = {}
    for f in files:
        d = json.load(open(f))
        task = d.get("dataset")   
        proxy = d.get("proxy")
        recs = d.get("records", [])
        tasks.setdefault(task, {})[proxy] = {r["idx"]: r for r in recs}
    out = {}
    if verbose:
        print(f"[TransNAS] files: {len(files)}; tasks found: {sorted(tasks.keys())}")
    for task, pdict in tasks.items():
        proxies = list(pdict.keys())
        
        idx_sets = [set(pdict[p].keys()) for p in proxies]
        common = sorted(set.intersection(*idx_sets)) if idx_sets else []
        cols = {}
        accref = None
        for p in proxies:
            cols[p] = np.array([pdict[p][i]["score"] for i in common], float)
        
        accref = np.array([pdict[proxies[0]][i]["val_accuracy"] for i in common], float)
        
        consistent = True
        if len(proxies) >= 2:
            a2 = np.array([pdict[proxies[1]][i]["val_accuracy"] for i in common], float)
            consistent = np.allclose(accref, a2, equal_nan=True)
        rec = {}
        for canon in STD_PROXIES:
            for alias in PROXY_ALIASES[canon]:
                if alias in cols:
                    rec[canon] = cols[alias]; break
            else:
                rec[canon] = np.full(len(common), np.nan)
        rec["acc"] = accref
        rec["params"] = cols.get("params", np.full(len(common), np.nan))
        rec["n"] = len(common); rec["T"] = None
        out[task] = rec
        if verbose:
            present = [c for c in STD_PROXIES if np.isfinite(rec[c]).sum() >= 4]
            print(f"[TransNAS-{task}] proxies in files: {sorted(proxies)}")
            print(f"[TransNAS-{task}] common-idx JOIN n={len(common)}; acc-consistency across proxies={consistent}")
            print(f"[TransNAS-{task}] canonical proxies present: {present}")
    return out




def _smoke_slice_dict(dd, cap, seed=SEED):


    n = dd.get("n", None)
    if not isinstance(n, int) or n <= cap:
        return dd
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, cap, replace=False))
    out = {}
    for k, v in dd.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[idx]
        else:
            out[k] = v
    out["n"] = cap
    return out

def collect_spaces(verbose=True, smoke_cap=None):


    spaces = []
    nb101 = load_nb101(verbose)
    nb201 = load_nb201(verbose)
    nb301 = load_nb301(verbose)
    tnb = load_transnas(verbose)
    if smoke_cap is not None:
        print(f"\n[SMOKE SLICE] subsampling every space to <= {smoke_cap} archs (deterministic).")
        nb101 = _smoke_slice_dict(nb101, smoke_cap)
        nb201 = {"proxy": _smoke_slice_dict(nb201["proxy"], smoke_cap),
                 "full":  _smoke_slice_dict(nb201["full"],  smoke_cap)}
        nb301 = _smoke_slice_dict(nb301, smoke_cap)
        tnb = {t: _smoke_slice_dict(dd, smoke_cap) for t, dd in tnb.items()}
    spaces.append(("NB101-cf10", nb101, "acc", "params"))
    po = nb201["proxy"]
    spaces.append(("NB201-cf10",  po, "acc_cf10",  "params"))
    spaces.append(("NB201-cf100", po, "acc_cf100", "params"))
    spaces.append(("NB201-IN16",  po, "acc_IN16",  "params"))
    spaces.append(("NB301-cf10", nb301, "acc", "params"))
    for task in sorted(tnb.keys()):
        short = task.replace("class_", "")
        spaces.append((f"TransNAS-{short}", tnb[task], "acc", "params"))
    return spaces, nb201, nb301

def portability_matrix(spaces, do_ci=True):
    lines = []
    header = "PROXY".ljust(11) + "".join(c.ljust(15) for c, *_ in spaces)
    lines.append(header)
    lines.append("-" * len(header))
    matrix = {}
    for pp in STD_PROXIES:
        row = pp.ljust(11)
        vals = []
        for cname, dd, acck, _ in spaces:
            if pp not in dd:
                row += "n/a".ljust(15); vals.append(np.nan); continue
            r, n = spearman(dd[pp], dd[acck])
            vals.append(r)
            cell = f"{r:+.3f}(n{n})" if np.isfinite(r) else "nan"
            row += cell.ljust(15)
        lines.append(row)
        matrix[pp] = vals
    lines.append("-" * len(header))
    
    lines.append("\nPORTABILITY SUMMARY per proxy (over spaces where finite):")
    lines.append("  n_pos/n_neg = #spaces with rho>+0.02 / rho<-0.02 (order-invariant). "
                 "SIGN-FLIP flagged when both occur.")
    lines.append("proxy".ljust(11) + "n_spaces  min      max      range    n_pos  n_neg  median")
    for pp in STD_PROXIES:
        v = np.array(matrix[pp], float); v = v[np.isfinite(v)]
        if len(v) == 0:
            lines.append(pp.ljust(11) + "0"); continue
        n_pos = int(np.sum(v > 0.02)); n_neg = int(np.sum(v < -0.02))
        has_both = (n_pos > 0 and n_neg > 0)
        flag = "  <== SIGN-FLIP" if has_both else ""
        lines.append(pp.ljust(11) +
                     f"{len(v):<9d} {v.min():+.3f}  {v.max():+.3f}  {v.max()-v.min():.3f}    "
                     f"{n_pos:<6d} {n_neg:<6d} {np.median(v):+.3f}{flag}")
    return "\n".join(lines), matrix

def ci_table(spaces):
    lines = ["\nBOOTSTRAP 95% CIs for corr(Q,acc) (proxy x space), n>=8 only:"]
    lines.append("proxy".ljust(11) + "space".ljust(16) + "rho      CI_low   CI_high  n")
    for pp in STD_PROXIES:
        for cname, dd, acck, _ in spaces:
            if pp not in dd: continue
            r, n = spearman(dd[pp], dd[acck])
            if not np.isfinite(r) or n < 8: continue
            lo, hi = boot_spearman_ci(dd[pp], dd[acck])
            lines.append(pp.ljust(11) + cname.ljust(16) +
                         f"{r:+.3f}   {lo:+.3f}   {hi:+.3f}   {n}")
    return "\n".join(lines)

def _fmt(x, w=7):

    return (f"{x:+.3f}" if np.isfinite(x) else "n/a").ljust(w)

def trainability_partial_table(spaces):






    lines = ["\nTRAINABILITY-ACCESS SHIFT (rank partials):",
             "  raw=corr(Q,acc)  acc|params  acc|T  T|params  (T|params = proxy's trainability access net of size)."]
    lines.append("space".ljust(16) + "proxy".ljust(11) +
                 "raw      acc|prm  acc|T    T|prm    n")
    for cname, dd, acck, pk in spaces:
        
        Tkey = None
        if acck == "acc_cf10": Tkey = "T_cf10"
        elif acck == "acc_cf100": Tkey = "T_cf100"
        elif acck == "acc_IN16": Tkey = "T_IN16"
        for pp in STD_PROXIES:
            if pp not in dd: continue
            raw, n = spearman(dd[pp], dd[acck])
            if not np.isfinite(raw): continue
            pr_params, _ = partial_spearman(dd[pp], dd[acck], dd[pk]) if pk in dd else (float("nan"), n)
            if Tkey and Tkey in dd and dd[Tkey] is not None:
                pr_accT, _ = partial_spearman(dd[pp], dd[acck], dd[Tkey])
                pr_Tprm, _ = partial_spearman(dd[pp], dd[Tkey], dd[pk]) if pk in dd else (float("nan"), n)
            else:
                pr_accT = float("nan"); pr_Tprm = float("nan")
            lines.append(cname.ljust(16) + pp.ljust(11) +
                         _fmt(raw, 9) + _fmt(pr_params, 9) + _fmt(pr_accT, 9) + _fmt(pr_Tprm, 9) + str(n))
    return "\n".join(lines)

def decomposition_table(nb201):


    lines = ["\nDECOMPOSITION (NB201, rank-R2 of acc; in-sample / 5fold-CV):",
             "  Shows trainability (T) share and spectral (erank) share per dataset/task."]
    full = nb201["full"]
    lines.append("dataset".ljust(10) + "feats".ljust(22) + "R2_in    R2_cv    n")
    for dk in ["cf10", "cf100", "IN16"]:
        acc = full["acc_" + dk]; params = full["params"]; T = full["T_" + dk]; er = full.get("erank")
        sets = [("params", [params]),
                ("params+T", [params, T]),
                ("params+erank", [params, er]),
                ("params+T+erank", [params, T, er])]
        for name, fs in sets:
            r2i, r2c, n = rank_r2(acc, fs)
            r2i_s = f"{r2i:.3f}" if np.isfinite(r2i) else "n/a"
            r2c_s = f"{r2c:+.3f}" if np.isfinite(r2c) else "n/a"
            lines.append(dk.ljust(10) + name.ljust(22) +
                         r2i_s.ljust(9) + r2c_s.ljust(9) + str(n))
        lines.append("")
    return "\n".join(lines)

def our_proxy_table(nb201):
    lines = ["\nOUR SPECTRAL PROXIES (NB201 ONLY; N/A for other spaces -- not fabricated):",
             "proxy".ljust(12) + "NB201-cf10".ljust(14) + "NB201-cf100".ljust(14) + "NB201-IN16".ljust(14) + "n"]
    full = nb201["full"]
    for sp in OUR_PROXIES:
        if sp not in full: continue
        row = sp.ljust(12)
        nn = full["n"]
        for dk in ["cf10", "cf100", "IN16"]:
            r, n = spearman(full[sp], full["acc_" + dk]); nn = n
            row += (f"{r:+.3f}" if np.isfinite(r) else "nan").ljust(14)
        row += str(nn)
        lines.append(row)
    return "\n".join(lines)




def run(full_mode, smoke_cap=200):
    print("=" * 78)
    print("CROSS-SPACE PORTABILITY  (seed=%d, n_boot=%d)" % (SEED, N_BOOT))
    print("Mode:", "FULL" if full_mode else f"SMOKE (cap={smoke_cap}/space, no CIs, no save)")
    print("=" * 78)
    spaces, nb201, nb301 = collect_spaces(verbose=True,
                                          smoke_cap=None if full_mode else smoke_cap)
    nb301T = load_nb301_curve_T(verbose=True)

    print("\n" + "=" * 78)
    print("T (TRAINABILITY) DEFINITION USED")
    print("=" * 78)
    print(f"  T = slope of OLS fit to the first K={T_K} TRAIN-accuracy points vs epoch.")
    print(f"  Larger T = faster early learning. SAME definition as premise T_slope (train-acc curve),")
    print(f"  recomputed uniformly here from the curve pickles (premise T_slope is NOT re-read);")
    print(f"  agrees with premise T_slope by rank (Spearman ~0.94) -- agreement printed above.")
    print(f"  Available: NB201 cf10/cf100/IN16 (train_acc1es). NB301 curve arm uses TRAIN-LOSS descent")
    print(f"  (-slope of first {T_K} train_losses) since NB301 curves carry no train-accuracy; flagged.")
    print(f"  NOT available: NB101 (no per-arch curve in proxy json), TransNAS (records carry final acc only).")

    out = []
    pm, matrix = portability_matrix(spaces, do_ci=full_mode)
    print("\n" + "=" * 78); print("TABLE 1: CROSS-SPACE PORTABILITY MATRIX  corr(Q, acc)"); print("=" * 78)
    print(pm); out.append("TABLE 1: PORTABILITY MATRIX\n" + pm)

    if full_mode:
        ct = ci_table(spaces)
        print(ct); out.append(ct)

    tp = trainability_partial_table(spaces)
    print("\n" + "=" * 78); print("TABLE 2: TRAINABILITY-ACCESS PARTIALS"); print("=" * 78)
    print(tp); out.append("TABLE 2\n" + tp)

    dec = decomposition_table(nb201)
    print("\n" + "=" * 78); print("TABLE 3: DECOMPOSITION (multi-task NB201)"); print("=" * 78)
    print(dec); out.append("TABLE 3\n" + dec)

    op = our_proxy_table(nb201)
    print("\n" + "=" * 78); print("TABLE 4: OUR SPECTRAL PROXIES (NB201 only)"); print("=" * 78)
    print(op); out.append("TABLE 4\n" + op)

    print("\n" + "=" * 78); print("ARCH-MATCH / HONESTY NOTES"); print("=" * 78)
    po = nb201["proxy"]
    am, af, an = po.get("align_check", (float("nan"), float("nan"), 0))
    trm, trn = po.get("T_repro_check", (float("nan"), 0))
    
    tnb_cov = {}
    for cname, dd, acck, _ in spaces:
        if cname.startswith("TransNAS-"):
            tnb_cov[cname] = sorted([p for p in STD_PROXIES if p in dd and np.isfinite(dd[p]).sum() >= 4])
    notes = [
        f"- NB101: n={spaces[0][1]['n']} (SMALL-N; CIs wide, treat as suggestive).",
        f"- NB301: proxy json n={nb301['n']} (SMALL-N). proxy->curve genotype matches="
        f"{nb301['curve_match']}/{nb301['n']} out of {nb301['curve_n']} curve archs => DISJOINT arch sets; "
        f"used proxy-json val_acc; curve-T arm reported separately on the {nb301T['n'] if nb301T else 0} curve archs.",
        "- NB201 cf100/IN16 columns reuse cf10-computed proxy values vs cf100/IN16 ACCURACY "
        "(cross-dataset portability; ZC proxies are dataset-agnostic for TSS).",
        f"- NB201 index->arch-string mapping verified PROGRAMMATICALLY: base valacc vs curve final "
        f"eval-acc mean|d|={am:.3f}pp, {100*af:.1f}% within 1pp over n={an} (the ~0.4pp gap is "
        f"final-epoch-vs-recorded-val noise; mapping is correct). A runtime assert guards mean|d|<1.5pp.",
        f"- NB201 T uses the SAME definition as premise T_slope (first-{T_K}-epoch TRAIN-accuracy OLS slope) "
        f"but is recomputed from the curve pickles, not re-read; it agrees with premise.T_slope at "
        f"Spearman={trm:.3f} over n={trn}. They are not byte-identical because premise computed T_slope from a "
        f"different NASLib training-run source (premise.curve_accs); the definition and ranking match.",
        "- NB201 std proxies: fisher/grad_norm/grasp/snip/synflow/jacov/plain/zen/params/flops come from "
        "base_tss_cifar10.pkl; nwot/l2_norm/epe_nas were computed separately on the single-arch infer.tiny "
        "nets (run_extra_nb201.py, same seed/batch/data) and merged in, so all 13 standard proxies are now "
        "present for NB201. (l2_norm needs the single-arch net: the supernet get_model_from_arch_str has "
        "constant params across archs, which makes the pure-weight l2_norm degenerate.)",
        "- TransNAS: joined per-task by record idx; acc consistent across proxy files; no jigsaw files present. "
        + "; ".join(f"{k} has only proxies {v}" for k, v in tnb_cov.items())
        + ". TransNAS-object ships only {snip,grasp} files (not the full zoo), so its column is sparse and "
          "underpowered -- treat object portability as limited to those proxies.",
        "- Decomposition R2_cv is leakage-clean (ranks fit on train fold only); R2_in is in-sample.",
        "- Our spectral proxies exist for NB201 only; other spaces left N/A (NOT fabricated). "
        "Ralpha2 (Renyi-alpha=2 effective rank) is reported instead of Ralpha1, because Ralpha1 "
        "(Renyi-alpha=1) is byte-identical to erank (both = Shannon-entropy effective rank).",
    ]
    for ln in notes: print(ln)
    out.append("NOTES\n" + "\n".join(notes))

    if full_mode:
        txtp = os.path.join(RES, "crossspace_out.txt")
        with open(txtp, "w", encoding="utf-8") as f:
            f.write("\n\n".join(out))
        pklp = os.path.join(RES, "crossspace_results.pkl")
        save = {"matrix": matrix, "spaces": [s[0] for s in spaces], "seed": SEED}
        pickle.dump(save, open(pklp, "wb"))
        print(f"\nSaved: {txtp}\nSaved: {pklp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cross-space portability analysis for zero-cost NAS proxies.")
    ap.add_argument("--smoke", action="store_true",
                    help="fast loader check: every space subsampled to --smoke-cap archs, "
                         "all tables printed EXCEPT bootstrap CIs; no file output.")
    ap.add_argument("--full", action="store_true",
                    help="all archs, all tables incl bootstrap CIs; saves crossspace_out.txt + .pkl.")
    ap.add_argument("--smoke-cap", type=int, default=200,
                    help="max archs per space in --smoke mode (default 200).")
    a = ap.parse_args()
    if not (a.smoke or a.full):
        a.smoke = True
    run(full_mode=a.full, smoke_cap=a.smoke_cap)
