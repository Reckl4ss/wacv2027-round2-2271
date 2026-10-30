# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

STD_PROXIES = [
    "fisher", "grad_norm", "grasp", "snip", "synflow", "jacob_cov",
    "plain", "zico", "meco", "zen", "ntk", "gradsign",
]
SPEC_PROXIES = [
    "cond_sum", "stable_sum", "Ralpha_0.5", "Ralpha_2.0",
    "Ralpha_4.0", "Ralpha_8.0", "Ralpha_16.0",
]
PROBE_PROXIES = [
    "Q_ntk_logcond", "Q_ntk_logtrace", "Q_ntk_effrank", "Q_ntk_loglmax",
    "Q_zico_rand", "Q_zico_rot", "Q_coh_rand", "Q_coh_rot",
    "Q_adapt_rot", "Q_zico_oracle",
]
ALL_PROXIES = STD_PROXIES + SPEC_PROXIES + PROBE_PROXIES
LABEL_USING = {"Q_zico_oracle"}

GLOBAL_SIGNS = {q: 1 for q in ALL_PROXIES}
GLOBAL_SIGNS.update({
    "ntk": -1,
    "cond_sum": -1,
    "Q_ntk_logcond": -1,
    "Q_coh_rand": -1,
    "Q_coh_rot": -1,
})

LOG_PROXIES = {
    "synflow", "fisher", "gradsign", "grasp", "ntk", "zico", "snip",
    "grad_norm", "meco", "zen",
}

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def default_input_dir() -> Path:
    return repo_root() / "data" / "frozen"

def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def input_manifest(input_dir: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        path = input_dir / name
        rows.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows

def load_round1_tables(input_dir: Path | None = None) -> dict[str, Any]:
    root = input_dir or default_input_dir()
    names = {
        "base_tss": "base_tss_cifar10.pkl",
        "base_sss": "base_sss_cifar10.pkl",
        "components": "exp_alpha_components.pkl",
        "premise": "premise_data.pkl",
        "probes": "trainability_probes.pkl",
    }
    return {key: load_pickle(root / name) for key, name in names.items()}

def _indexed(records: list[dict[str, Any]], space: str) -> dict[int, dict[str, Any]]:
    return {int(r["key"]): r for r in records if r.get("space") == space}

def build_full_zoo(tables: dict[str, Any], space: str) -> list[dict[str, float]]:

    base = tables[f"base_{space}"]
    components = _indexed(tables["components"], space)
    premise = _indexed(tables["premise"], space)
    probes = _indexed(tables["probes"], space)
    rows: list[dict[str, float]] = []
    for b in base:
        idx = int(b["i"])
        if idx not in components or idx not in premise or idx not in probes:
            continue
        c, p, q = components[idx], premise[idx], probes[idx]
        row: dict[str, float] = {
            "index": idx,
            "acc": float(b["testacc"]),
            "valacc": float(b["valacc"]),
            "params": float(b["params"]),
            "param_count": float(b.get("param_count", np.nan)),
            "erank": float(c["Ralpha_sum"][1.0]),
            "T_slope": float(p["T_slope"]),
            "T_loss": float(p["T_loss"]),

            "cached_length": float(c["n_layers_used"]),
        }
        for name in STD_PROXIES:
            row[name] = float(b.get(name, np.nan))
        row["cond_sum"] = float(c["cond_sum"])
        row["stable_sum"] = float(c["stable_sum"])
        for alpha in (0.5, 2.0, 4.0, 8.0, 16.0):
            row[f"Ralpha_{alpha}"] = float(c["Ralpha_sum"][alpha])
        for name in PROBE_PROXIES:
            row[name] = float(q.get(name, np.nan))
        rows.append(row)
    return rows

def component_rows(tables: dict[str, Any], space: str) -> list[dict[str, float]]:
    rows = []
    for r in tables["components"]:
        if r.get("space") != space:
            continue
        rows.append({
            "index": int(r["key"]),
            "acc": float(r["acc"]),
            "params": float(r["params"]),
            "cond_sum": float(r["cond_sum"]),
            "stable_sum": float(r["stable_sum"]),
            "erank": float(r["Ralpha_sum"][1.0]),

            "cached_length": float(r["n_layers_used"]),
        })
    return rows

def array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([r.get(key, np.nan) for r in rows], dtype=float)

def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]

def rankdata_finite(values: np.ndarray) -> np.ndarray:

    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    keep = np.isfinite(values)
    if keep.any():
        out[keep] = rankdata(values[keep])
    return out

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def _rank_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    if matrix.ndim == 1:
        return rankdata(matrix)[:, None]
    return np.column_stack([rankdata(matrix[:, j]) for j in range(matrix.shape[1])])

def residualize_rank(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    ry = rankdata(np.asarray(y, float))
    rz = _rank_matrix(np.asarray(controls, float))
    design = np.column_stack([np.ones(len(ry)), rz])
    return ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]

def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    x, y, controls = np.asarray(x, float), np.asarray(y, float), np.asarray(controls, float)
    if controls.ndim == 1:
        controls = controls[:, None]
    keep = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if keep.sum() < controls.shape[1] + 4:
        return float("nan")
    rx = residualize_rank(x[keep], controls[keep])
    ry = residualize_rank(y[keep], controls[keep])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def rank_r2(y: np.ndarray, predictors: np.ndarray) -> float:
    y, predictors = np.asarray(y, float), np.asarray(predictors, float)
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    keep = np.isfinite(y) & np.all(np.isfinite(predictors), axis=1)
    ry = rankdata(y[keep])
    rx = _rank_matrix(predictors[keep])
    design = np.column_stack([np.ones(len(ry)), rx])
    pred = design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    denom = float(np.sum((ry - ry.mean()) ** 2))
    return float(1.0 - np.sum((ry - pred) ** 2) / denom)

def bootstrap_ci(
    fn: Callable[[np.ndarray], float], n: int, seed: int, nboot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.empty(nboot, dtype=float)
    for i in range(nboot):
        vals[i] = fn(rng.integers(0, n, size=n))
    alpha = (1.0 - ci) / 2.0
    return tuple(float(v) for v in np.nanquantile(vals, [alpha, 1.0 - alpha]))

def stable_seed(label: str, base: int = 2027000) -> int:
    value = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return base + value % 1_000_000_000

def zscore(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    sd[sd == 0] = 1.0
    return (matrix - mean) / sd

def signed_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return np.sign(values) * np.log1p(np.abs(values))

def pca_summary(matrix: np.ndarray, anchor_index: int | None = None) -> dict[str, Any]:
    matrix = zscore(matrix)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    scores = u[:, 0] * singular[0]
    loading = vt[0].copy()
    if anchor_index is not None and loading[anchor_index] < 0:
        scores *= -1
        loading *= -1
    eigen = singular ** 2
    return {
        "scores": scores,
        "loadings": loading,
        "var_pc1": float(eigen[0] / eigen.sum()),
        "var_pc12": float(eigen[:2].sum() / eigen.sum()),
    }

def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values), float)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "q025": float(np.nanquantile(values, 0.025)),
        "q975": float(np.nanquantile(values, 0.975)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }

def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
