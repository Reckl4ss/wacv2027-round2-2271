# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ALL_PROXIES, GLOBAL_SIGNS, build_full_zoo, load_round1_tables  

MATCHED_N = 138
N_RESAMPLES = 1000
SEED_BASE = 2027000
FLIP_PROXIES = ("cond_sum", "meco", "grasp")
NUMERIC_METRICS = (
    "n_pca_complete", "var_pc1", "var_pc12", "corr_pc1_erank",
    "corr_pc1_params", "corr_pc1_acc", "loading_cosine_vs_sss",
    "rank_r2_acc_erank_params", "rank_r2_acc_erank_params_T",
    "trainability_increment", "rho_cond_sum", "rho_meco", "rho_grasp",
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def rank(values: np.ndarray) -> np.ndarray:

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=float)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        result[order[left:right]] = (left + right + 1) / 2.0
        left = right
    return result

def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 3:
        return float("nan")
    x, y = rank(left[keep]), rank(right[keep])
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def values(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)

def rank_r2(target: np.ndarray, predictors: np.ndarray) -> float:
    target, predictors = np.asarray(target, float), np.asarray(predictors, float)
    keep = np.isfinite(target) & np.all(np.isfinite(predictors), axis=1)
    ranked_target = rank(target[keep])
    ranked_predictors = np.column_stack(
        [rank(predictors[keep, column]) for column in range(predictors.shape[1])]
    )
    design = np.column_stack([np.ones(len(ranked_target)), ranked_predictors])
    fitted = design @ np.linalg.lstsq(design, ranked_target, rcond=None)[0]
    denominator = float(np.sum((ranked_target - ranked_target.mean()) ** 2))
    return float(1.0 - np.sum((ranked_target - fitted) ** 2) / denominator)

def oriented_pca(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, float, float]:
    accuracy = values(rows, "acc")
    columns = []
    for proxy in ALL_PROXIES:
        proxy_values = values(rows, proxy)
        rho = spearman(proxy_values, accuracy)
        orientation = int(np.sign(rho)) if np.isfinite(rho) and abs(rho) >= 0.02 else GLOBAL_SIGNS[proxy]
        columns.append(rank(orientation * proxy_values))
    matrix = np.column_stack(columns)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    scores, loadings = u[:, 0] * singular[0], vt[0].copy()
    if spearman(scores, accuracy) < 0:
        scores, loadings = -scores, -loadings
    variance = singular ** 2
    return scores, loadings, float(variance[0] / variance.sum()), float(variance[:2].sum() / variance.sum())

def percentile(values_: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values_), dtype=float)
    return {
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)), "q975": float(np.quantile(array, 0.975)),
        "min": float(np.min(array)), "max": float(np.max(array)),
    }

def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half

def close(observed: float, expected: float, label: str, tolerance: float = 5e-12) -> float:
    error = abs(observed - expected)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {observed!r} != {expected!r}")
    return error

def verify_inputs(input_dir: Path, manifest_path: Path) -> None:
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, name = line.split("  ", 1)
        actual = sha256_file(input_dir / name)
        if actual != expected:
            raise AssertionError(f"input checksum mismatch: {name}")

def write_selected_ids(path: Path, selected_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            with __import__("io").TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=("replicate", "seed", "selected_architecture_ids"))
                writer.writeheader()
                writer.writerows(selected_rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, default=root / "outputs")
    parser.add_argument("--input-manifest", type=Path, default=root / "input_checksums.sha256")
    parser.add_argument("--selected-output", type=Path, default=root / "outputs/e3_selected_ids.csv.gz")
    parser.add_argument("--report", type=Path, default=root / "outputs/e3_independent_verification.json")
    args = parser.parse_args()

    verify_inputs(args.input_dir, args.input_manifest)
    tables = load_round1_tables(args.input_dir)
    tss_join = build_full_zoo(tables, "tss")
    sss_join = build_full_zoo(tables, "sss")
    tss_pool = [row for row in tss_join if all(np.isfinite(row[proxy]) for proxy in ALL_PROXIES)]
    sss_pool = [row for row in sss_join if all(np.isfinite(row[proxy]) for proxy in ALL_PROXIES)]
    if (len(tss_pool), len(sss_pool)) != (317, 138):
        raise AssertionError(f"complete-case population changed: {(len(tss_pool), len(sss_pool))}")

    with (args.outputs / "e3_matched_n_runs.csv").open(newline="", encoding="utf-8") as stream:
        observed_runs = {int(row["replicate"]): row for row in csv.DictReader(stream)}
    with (args.outputs / "e3_matched_n_summary.csv").open(newline="", encoding="utf-8") as stream:
        observed_summary = {row["metric"]: row for row in csv.DictReader(stream)}
    if sorted(observed_runs) != list(range(N_RESAMPLES)):
        raise AssertionError("matched-n run stream is incomplete")

    _, sss_loadings, _, _ = oriented_pca(sss_pool)
    sss_signs = {proxy: spearman(values(sss_join, proxy), values(sss_join, "acc")) for proxy in FLIP_PROXIES}
    reconstructed: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    max_error = 0.0
    for replicate in range(N_RESAMPLES):
        seed = SEED_BASE + replicate
        chosen = np.random.default_rng(seed).choice(len(tss_pool), size=MATCHED_N, replace=False)
        sample = [tss_pool[int(index)] for index in chosen]
        selected_rows.append({
            "replicate": replicate, "seed": seed,
            "selected_architecture_ids": "|".join(str(int(row["index"])) for row in sample),
        })
        accuracy, erank = values(sample, "acc"), values(sample, "erank")
        params, target = values(sample, "params"), values(sample, "T_slope")
        scores, loadings, var_pc1, var_pc12 = oriented_pca(sample)
        base_r2 = rank_r2(accuracy, np.column_stack([erank, params]))
        full_r2 = rank_r2(accuracy, np.column_stack([erank, params, target]))
        row: dict[str, object] = {
            "n_pca_complete": MATCHED_N, "var_pc1": var_pc1, "var_pc12": var_pc12,
            "corr_pc1_erank": spearman(scores, erank), "corr_pc1_params": spearman(scores, params),
            "corr_pc1_acc": spearman(scores, accuracy),
            "loading_cosine_vs_sss": float(np.dot(loadings, sss_loadings) /
                                             (np.linalg.norm(loadings) * np.linalg.norm(sss_loadings))),
            "rank_r2_acc_erank_params": base_r2, "rank_r2_acc_erank_params_T": full_r2,
            "trainability_increment": full_r2 - base_r2,
        }
        for proxy in FLIP_PROXIES:
            rho = spearman(values(sample, proxy), accuracy)
            row[f"rho_{proxy}"] = rho
            row[f"flip_vs_sss_{proxy}"] = bool(rho * sss_signs[proxy] < 0)
        observed = observed_runs[replicate]
        if int(observed["seed"]) != seed:
            raise AssertionError(f"replicate {replicate}: seed mismatch")
        for metric in NUMERIC_METRICS:
            max_error = max(max_error, close(float(row[metric]), float(observed[metric]), f"replicate {replicate}/{metric}"))
        for proxy in FLIP_PROXIES:
            expected_bool = observed[f"flip_vs_sss_{proxy}"].lower() == "true"
            if bool(row[f"flip_vs_sss_{proxy}"]) != expected_bool:
                raise AssertionError(f"replicate {replicate}/{proxy}: flip mismatch")
        reconstructed.append(row)

    for metric in NUMERIC_METRICS:
        summary = percentile(float(row[metric]) for row in reconstructed)
        observed = observed_summary[metric]
        for field, value in summary.items():
            max_error = max(max_error, close(value, float(observed[field]), f"summary {metric}/{field}"))
    for proxy in FLIP_PROXIES:
        successes = sum(bool(row[f"flip_vs_sss_{proxy}"]) for row in reconstructed)
        low, high = wilson(successes, N_RESAMPLES)
        observed = observed_summary[f"flip_survival_rate_{proxy}"]
        max_error = max(max_error, close(successes / N_RESAMPLES, float(observed["mean"]), f"survival {proxy}/mean"))
        max_error = max(max_error, close(low, float(observed["q025"]), f"survival {proxy}/low"))
        max_error = max(max_error, close(high, float(observed["q975"]), f"survival {proxy}/high"))

    write_selected_ids(args.selected_output, selected_rows)
    report = {
        "status": "PASS", "verifier": "independent_of_run_e3",
        "tss_sampling_population": len(tss_pool), "matched_n": MATCHED_N,
        "seeds_reconstructed": N_RESAMPLES, "selected_ids_rows": len(selected_rows),
        "run_metrics_recomputed": N_RESAMPLES * len(NUMERIC_METRICS),
        "summary_metrics_recomputed": len(NUMERIC_METRICS) + len(FLIP_PROXIES),
        "max_absolute_numeric_error": max_error,
        "selected_ids_sha256": sha256_file(args.selected_output),
        "interval_scope": "algorithmic subsampling percentiles; per-proxy marginal Wilson intervals",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
