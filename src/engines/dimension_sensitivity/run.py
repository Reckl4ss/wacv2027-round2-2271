# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  
    array, bootstrap_ci, component_rows, default_input_dir, input_manifest,
    load_pickle, partial_spearman, spearman, stable_seed, write_csv,
    write_json,
)

INPUT_FILES = ["exp_alpha_components.pkl"]
SCORES = {
    "cond_sum_raw": lambda cond, length, erank: cond,
    "cond_sum_per_cached_length": lambda cond, length, erank: cond / length,
    "cond_sum_per_effective_rank": lambda cond, length, erank: cond / erank,
}

SEED_SCORE_TOKENS = {
    "cond_sum_raw": "cond_sum_raw",
    "cond_sum_per_cached_length": "cond_sum_per_spectrum_term",
    "cond_sum_per_effective_rank": "cond_sum_per_effective_rank",
}

def corr_with_ci(x: np.ndarray, y: np.ndarray, label: str, nboot: int) -> tuple[float, tuple[float, float]]:
    point = spearman(x, y)
    ci = bootstrap_ci(
        lambda idx: spearman(x[idx], y[idx]), len(x), stable_seed(label), nboot,
    )
    return point, ci

def ecdf_sup(x: np.ndarray, y: np.ndarray) -> float:
    support = np.sort(np.unique(np.concatenate([x, y])))
    return float(np.max(np.abs(
        np.searchsorted(np.sort(x), support, side="right") / len(x)
        - np.searchsorted(np.sort(y), support, side="right") / len(y)
    )))

def run(input_dir: Path, output_dir: Path, nboot: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"components": load_pickle(input_dir / "exp_alpha_components.pkl")}
    rows = {space: component_rows(tables, space) for space in ("tss", "sss")}
    if {k: len(v) for k, v in rows.items()} != {"tss": 500, "sss": 500}:
        raise RuntimeError("Expected the balanced 500/500 spectral arm")

    score_rows: list[dict[str, object]] = []
    score_arrays: dict[str, dict[str, np.ndarray]] = {"tss": {}, "sss": {}}
    for space in ("tss", "sss"):
        cond = array(rows[space], "cond_sum")
        length = array(rows[space], "cached_length")
        erank = array(rows[space], "erank")
        acc = array(rows[space], "acc")
        params = array(rows[space], "params")
        for name, formula in SCORES.items():
            score = formula(cond, length, erank)
            score_arrays[space][name] = score
            seed_name = SEED_SCORE_TOKENS[name]
            rho, ci = corr_with_ci(score, acc, f"E2|{space}|{seed_name}|rho", nboot)
            p_len = partial_spearman(score, acc, length)
            p_erank = partial_spearman(score, acc, erank)
            p_both = partial_spearman(score, acc, np.column_stack([length, erank, params]))
            score_rows.append({
                "space": space,
                "score": name,
                "n": len(score),
                "spearman_acc": rho,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "partial_acc_given_cached_length": p_len,
                "partial_acc_given_effective_rank": p_erank,
                "partial_acc_given_cached_length_erank_params": p_both,
            })

    cached_length_summary: dict[str, object] = {}
    for space in ("tss", "sss"):
        length = array(rows[space], "cached_length")
        cached_length_summary[space] = {
            "n": len(length),
            "min": float(length.min()),
            "median": float(np.median(length)),
            "max": float(length.max()),
            "unique": int(len(np.unique(length))),
            "counts": {str(int(k)): int(v) for k, v in sorted(Counter(length.astype(int)).items())},
        }

    common_lengths = sorted(set(array(rows["tss"], "cached_length").astype(int)) &
                            set(array(rows["sss"], "cached_length").astype(int)))
    cached_length_summary["cross_axis_exact_cached_length_overlap"] = {
        "common_cached_lengths": common_lengths,
        "possible": bool(common_lengths),
        "status": "descriptive_overlap_available" if common_lengths else "blocked_by_no_common_cached_length",
        "interpretation": "This is an overlap audit for the cached n_layers_used field only; it is not a validated representation-dimension match.",
    }

    strata_rows: list[dict[str, object]] = []
    for space in ("tss", "sss"):
        length = array(rows[space], "cached_length").astype(int)
        acc = array(rows[space], "acc")
        for value in sorted(np.unique(length)):
            keep = length == value
            if keep.sum() < 20:
                continue
            row: dict[str, object] = {
                "space": space, "cached_length": int(value), "n": int(keep.sum()),
            }
            for name in SCORES:
                row[f"rho_{name}"] = spearman(score_arrays[space][name][keep], acc[keep])
            strata_rows.append(row)

    erank_t = array(rows["tss"], "erank")
    erank_s = array(rows["sss"], "erank")
    overlap_low = float(max(erank_t.min(), erank_s.min()))
    overlap_high = float(min(erank_t.max(), erank_s.max()))
    common_range_rows: list[dict[str, object]] = []
    for space in ("tss", "sss"):
        erank = array(rows[space], "erank")
        acc = array(rows[space], "acc")
        keep = (erank >= overlap_low) & (erank <= overlap_high)
        for name in SCORES:
            seed_name = SEED_SCORE_TOKENS[name]
            rho, ci = corr_with_ci(
                score_arrays[space][name][keep], acc[keep],
                f"E2|{space}|{seed_name}|erank_common_support", nboot,
            )
            common_range_rows.append({
                "space": space, "score": name, "n": int(keep.sum()),
                "erank_common_range_low": overlap_low, "erank_common_range_high": overlap_high,
                "spearman_acc": rho, "ci_low": ci[0], "ci_high": ci[1],
            })

    output = {
        "experiment": "E2_dimension",
        "protocol": {
            "primary_population": "balanced spectral arm, 500 architectures per axis",
            "normalizations": {
                "raw": "cond_sum",
                "cached_length": "cond_sum / n_layers_used; n_layers_used is only a frozen cached count printed as 'layers' by alpha_sweep.log, with no checked-in generator/data dictionary certifying dimension semantics",
                "effective_rank": "cond_sum / Ralpha_sum[1.0]",
            },
            "cached_length_overlap_rule": "Exact equality of cached n_layers_used; no caliper is invented after viewing results. This is not called dimension matching.",
            "true_representation_dimension_match": {
                "status": "blocked_missing_validated_representation_dimension_field",
                "reason": "The frozen input lacks a validated representation-dimension field and the generating alpha-sweep implementation/data dictionary is not checked in.",
            },
            "bootstrap_draws": nboot,
            "bootstrap_seed_compatibility": "The corrected cached-length schema retains the frozen legacy internal seed token cond_sum_per_spectrum_term so reporting-only renaming does not change any bootstrap draw or numeric interval.",
            "interval_type": "marginal per-estimand percentile bootstrap interval; not simultaneous and not multiplicity-adjusted",
        },
        "input_manifest": input_manifest(input_dir, INPUT_FILES),
        "cached_length_summary": cached_length_summary,
        "effective_rank_common_range": {
            "low": overlap_low, "high": overlap_high,
            "ecdf_sup_before_restriction": ecdf_sup(erank_t, erank_s),
        },
    }
    write_json(output_dir / "e2_results.json", output)
    write_csv(output_dir / "e2_score_correlations.csv", score_rows)
    write_csv(output_dir / "e2_cached_length_strata.csv", strata_rows)
    write_csv(output_dir / "e2_erank_common_range.csv", common_range_rows)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
        for ax, space in zip(axes, ("tss", "sss")):
            selected = [r for r in score_rows if r["space"] == space]
            x = np.arange(len(selected))
            values = np.array([r["spearman_acc"] for r in selected], float)
            lo = values - np.array([r["ci_low"] for r in selected], float)
            hi = np.array([r["ci_high"] for r in selected], float) - values
            ax.errorbar(x, values, yerr=np.vstack([lo, hi]), fmt="o", capsize=4)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x, ["raw", "/ cached length", "/ erank"])
            ax.set_ylim(-1.02, 1.02)
            ax.set_title(space.upper())
            ax.set_ylabel("Spearman(score, accuracy)")
        fig.savefig(output_dir / "e2_normalization_correlations.png", dpi=220)
        fig.savefig(output_dir / "e2_normalization_correlations.pdf")
        plt.close(fig)
    except Exception as exc:
        (output_dir / "figure_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "cached_length_overlap": cached_length_summary["cross_axis_exact_cached_length_overlap"]["status"],
        "true_dimension_match": "blocked_missing_validated_representation_dimension_field",
        "results": "outputs/e2_results.json",
    }, sort_keys=True))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--nboot", type=int, default=2000)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve(), args.nboot)

if __name__ == "__main__":
    main()
