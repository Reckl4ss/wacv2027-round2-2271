# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  
    ALL_PROXIES, GLOBAL_SIGNS, array, build_full_zoo, default_input_dir,
    input_manifest, load_round1_tables, pca_summary, percentile_summary,
    rank_r2, rankdata_finite, spearman, write_csv, write_json,
)

INPUT_FILES = [
    "base_tss_cifar10.pkl", "base_sss_cifar10.pkl",
    "exp_alpha_components.pkl", "premise_data.pkl",
    "trainability_probes.pkl",
]
FLIP_PROXIES = ["cond_sum", "meco", "grasp"]

def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    support = np.sort(np.unique(np.concatenate([x, y])))
    fx = np.searchsorted(np.sort(x), support, side="right") / len(x)
    fy = np.searchsorted(np.sort(y), support, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))

def selection_randomization_pvalue(subset: np.ndarray, full: np.ndarray, seed: int, draws: int) -> tuple[float, float]:

    observed = ks_statistic(subset, full)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        draw = full[rng.choice(len(full), size=len(subset), replace=False)]
        if ks_statistic(draw, full) >= observed - 1e-15:
            exceed += 1
    return observed, float((exceed + 1) / (draws + 1))

def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    return {
        "n": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)), "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)), "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }

def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half

def add_holm_adjustment(rows: list[dict[str, object]]) -> None:

    for space in ("tss", "sss"):
        indices = [i for i, row in enumerate(rows) if row["space"] == space]
        ordered = sorted(indices, key=lambda i: float(rows[i]["selection_randomization_pvalue"]))
        running = 0.0
        m = len(ordered)
        for rank, index in enumerate(ordered):
            raw = float(rows[index]["selection_randomization_pvalue"])
            running = max(running, min(1.0, (m - rank) * raw))
            rows[index]["holm_adjusted_pvalue_within_axis"] = running

def oriented_rank_pca(rows: list[dict[str, float]]) -> dict[str, object]:
    acc = array(rows, "acc")
    columns = []
    signs: dict[str, int] = {}
    for proxy in ALL_PROXIES:
        values = array(rows, proxy)
        rho = spearman(values, acc)
        signs[proxy] = int(np.sign(rho)) if np.isfinite(rho) and abs(rho) >= 0.02 else GLOBAL_SIGNS[proxy]
        columns.append(rankdata_finite(signs[proxy] * values))
    matrix = np.column_stack(columns)
    complete = np.all(np.isfinite(matrix), axis=1)
    summary = pca_summary(matrix[complete])
    if spearman(summary["scores"], acc[complete]) < 0:
        summary["scores"] *= -1
        summary["loadings"] *= -1
    return {**summary, "signs": signs, "complete_mask": complete}

def run(input_dir: Path, output_dir: Path, n_resamples: int, permutation_draws: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_round1_tables(input_dir)
    index_sets = {
        name: {
            space: {int(r["key"]) for r in tables[name] if r.get("space") == space}
            for space in ("tss", "sss")
        }
        for name in ("components", "premise", "probes")
    }

    inclusion_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    distribution_tests: list[dict[str, object]] = []
    for space in ("tss", "sss"):
        base = tables[f"base_{space}"]
        for r in base:
            idx = int(r["i"])
            flags = {name: idx in index_sets[name][space] for name in index_sets}
            inclusion_rows.append({
                "space": space, "index": idx, "accuracy": float(r["testacc"]),
                "params": float(r["params"]),
                "param_count": float(r.get("param_count", np.nan)),
                "has_components": flags["components"],
                "has_premise": flags["premise"],
                "has_probes": flags["probes"],
                "included_full_zoo": all(flags.values()),
            })
        own = [r for r in inclusion_rows if r["space"] == space]
        flow_rows.append({
            "space": space,
            "base_available": len(base),
            "components_available_global": len(index_sets["components"][space]),
            "premise_available_global": len(index_sets["premise"][space]),
            "probes_available_global": len(index_sets["probes"][space]),
            "base_with_components": sum(r["has_components"] for r in own),
            "base_with_premise": sum(r["has_premise"] for r in own),
            "base_with_probes": sum(r["has_probes"] for r in own),
            "four_table_intersection": sum(r["included_full_zoo"] for r in own),
            "excluded_from_full_zoo": sum(not r["included_full_zoo"] for r in own),
        })
        for variable_index, variable in enumerate(("accuracy", "params", "param_count")):
            full = np.asarray([r[variable] for r in own], float)
            included = np.asarray([r[variable] for r in own if r["included_full_zoo"]], float)
            for population, values in (("base_available", full), ("full_zoo_intersection", included)):
                distribution_rows.append({
                    "space": space, "variable": variable, "population": population,
                    **distribution_summary(values),
                })
            ks, p = selection_randomization_pvalue(
                included, full, 2027301 + (0 if space == "tss" else 10) + variable_index,
                permutation_draws,
            )
            distribution_tests.append({
                "space": space, "variable": variable,
                "comparison": "intersection_vs_base_available",
                "ks_statistic": ks, "selection_randomization_pvalue": p,
                "randomization_draws": permutation_draws,
                "mean_difference_intersection_minus_base": float(included.mean() - full.mean()),
            })

    joined_tss = build_full_zoo(tables, "tss")
    joined_sss = build_full_zoo(tables, "sss")
    add_holm_adjustment(distribution_tests)
    if len(joined_tss) != 324 or len(joined_sss) != 138:
        raise RuntimeError("Frozen full-zoo counts changed")
    def pca_eligible(row: dict[str, float]) -> bool:
        return all(np.isfinite(row[proxy]) for proxy in ALL_PROXIES)

    tss_sampling_pool = [row for row in joined_tss if pca_eligible(row)]
    sss_sampling_pool = [row for row in joined_sss if pca_eligible(row)]
    if len(tss_sampling_pool) != 317 or len(sss_sampling_pool) != 138:
        raise RuntimeError("Frozen PCA complete-case counts changed")
    sss_pca = oriented_rank_pca(sss_sampling_pool)
    sss_flip_rho = {proxy: spearman(array(joined_sss, proxy), array(joined_sss, "acc")) for proxy in FLIP_PROXIES}

    run_rows: list[dict[str, object]] = []
    for replicate in range(n_resamples):
        seed = 2027000 + replicate
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(tss_sampling_pool), size=138, replace=False)
        sample = [tss_sampling_pool[int(i)] for i in selected]
        pca = oriented_rank_pca(sample)
        scores = pca["scores"]
        acc = array(sample, "acc")
        erank = array(sample, "erank")
        params = array(sample, "params")
        target = array(sample, "T_slope")
        base_r2 = rank_r2(acc, np.column_stack([erank, params]))
        full_r2 = rank_r2(acc, np.column_stack([erank, params, target]))
        complete = pca["complete_mask"]
        loading_cosine = float(np.dot(pca["loadings"], sss_pca["loadings"]) /
                               (np.linalg.norm(pca["loadings"]) * np.linalg.norm(sss_pca["loadings"])))
        record: dict[str, object] = {
            "replicate": replicate, "seed": seed,
            "n_pca_complete": int(complete.sum()),
            "var_pc1": pca["var_pc1"], "var_pc12": pca["var_pc12"],
            "corr_pc1_erank": spearman(scores, erank[complete]),
            "corr_pc1_params": spearman(scores, params[complete]),
            "corr_pc1_acc": spearman(scores, acc[complete]),
            "loading_cosine_vs_sss": loading_cosine,
            "rank_r2_acc_erank_params": base_r2,
            "rank_r2_acc_erank_params_T": full_r2,
            "trainability_increment": full_r2 - base_r2,
        }
        for proxy in FLIP_PROXIES:
            rho = spearman(array(sample, proxy), acc)
            record[f"rho_{proxy}"] = rho
            record[f"flip_vs_sss_{proxy}"] = bool(rho * sss_flip_rho[proxy] < 0)
        run_rows.append(record)

    summary_rows: list[dict[str, object]] = []
    numeric_metrics = [
        "n_pca_complete", "var_pc1", "var_pc12", "corr_pc1_erank", "corr_pc1_params",
        "corr_pc1_acc", "loading_cosine_vs_sss", "rank_r2_acc_erank_params",
        "rank_r2_acc_erank_params_T", "trainability_increment",
    ] + [f"rho_{p}" for p in FLIP_PROXIES]
    for metric in numeric_metrics:
        summary_rows.append({"metric": metric, **percentile_summary(r[metric] for r in run_rows)})
    for proxy in FLIP_PROXIES:
        successes = sum(bool(r[f"flip_vs_sss_{proxy}"]) for r in run_rows)
        low, high = wilson_interval(successes, n_resamples)
        summary_rows.append({
            "metric": f"flip_survival_rate_{proxy}",
            "mean": successes / n_resamples, "median": "", "q025": low,
            "q975": high, "min": "", "max": "",
        })

    output = {
        "experiment": "E3_sampling",
        "protocol": {
            "matched_n": 138,
            "source_population": "TSS four-table intersection restricted by the unchanged Round 1 all-29-proxy PCA complete-case rule, n=317 (7/324 non-finite rows excluded before sampling)",
            "resamples": n_resamples,
            "seed_schedule": f"2027000..{2027000 + n_resamples - 1}",
            "sampling": "without replacement independently for each seed",
            "survival_definition": "TSS subsample raw Spearman sign is opposite the fixed SSS n=138 raw Spearman sign",
            "uncertainty": "2.5/97.5 percentiles across matched-size subsamples; Wilson interval for survival proportions",
        },
        "input_manifest": input_manifest(input_dir, INPUT_FILES),
        "flow": flow_rows,
        "measurement_cohorts": {
            "components_equals_premise": all(index_sets["components"][s] == index_sets["premise"][s] for s in ("tss", "sss")),
            "components_equals_probes": all(index_sets["components"][s] == index_sets["probes"][s] for s in ("tss", "sss")),
            "interpretation": "The three added measurement tables use the same 500-architecture cohort per axis; full-zoo exclusions arise at the join with the independently covered base table, not from discordant missing rows among those three tables.",
        },
        "distribution_tests": distribution_tests,
        "sss_reference": {
            "n": len(sss_sampling_pool), "flip_rhos": sss_flip_rho,
            "pca_var_pc1": sss_pca["var_pc1"],
            "pca_corr_pc1_erank": spearman(sss_pca["scores"], array(sss_sampling_pool, "erank")[sss_pca["complete_mask"]]),
        },
    }
    write_json(output_dir / "e3_results.json", output)
    write_csv(output_dir / "e3_inclusion_audit.csv", inclusion_rows)
    write_csv(output_dir / "e3_population_flow.csv", flow_rows)
    write_csv(output_dir / "e3_distribution_summary.csv", distribution_rows)
    write_csv(output_dir / "e3_distribution_tests.csv", distribution_tests)
    write_csv(output_dir / "e3_matched_n_runs.csv", run_rows)
    write_csv(output_dir / "e3_matched_n_summary.csv", summary_rows)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
        sss = [r for r in inclusion_rows if r["space"] == "sss"]
        for variable, ax in zip(("accuracy", "params", "param_count"), axes):
            full = np.sort(np.asarray([r[variable] for r in sss], float))
            inc = np.sort(np.asarray([r[variable] for r in sss if r["included_full_zoo"]], float))
            ax.step(full, np.arange(1, len(full) + 1) / len(full), where="post", label="SSS available (425)")
            ax.step(inc, np.arange(1, len(inc) + 1) / len(inc), where="post", label="intersection (138)")
            ax.set_xlabel(variable)
            ax.set_ylabel("ECDF")
            ax.legend(frameon=False)
        fig.savefig(output_dir / "e3_sss_selection_ecdf.png", dpi=220)
        fig.savefig(output_dir / "e3_sss_selection_ecdf.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
        for ax, metric, label in zip(
            axes,
            ("var_pc1", "corr_pc1_erank", "trainability_increment"),
            ("PC1 variance", "corr(PC1, erank)", "R² increment from T"),
        ):
            ax.hist([r[metric] for r in run_rows], bins=30, color="#4C78A8")
            ax.set_xlabel(label)
            ax.set_ylabel("matched-n runs")
        fig.savefig(output_dir / "e3_matched_n_distributions.png", dpi=220)
        fig.savefig(output_dir / "e3_matched_n_distributions.pdf")
        plt.close(fig)
    except Exception as exc:
        (output_dir / "figure_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok", "flow": flow_rows,
        "results": "outputs/e3_results.json",
    }, sort_keys=True))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--permutation-draws", type=int, default=2000)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve(), args.resamples, args.permutation_draws)

if __name__ == "__main__":
    main()
