# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  
    ALL_PROXIES, GLOBAL_SIGNS, LABEL_USING, LOG_PROXIES, PROBE_PROXIES,
    SPEC_PROXIES, STD_PROXIES, array, bootstrap_ci, build_full_zoo,
    default_input_dir, input_manifest, load_round1_tables, partial_spearman,
    pca_summary, rankdata_finite, signed_log, spearman, stable_seed, write_csv,
    write_json,
)

INPUT_FILES = [
    "base_tss_cifar10.pkl", "base_sss_cifar10.pkl",
    "exp_alpha_components.pkl", "premise_data.pkl",
    "trainability_probes.pkl",
]

def per_space_signs(rows: list[dict[str, float]]) -> dict[str, int]:
    acc = array(rows, "acc")
    signs = {}
    for proxy in ALL_PROXIES:
        rho = spearman(array(rows, proxy), acc)
        signs[proxy] = int(np.sign(rho)) if np.isfinite(rho) and abs(rho) >= 0.02 else GLOBAL_SIGNS[proxy]
    return signs

def feature_matrix(rows: list[dict[str, float]], scale: str, signs: dict[str, int]) -> np.ndarray:
    columns = []
    for proxy in ALL_PROXIES:
        values = array(rows, proxy)
        if scale == "raw_z" and proxy in LOG_PROXIES:
            values = signed_log(values)
        values = signs[proxy] * values
        if scale == "rank":
            values = rankdata_finite(values)
        columns.append(values)
    return np.column_stack(columns)

def run(input_dir: Path, output_dir: Path, nboot: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_round1_tables(input_dir)
    joined = {space: build_full_zoo(tables, space) for space in ("tss", "sss")}
    expected = {"tss": 324, "sss": 138}
    actual = {space: len(rows) for space, rows in joined.items()}
    if actual != expected:
        raise RuntimeError(f"Frozen join mismatch: expected {expected}, observed {actual}")

    sign_sets = {
        space: {
            "unoriented": {q: 1 for q in ALL_PROXIES},
            "global_fixed": dict(GLOBAL_SIGNS),
            "per_space_r1": per_space_signs(joined[space]),
        }
        for space in ("tss", "sss")
    }

    details: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    table_rows: list[dict[str, object]] = []
    stable_anchor = ALL_PROXIES.index("stable_sum")
    for orientation in ("per_space_r1", "unoriented", "global_fixed"):
        details[orientation] = {}
        for scale in ("rank", "raw_z"):
            details[orientation][scale] = {}
            for space in ("tss", "sss"):
                rows = joined[space]
                matrix = feature_matrix(rows, scale, sign_sets[space][orientation])
                complete = np.all(np.isfinite(matrix), axis=1)
                if complete.sum() < 10:
                    raise RuntimeError(f"Too few complete PCA rows for {orientation}/{scale}/{space}")
                summary = pca_summary(
                    matrix[complete],
                    anchor_index=None if orientation == "per_space_r1" else stable_anchor,
                )
                
                if orientation == "per_space_r1" and spearman(summary["scores"], array(rows, "acc")[complete]) < 0:
                    summary["scores"] *= -1
                    summary["loadings"] *= -1
                scores = summary["scores"]
                record = {
                    "n": len(rows),
                    "n_pca_complete": int(complete.sum()),
                    "var_pc1": summary["var_pc1"],
                    "var_pc12": summary["var_pc12"],
                    "corr_pc1_erank": spearman(scores, array(rows, "erank")[complete]),
                    "corr_pc1_params": spearman(scores, array(rows, "params")[complete]),
                    "corr_pc1_acc": spearman(scores, array(rows, "acc")[complete]),
                    "corr_pc1_T": spearman(scores, array(rows, "T_slope")[complete]),
                    "loadings": {q: float(v) for q, v in zip(ALL_PROXIES, summary["loadings"])},
                    "signs": sign_sets[space][orientation],
                }
                details[orientation][scale][space] = record
                table_rows.append({
                    "orientation": orientation,
                    "scale": scale,
                    "space": space,
                    **{k: record[k] for k in (
                        "n", "n_pca_complete", "var_pc1", "var_pc12", "corr_pc1_erank",
                        "corr_pc1_params", "corr_pc1_acc", "corr_pc1_T",
                    )},
                })

            a = np.array([details[orientation][scale]["tss"]["loadings"][q] for q in ALL_PROXIES])
            b = np.array([details[orientation][scale]["sss"]["loadings"][q] for q in ALL_PROXIES])
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            rank_alignment = spearman(a, b)
            details[orientation][scale]["cross_axis"] = {
                "loading_cosine": cosine,
                "loading_spearman": rank_alignment,
                "note": "PC signs use the preregistered stable_sum anchor except the Round-1 reproduction arm, which follows Round 1 and points PC1 toward accuracy.",
            }
            for row in table_rows[-2:]:
                row["cross_axis_loading_cosine"] = cosine
                row["cross_axis_loading_spearman"] = rank_alignment

    partial_rows: list[dict[str, object]] = []
    for space in ("tss", "sss"):
        rows = joined[space]
        target = array(rows, "T_slope")
        acc = array(rows, "acc")
        controls = np.column_stack([array(rows, "erank"), array(rows, "params")])
        for proxy in ALL_PROXIES:
            oriented = GLOBAL_SIGNS[proxy] * array(rows, proxy)
            p_t = partial_spearman(oriented, target, controls)
            p_acc = partial_spearman(oriented, acc, controls)
            ci_t = bootstrap_ci(
                lambda idx, x=oriented, y=target, z=controls: partial_spearman(x[idx], y[idx], z[idx]),
                len(rows), stable_seed(f"E1|{space}|{proxy}|T"), nboot,
            )
            ci_acc = bootstrap_ci(
                lambda idx, x=oriented, y=acc, z=controls: partial_spearman(x[idx], y[idx], z[idx]),
                len(rows), stable_seed(f"E1|{space}|{proxy}|acc"), nboot,
            )
            partial_rows.append({
                "space": space,
                "proxy": proxy,
                "label_using": proxy in LABEL_USING,
                "global_sign": GLOBAL_SIGNS[proxy],
                "partial_T_given_erank_params": p_t,
                "partial_T_ci_low": ci_t[0],
                "partial_T_ci_high": ci_t[1],
                "partial_acc_given_erank_params": p_acc,
                "partial_acc_ci_low": ci_acc[0],
                "partial_acc_ci_high": ci_acc[1],
            })

    manifest = input_manifest(input_dir, INPUT_FILES)
    output = {
        "experiment": "E1_orientation",
        "protocol": {
            "orientations": ["per_space_r1", "unoriented", "global_fixed"],
            "global_sign_rule": "Frozen outcome-independent Round-1 prior sign map in common.GLOBAL_SIGNS, applied identically across axes without target labels; not claimed as a newly sourced definition-based direction for every proxy.",
            "pca_sign_rule": "stable_sum loading positive for unoriented/global; Round-1 arm points PC1 toward accuracy for exact presentation reproduction.",
            "scales": ["within-space ranks then z-score", "signed-log heavy tails/raw values then z-score"],
            "bootstrap_draws": nboot,
        },
        "input_manifest": manifest,
        "join_counts": actual,
        "pca": details,
    }
    write_json(output_dir / "e1_results.json", output)
    write_csv(output_dir / "e1_pca_summary.csv", table_rows)
    write_csv(output_dir / "e1_global_trainability_partials.csv", partial_rows)

    try:
        import matplotlib.pyplot as plt

        labels = ["R1 oriented", "Unoriented", "Global fixed"]
        orientations = ["per_space_r1", "unoriented", "global_fixed"]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
        for ax, scale in zip(axes, ("rank", "raw_z")):
            x = np.arange(3)
            vals = [details[o][scale]["cross_axis"]["loading_cosine"] for o in orientations]
            ax.bar(x, vals, color=["#777777", "#4C78A8", "#F58518"])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x, labels, rotation=18, ha="right")
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("TSS–SSS PC1 loading cosine")
            ax.set_title("Rank PCA" if scale == "rank" else "Raw-z PCA")
        fig.savefig(output_dir / "e1_loading_alignment.png", dpi=220)
        fig.savefig(output_dir / "e1_loading_alignment.pdf")
        plt.close(fig)
    except Exception as exc:  
        (output_dir / "figure_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok", "join_counts": actual,
        "results": "outputs/e1_results.json",
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
