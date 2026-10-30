# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 20270825

def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def decoded_number(row: dict, field: str) -> float:
    value = row.get(field)
    if value is not None:
        return float(value)
    marker = row.get("nonfinite_fields", {}).get(field)
    if marker == "negative_infinity":
        return float("-inf")
    if marker == "positive_infinity":
        return float("inf")
    if marker == "nan":
        return float("nan")
    raise AssertionError(f"null numeric field without a sentinel marker: {field}")

def average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    boundaries = np.concatenate(
        ([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, [len(values)])
    )
    lengths = np.diff(boundaries)
    average = (boundaries[:-1] + 1 + boundaries[1:]) / 2.0
    ranks[order] = np.repeat(average, lengths)
    return ranks

def rho(x, y) -> float:
    x_rank, y_rank = average_ranks(x), average_ranks(y)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])

def bootstrap(x, y, draws: int, seed: int) -> tuple[float, float]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indexes = rng.integers(0, len(x), size=len(x))
        samples[draw] = rho(x[indexes], y[indexes])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-results", "--round1-results", dest="round1_results", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--architecture-scores", type=Path, required=True)
    parser.add_argument("--correlations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = read_jsonl(args.raw)
    architecture = read_jsonl(args.architecture_scores)
    reported = json.loads(args.results.read_text(encoding="utf-8"))
    with args.correlations.open(encoding="utf-8", newline="") as handle:
        correlation_rows = list(csv.DictReader(handle))

    populations = {
        "tss": pd.read_pickle(args.round1_results / "base_tss_cifar10.pkl"),
        "sss": pd.read_pickle(args.round1_results / "base_sss_cifar10.pkl"),
    }
    expected_n = {"tss": 1000, "sss": 425}
    expected_repeats = {"swap": list(range(32)), "aznas": list(range(5))}
    expected_seeds = {"swap": list(range(32)), "aznas": [1, 2, 3, 4, 5]}
    expected_raw = sum(expected_n.values()) * (32 + 5)
    if len(raw) != expected_raw:
        raise AssertionError(f"raw row count: {len(raw)} != {expected_raw}")
    if len(architecture) != sum(expected_n.values()) * 2:
        raise AssertionError(f"architecture row count: {len(architecture)}")

    cached = {}
    for space, rows in populations.items():
        if len(rows) != expected_n[space]:
            raise AssertionError(f"locked {space} population changed")
        for row in rows:
            cached[(space, int(row["i"]))] = row

    grouped: dict[tuple[str, str, int], list[dict]] = {}
    nonfinite_counts: dict[tuple[str, str, str], int] = {}
    raw_keys = set()
    for row in raw:
        method, space, architecture_id, repeat = (
            row["method"], row["space"], int(row["architecture_id"]), int(row["repeat"])
        )
        key = (method, space, architecture_id, repeat)
        if key in raw_keys:
            raise AssertionError(f"duplicate raw key: {key}")
        raw_keys.add(key)
        grouped.setdefault((method, space, architecture_id), []).append(row)
        reference = cached[(space, architecture_id)]
        expected_params = float(reference["params"]) * 1_000_000.0
        if abs(int(row["actual_parameter_count"]) - expected_params) > 0.500001:
            raise AssertionError(f"parameter mismatch: {key}")
        if not row["model_identity_exact"] or row["status"] != "completed":
            raise AssertionError(f"failed status/identity: {key}")
        required_scores = ["swap_score"] if method == "swap" else [
            "az_expressivity", "az_progressivity", "az_trainability"
        ]
        for field in required_scores:
            value = decoded_number(row, field)
            if math.isnan(value) or value == float("inf"):
                raise AssertionError(f"unsupported non-finite score: {key}/{field}")
            if value == float("-inf"):
                count_key = (space, field, "negative_infinity")
                nonfinite_counts[count_key] = nonfinite_counts.get(count_key, 0) + 1

    architecture_map = {
        (row["method"], row["space"], int(row["architecture_id"])): row for row in architecture
    }
    if len(architecture_map) != len(architecture):
        raise AssertionError("duplicate architecture-score key")
    for method in ("swap", "aznas"):
        for space, count in expected_n.items():
            ids = sorted(key[2] for key in grouped if key[0] == method and key[1] == space)
            if ids != sorted(int(row["i"]) for row in populations[space]):
                raise AssertionError(f"population mismatch: {method}/{space}")
            if len(ids) != count:
                raise AssertionError(f"join count mismatch: {method}/{space}")
            for architecture_id in ids:
                rows = sorted(grouped[(method, space, architecture_id)], key=lambda row: row["repeat"])
                if [row["repeat"] for row in rows] != expected_repeats[method]:
                    raise AssertionError(f"repeat mismatch: {method}/{space}/{architecture_id}")
                if [row["seed"] for row in rows] != expected_seeds[method]:
                    raise AssertionError(f"seed mismatch: {method}/{space}/{architecture_id}")
                summary = architecture_map[(method, space, architecture_id)]
                if method == "swap":
                    expected_score = float(np.mean([decoded_number(row, "swap_score") for row in rows]))
                    if not math.isclose(expected_score, summary["proxy_score"], abs_tol=1e-12):
                        raise AssertionError(f"SWAP mean mismatch: {space}/{architecture_id}")
                else:
                    for component in ("expressivity", "progressivity", "trainability"):
                        field = f"az_{component}"
                        values = np.asarray([decoded_number(row, field) for row in rows], dtype=np.float64)
                        expected_mean = float(np.mean(values))
                        observed_mean = decoded_number(summary, f"az_{component}_mean")
                        if expected_mean == float("-inf"):
                            if observed_mean != float("-inf"):
                                raise AssertionError(f"AZ sentinel mismatch: {space}/{architecture_id}/{component}")
                        elif not math.isclose(expected_mean, observed_mean, abs_tol=1e-12):
                            raise AssertionError(f"AZ component mismatch: {space}/{architecture_id}/{component}")
                        expected_negative = int(np.sum(np.isneginf(values)))
                        if int(summary[f"az_{component}_negative_infinity_repeats"]) != expected_negative:
                            raise AssertionError(f"AZ sentinel count mismatch: {space}/{architecture_id}/{component}")

    for space in ("tss", "sss"):
        selected = sorted(
            [row for row in architecture if row["method"] == "aznas" and row["space"] == space],
            key=lambda row: row["architecture_id"],
        )
        aggregate = np.zeros(len(selected), dtype=np.float64)
        for field in (
            "cached_flops_million", "az_expressivity_mean", "az_progressivity_mean",
            "az_trainability_mean",
        ):
            aggregate += np.log(average_ranks([decoded_number(row, field) for row in selected]) / len(selected))
        if not np.allclose(aggregate, [row["proxy_score"] for row in selected], atol=1e-12, rtol=0):
            raise AssertionError(f"AZ aggregate mismatch: {space}")

    correlation_map = {(row["method"], row["space"], row["target"]): row for row in correlation_rows}
    draws = int(reported["bootstrap_draws"])
    checked_correlations = []
    for method in ("swap", "aznas"):
        for space in ("tss", "sss"):
            selected = [row for row in architecture if row["method"] == method and row["space"] == space]
            scores = np.asarray([row["proxy_score"] for row in selected], dtype=np.float64)
            for target_name, target_field, target_offset in (
                ("test_accuracy", "cached_accuracy_test_percent", 0),
                ("validation_accuracy", "cached_accuracy_validation_percent", 100),
            ):
                targets = np.asarray([row[target_field] for row in selected], dtype=np.float64)
                independent_rho = rho(scores, targets)
                seed = BOOTSTRAP_SEED + (0 if method == "swap" else 10_000) + (0 if space == "tss" else 1_000) + target_offset
                ci_low, ci_high = bootstrap(scores, targets, draws, seed)
                csv_row = correlation_map[(method, space, target_name)]
                json_row = reported["methods"][method][space][target_name]
                comparisons = (
                    (independent_rho, float(csv_row["spearman_rho"])),
                    (independent_rho, float(json_row["spearman_rho"])),
                    (ci_low, float(csv_row["ci_low"])),
                    (ci_high, float(csv_row["ci_high"])),
                    (ci_low, float(json_row["ci_percentile"][0])),
                    (ci_high, float(json_row["ci_percentile"][1])),
                )
                if not all(math.isclose(left, right, abs_tol=1e-12, rel_tol=0) for left, right in comparisons):
                    raise AssertionError(f"correlation mismatch: {method}/{space}/{target_name}")
                checked_correlations.append(
                    {"method": method, "space": space, "target": target_name, "rho": independent_rho, "ci": [ci_low, ci_high]}
                )

    verdict = {
        "status": "PASS",
        "raw_rows": len(raw),
        "architecture_rows": len(architecture),
        "population_counts": expected_n,
        "model_identity_exact": True,
        "score_validity": "all scores finite except preserved official negative-infinity sentinels",
        "official_negative_infinity_counts": {
            "/".join(key): value for key, value in sorted(nonfinite_counts.items())
        },
        "correlations": checked_correlations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
