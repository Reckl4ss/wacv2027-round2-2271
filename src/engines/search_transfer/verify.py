# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

PROTOCOLS = ("round1_compatible", "identity_disjoint")
PAIRS = ("nb201_tss_to_sss", "nb201_sss_to_tss", "transnas_scene_to_nb101", "transnas_scene_to_nb201")
KS = (1, 5, 10, 50)
METHODS = ("source_best", "params", "synflow", "rank_ensemble", "random", "sign_stability_screen")
DETERMINISTIC = ("source_best", "params", "synflow", "rank_ensemble", "sign_stability_screen")
COMPARATORS = tuple(method for method in METHODS if method != "sign_stability_screen")
ENSEMBLE = ("params", "synflow", "snip")
SEED, N_BOOT, N_SPLITS, N_RANDOM, ORIENT_EPS = 1234, 1000, 40, 50, 0.02

def close(left: float, right: float) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"numeric mismatch: {left!r} != {right!r}")

def ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    keep = np.isfinite(values)
    finite = values[keep]
    order = np.argsort(finite, kind="mergesort")
    sorted_values, ranked = finite[order], np.empty(len(finite), dtype=float)
    left = 0
    while left < len(finite):
        right = left + 1
        while right < len(finite) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranked[order[left:right]] = (left + right - 1) / 2.0
        left = right
    result[keep] = ranked
    return result

def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 4:
        return float("nan")
    x, y = ranks(left[keep]), ranks(right[keep])
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def sign(value: float) -> int:
    if not math.isfinite(value) or abs(value) < ORIENT_EPS:
        return 0
    return 1 if value > 0 else -1

def oriented(values: np.ndarray, orientation: int) -> np.ndarray:
    score = orientation * np.asarray(values, dtype=float)
    return np.where(np.isfinite(score), score, -np.inf)

def ensemble(columns: dict[str, np.ndarray], rows: np.ndarray, signs: dict[str, int]) -> np.ndarray:
    pieces = [ranks(oriented(columns[p][rows], signs[p])) / max(len(rows) - 1, 1) for p in ENSEMBLE]
    return np.mean(np.vstack(pieces), axis=0)

def membership_bits(rows: np.ndarray, n: int) -> np.ndarray:
    membership = np.zeros(n, dtype=np.uint8)
    membership[np.unique(rows)] = 1
    return np.packbits(membership, bitorder="little")

def reconstruct_split(protocol: str, bootstrap: int, split: int, outer: np.ndarray, n: int):
    split_seed = SEED + 50000 + bootstrap * N_SPLITS + split
    random_rng = np.random.RandomState(split_seed)
    if protocol == "round1_compatible":
        permutation = random_rng.permutation(len(outer))
        cut = int(round(0.5 * len(permutation)))
        return outer[permutation[:cut]], outer[permutation[cut:]], random_rng, split_seed
    partition = random_rng.permutation(n)
    cut = int(round(0.5 * n))
    bootstrap_rng = np.random.RandomState(SEED + 900000 + bootstrap * N_SPLITS + split)
    calibration = bootstrap_rng.choice(partition[:cut], cut, replace=True)
    evaluation = bootstrap_rng.choice(partition[cut:], n - cut, replace=True)
    return calibration, evaluation, random_rng, split_seed

def method_scores(manifest, columns, accuracy, calibration, evaluation):
    source_rhos = {key: float(value) for key, value in manifest["source_rhos"].items()}
    signs = {key: int(value) for key, value in manifest["source_signs"].items()}
    full = list(manifest["full_coverage_proxies"])
    calib_rhos = {proxy: spearman(columns[proxy][calibration], accuracy[calibration]) for proxy in full}
    survivors = [proxy for proxy in full if sign(calib_rhos[proxy]) == signs[proxy]]
    screen = max(survivors, key=lambda proxy: abs(source_rhos[proxy])) if survivors else manifest["source_best"]
    scores = [
        oriented(columns[manifest["source_best"]][evaluation], signs[manifest["source_best"]]),
        oriented(columns["params"][evaluation], signs["params"]),
        oriented(columns["synflow"][evaluation], signs["synflow"]),
        ensemble(columns, evaluation, signs),
        oriented(columns[screen][evaluation], signs[screen]),
    ]
    return scores, str(screen)

def check_deterministic(data, row: tuple[int, ...], evaluation: np.ndarray, scores: list[np.ndarray]):
    selected_count = score_count = 0
    for method_index, method_score in enumerate(scores):
        order = np.argsort(-method_score, kind="mergesort")
        for k in KS:
            expected_indices = evaluation[order[:k]].astype(np.uint16)
            expected_scores = method_score[order[:k]].astype(np.float64)
            observed_indices = data[f"det_idx_k{k}"][row + (method_index,)]
            observed_scores = data[f"det_score_k{k}"][row + (method_index,)]
            if not np.array_equal(observed_indices, expected_indices):
                raise AssertionError(f"deterministic top-K mismatch row={row}, method={method_index}, K={k}")
            if not np.array_equal(observed_scores, expected_scores):
                raise AssertionError(f"stored score mismatch row={row}, method={method_index}, K={k}")
            selected_count += int(observed_indices.size)
            score_count += int(observed_scores.size)
    return selected_count, score_count

def check_random_sample(data, row: tuple[int, ...], evaluation: np.ndarray, random_rng) -> int:
    checked = 0
    for k in KS:
        expected = np.stack([
            evaluation[random_rng.choice(len(evaluation), k, replace=False)] for _ in range(N_RANDOM)
        ]).astype(np.uint16)
        observed = data[f"random_idx_k{k}"][row]
        if not np.array_equal(observed, expected):
            raise AssertionError(f"random seed reconstruction mismatch row={row}, K={k}")
        checked += int(observed.size)
    return checked

def parse_int_set(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return parsed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-bootstrap-offsets", type=parse_int_set, default=(0, 50, 99))
    parser.add_argument("--random-splits", type=parse_int_set, default=(0, 19, 39))
    parser.add_argument("--max-chunks-per-cell", type=int)
    parser.add_argument("--max-bootstraps-per-chunk", type=int)
    parser.add_argument("--max-splits-per-bootstrap", type=int)
    parser.add_argument("--allow-legacy-manifest", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if (args.max_bootstraps_per_chunk is not None or args.max_splits_per_bootstrap is not None) and args.max_chunks_per_cell is None:
        raise SystemExit("bootstrap/split limits are smoke-only and require --max-chunks-per-cell")
    root = args.output

    values, points = {}, {}
    trace_chunks = deterministic_count = stored_score_count = 0
    random_count = screen_decisions = point_random_count = 0
    round1_overlap_max = corrected_overlap_max = duplicate_ids = 0
    legacy_manifests = []

    for protocol in PROTOCOLS:
        for pair in PAIRS:
            with gzip.open(root / "manifests" / f"{pair}.json.gz", "rt", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if "full_coverage_proxies" not in manifest:
                if not args.allow_legacy_manifest:
                    raise AssertionError(f"{pair}: regenerate enhanced manifest before formal verification")
                manifest["full_coverage_proxies"] = list(manifest["source_rhos"])
                legacy_manifests.append(pair)
            architecture_ids = list(manifest["target_architecture_ids"])
            duplicate_ids += len(architecture_ids) - len(set(architecture_ids))
            if len(set(architecture_ids)) != len(architecture_ids):
                raise AssertionError(f"{pair}: duplicate canonical architecture IDs")
            accuracy = np.asarray(manifest["target_cached_accuracy"], dtype=float)
            columns = {key: np.asarray([np.nan if v is None else v for v in entries], dtype=float)
                       for key, entries in manifest["target_proxy_scores"].items()}
            proxy_order, n = list(manifest["proxy_order"]), len(accuracy)
            bootstrap_rng = np.random.RandomState(SEED + 99)
            round1_outer = [bootstrap_rng.randint(0, n, n) for _ in range(N_BOOT)]
            outer_values = {(k, method): np.full(N_BOOT, np.nan) for k in KS for method in METHODS}
            seen_bootstrap_ids = []
            chunks = sorted((root / "traces" / protocol / pair).glob("chunk_*.npz"))
            if len(chunks) != 10:
                raise AssertionError(f"{protocol}/{pair}: expected 10 chunks, found {len(chunks)}")
            if args.max_chunks_per_cell is not None:
                chunks = chunks[:args.max_chunks_per_cell]
            for path in chunks:
                with np.load(path) as archive:

                    data = {name: archive[name] for name in archive.files}
                    bootstrap_ids = data["bootstrap_ids"].astype(int)
                    seen_bootstrap_ids.extend(bootstrap_ids.tolist())
                    sampled_bootstraps = {int(bootstrap_ids[min(max(o, 0), len(bootstrap_ids) - 1)])
                                          for o in args.random_bootstrap_offsets}
                    local_bootstraps = range(len(bootstrap_ids))
                    if args.max_bootstraps_per_chunk is not None:
                        local_bootstraps = range(min(len(bootstrap_ids), args.max_bootstraps_per_chunk))
                    split_ids = range(N_SPLITS)
                    if args.max_splits_per_bootstrap is not None:
                        split_ids = range(min(N_SPLITS, args.max_splits_per_bootstrap))
                    for local_b in local_bootstraps:
                        bootstrap = bootstrap_ids[local_b]
                        base_outer = round1_outer[int(bootstrap)] if protocol == "round1_compatible" else np.arange(n)
                        for split in split_ids:
                            calibration, evaluation, random_rng, split_seed = reconstruct_split(
                                protocol, int(bootstrap), split, base_outer, n
                            )
                            row = (local_b, split)
                            if int(data["split_seeds"][row]) != split_seed:
                                raise AssertionError(f"{path}: split seed mismatch {row}")
                            if not np.array_equal(data["calib_membership"][row], membership_bits(calibration, n)):
                                raise AssertionError(f"{path}: CALIB mismatch {row}")
                            if not np.array_equal(data["eval_membership"][row], membership_bits(evaluation, n)):
                                raise AssertionError(f"{path}: EVAL mismatch {row}")
                            overlap = len(set(calibration.tolist()) & set(evaluation.tolist()))
                            if overlap != int(data["identity_overlap"][row]):
                                raise AssertionError(f"{path}: overlap mismatch {row}")
                            if protocol == "identity_disjoint":
                                corrected_overlap_max = max(corrected_overlap_max, overlap)
                            else:
                                round1_overlap_max = max(round1_overlap_max, overlap)
                            scores, screen = method_scores(manifest, columns, accuracy, calibration, evaluation)
                            if proxy_order[int(data["screen_proxy_code"][row])] != screen:
                                raise AssertionError(f"{path}: screen mismatch {row}")
                            screen_decisions += 1
                            checked, checked_scores = check_deterministic(data, row, evaluation, scores)
                            deterministic_count += checked
                            stored_score_count += checked_scores
                            if int(bootstrap) in sampled_bootstraps and split in args.random_splits:
                                random_count += check_random_sample(data, row, evaluation, random_rng)
                    for k in KS:
                        det = data[f"det_idx_k{k}"].astype(int)
                        for method_index, method in enumerate(DETERMINISTIC):
                            split_mean = np.mean(accuracy[det[:, :, method_index, :]], axis=-1)
                            outer_values[(k, method)][bootstrap_ids] = np.mean(split_mean, axis=1)
                        random_idx = data[f"random_idx_k{k}"].astype(int)
                        random_split = np.mean(np.mean(accuracy[random_idx], axis=-1), axis=-1)
                        outer_values[(k, "random")][bootstrap_ids] = np.mean(random_split, axis=1)
                trace_chunks += 1

            if args.max_chunks_per_cell is not None:
                continue
            if seen_bootstrap_ids != list(range(N_BOOT)):
                raise AssertionError(f"{protocol}/{pair}: incomplete bootstrap stream")
            if any(not np.all(np.isfinite(array)) for array in outer_values.values()):
                raise AssertionError(f"{protocol}/{pair}: incomplete aggregates")
            for (k, method), array in outer_values.items():
                values[(protocol, pair, k, method)] = array

            with np.load(root / "traces" / protocol / pair / "point_trace.npz") as archive:
                point = {name: archive[name] for name in archive.files}
                point_values = {(k, method): [] for k in KS for method in METHODS}
                for split in range(N_SPLITS):
                    random_rng = np.random.RandomState(SEED + 1 + split)
                    partition = random_rng.permutation(n)
                    cut = int(round(0.5 * n))
                    calibration, evaluation = partition[:cut], partition[cut:]
                    row = (split,)
                    if not np.array_equal(point["calib_membership"][row], membership_bits(calibration, n)):
                        raise AssertionError(f"{protocol}/{pair}: point CALIB mismatch {split}")
                    if not np.array_equal(point["eval_membership"][row], membership_bits(evaluation, n)):
                        raise AssertionError(f"{protocol}/{pair}: point EVAL mismatch {split}")
                    scores, screen = method_scores(manifest, columns, accuracy, calibration, evaluation)
                    if proxy_order[int(point["screen_proxy_code"][row])] != screen:
                        raise AssertionError(f"{protocol}/{pair}: point screen mismatch {split}")
                    screen_decisions += 1
                    checked, checked_scores = check_deterministic(point, row, evaluation, scores)
                    deterministic_count += checked
                    stored_score_count += checked_scores
                    point_random_count += check_random_sample(point, row, evaluation, random_rng)
                    for k in KS:
                        det = point[f"det_idx_k{k}"][split].astype(int)
                        for method_index, method in enumerate(DETERMINISTIC):
                            point_values[(k, method)].append(float(np.mean(accuracy[det[method_index]])))
                        point_values[(k, "random")].append(float(np.mean(accuracy[point[f"random_idx_k{k}"][split].astype(int)])))
                for (k, method), entries in point_values.items():
                    points[(protocol, pair, k, method)] = float(np.mean(entries))

    if args.max_chunks_per_cell is not None:
        report = {
            "status": "PASS_SMOKE", "verification_schema_version": "4.0.0",
            "trace_chunks_checked": trace_chunks,
            "max_chunks_per_protocol_pair": args.max_chunks_per_cell,
            "max_bootstraps_per_chunk": args.max_bootstraps_per_chunk,
            "max_splits_per_bootstrap": args.max_splits_per_bootstrap,
            "full_deterministic_selected_indices_checked": deterministic_count,
            "stored_deterministic_scores_checked": stored_score_count,
            "screen_decisions_recomputed": screen_decisions,
            "random_selected_indices_seed_reconstructed": random_count,
            "legacy_manifests_inferred": sorted(set(legacy_manifests)),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    with (root / "paired_comparisons.csv").open(newline="", encoding="utf-8") as stream:
        observed = list(csv.DictReader(stream))
    if len(observed) != 160:
        raise AssertionError(f"paired comparison rows: {len(observed)}")
    forbidden = {"p_raw", "p_holm", "round1_style_p1", "inferential_p_value"}
    if forbidden.intersection(observed[0]):
        raise AssertionError("inferential p-value fields must not appear")
    for row in observed:
        protocol, pair, k, comparator = row["protocol"], row["pair_id"], int(row["K"]), row["comparator"]
        delta = values[(protocol, pair, k, "sign_stability_screen")] - values[(protocol, pair, k, comparator)]
        point_delta = points[(protocol, pair, k, "sign_stability_screen")] - points[(protocol, pair, k, comparator)]
        close(float(row["point_delta"]), point_delta)
        close(float(row["bootstrap_mean_delta"]), float(np.mean(delta)))
        close(float(row["ci95_low"]), float(np.percentile(delta, 2.5)))
        close(float(row["ci95_high"]), float(np.percentile(delta, 97.5)))
        close(float(row["resampling_positive_fraction"]), float(np.sum(delta > 0) / len(delta)))
        close(float(row["resampling_zero_fraction"]), float(np.sum(delta == 0) / len(delta)))
        close(float(row["resampling_negative_fraction"]), float(np.sum(delta < 0) / len(delta)))
        unit = "paired_target_position_outer_resample" if protocol == "round1_compatible" else "identity_partition_then_within_side_resample"
        if row["resampling_unit"] != unit:
            raise AssertionError(f"{protocol}: incorrect resampling unit")

    gates = {"nb201_tss_to_sss": 5.401575000000022, "nb201_sss_to_tss": 0.0,
             "transnas_scene_to_nb101": 1.972606360912323, "transnas_scene_to_nb201": 0.0}
    for pair, expected in gates.items():
        actual = points[("round1_compatible", pair, 10, "sign_stability_screen")] - points[("round1_compatible", pair, 10, "source_best")]
        close(actual, expected)
    if corrected_overlap_max != 0 or duplicate_ids != 0:
        raise AssertionError("identity-disjoint architecture identity gate failed")
    if legacy_manifests:
        raise AssertionError("formal verification may not infer legacy manifest fields")

    report = {
        "status": "PASS", "verification_schema_version": "4.0.0",
        "formal_selection_traces_unchanged": True, "trace_chunks": trace_chunks,
        "full_deterministic_selected_indices_checked": deterministic_count,
        "stored_deterministic_scores_checked": stored_score_count,
        "screen_decisions_recomputed": screen_decisions,
        "random_selected_indices_seed_reconstructed": random_count,
        "random_reconstruction_scope": {
            "every_chunk_protocol_pair_k": True,
            "bootstrap_offsets_within_each_chunk": list(args.random_bootstrap_offsets),
            "split_ids": list(args.random_splits), "draws_per_sampled_split_k": N_RANDOM,
        },
        "point_random_selected_indices_seed_reconstructed": point_random_count,
        "paired_comparisons_recomputed": len(observed),
        "round1_compatible_overlap_max": round1_overlap_max,
        "identity_disjoint_overlap_max": corrected_overlap_max,
        "target_architecture_id_duplicates": duplicate_ids,
        "point_gates": "PASS", "resampling_sign_mass_diagnostics_checked": len(observed),
        "inferential_p_values_reported": False,
    }
    if not args.no_write:
        (root / "independent_verification_nested.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
