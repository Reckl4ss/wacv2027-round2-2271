# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SEED = 1234
KS = (1, 5, 10, 50)
N_BOOT = 1000
N_SPLITS = 40
N_RANDOM = 50
TARGET_CAP = 1000
CALIB_FRAC = 0.5
ORIENT_EPS = 0.02
FULL_COVERAGE = 0.8
PROTOCOLS = ("round1_compatible", "identity_disjoint")
METHODS = ("source_best", "params", "synflow", "rank_ensemble", "random", "sign_stability_screen")
DETERMINISTIC = ("source_best", "params", "synflow", "rank_ensemble", "sign_stability_screen")
COMPARATORS = tuple(method for method in METHODS if method != "sign_stability_screen")
ENSEMBLE = ("params", "synflow", "snip")
NB201 = ("fisher", "grad_norm", "gradsign", "grasp", "jacob_cov", "meco", "ntk",
         "plain", "snip", "synflow", "zen", "zico", "params")
EXTERNAL = ("epe_nas", "fisher", "flops", "grad_norm", "grasp", "jacob_cov", "l2_norm",
            "nwot", "params", "plain", "snip", "synflow", "zen")
EXPECTED_SHA256 = {
    "base_tss_cifar10.pkl": "00e6b67e07ea7f3937b35cba8c4852ed00e440213bbb3ef254b6568633e59bb0",
    "base_sss_cifar10.pkl": "3b15bec29b9c38e10f5ef7df141bbb7d9c595aab5afd3bb570cc308f48f787d0",
    "zc_nasbench101.json": "aaf67baf526661c84b9c2bd449926cd0ff41d9451fe0aeaeba3450013ec6801f",
    "zc_transbench101_micro.json": "60934abb09b682abe0e1ca44acf08a1162e3001f34960ac78fda9f8867525c52",
}

@dataclass
class Axis:
    name: str
    ids: np.ndarray
    accuracy: np.ndarray
    columns: dict[str, np.ndarray]
    input_rows: np.ndarray

@dataclass
class Pair:
    pair_id: str
    role: str
    source: Axis
    target: Axis
    proxies: tuple[str, ...]

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def stable_id(prefix: str, raw: str) -> str:
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"

def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return value if math.isfinite(value) else math.nan

def ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    mask = np.isfinite(values)
    x = values[mask]
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranked = np.empty(len(x), dtype=float)
    left = 0
    while left < len(x):
        right = left + 1
        while right < len(x) and sx[right] == sx[left]:
            right += 1
        ranked[order[left:right]] = (left + right - 1) / 2.0
        left = right
    out[mask] = ranked
    return out

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return math.nan
    rx, ry = ranks(x[mask]), ranks(y[mask])
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def sign(rho: float) -> int:
    if not math.isfinite(rho) or abs(rho) < ORIENT_EPS:
        return 0
    return 1 if rho > 0 else -1

def load_nb201(path: Path, label: str) -> Axis:
    with path.open("rb") as stream:
        rows = pickle.load(stream)
    return Axis(
        name=label,
        ids=np.asarray([f"{label.lower()}:{int(row['i']):05d}" for row in rows]),
        accuracy=np.asarray([finite(row.get("testacc")) for row in rows]),
        columns={proxy: np.asarray([finite(row.get(proxy)) for row in rows]) for proxy in NB201},
        input_rows=np.arange(len(rows), dtype=np.int32),
    )

def load_external(path: Path, task: str, prefix: str, label: str) -> Axis:
    with path.open(encoding="utf-8") as stream:
        items = list(json.load(stream)[task].items())
    columns = {proxy: [] for proxy in EXTERNAL}
    ids, accuracy = [], []
    for raw_key, row in items:
        ids.append(stable_id(prefix, raw_key))
        accuracy.append(finite(row.get("val_accuracy")))
        for proxy in EXTERNAL:
            input_name = "jacov" if proxy == "jacob_cov" else proxy
            value = row.get(input_name)
            if isinstance(value, dict):
                value = value.get("score")
            columns[proxy].append(finite(value))
    return Axis(label, np.asarray(ids), np.asarray(accuracy),
                {key: np.asarray(value) for key, value in columns.items()},
                np.arange(len(ids), dtype=np.int32))

def subset_axis(axis: Axis, indices: np.ndarray) -> Axis:
    return Axis(axis.name, axis.ids[indices], axis.accuracy[indices],
                {key: value[indices] for key, value in axis.columns.items()}, axis.input_rows[indices])

def target_subset(axis: Axis) -> Axis:
    candidate = np.flatnonzero(np.isfinite(axis.accuracy))
    if len(candidate) > TARGET_CAP:
        chosen = np.random.RandomState(SEED).permutation(len(candidate))[:TARGET_CAP]
        candidate = candidate[np.sort(chosen)]
    return subset_axis(axis, candidate)

def build_pairs(round1: Path, official: Path) -> list[Pair]:
    paths = {
        "base_tss_cifar10.pkl": round1 / "base_tss_cifar10.pkl",
        "base_sss_cifar10.pkl": round1 / "base_sss_cifar10.pkl",
        "zc_nasbench101.json": official / "zc_nasbench101.json",
        "zc_transbench101_micro.json": official / "zc_transbench101_micro.json",
    }
    for name, path in paths.items():
        if file_sha256(path) != EXPECTED_SHA256[name]:
            raise RuntimeError(f"input checksum mismatch: {name}")
    tss = load_nb201(paths["base_tss_cifar10.pkl"], "NB201-TSS")
    sss = load_nb201(paths["base_sss_cifar10.pkl"], "NB201-SSS")
    nb101 = load_external(paths["zc_nasbench101.json"], "cifar10", "nb101", "NB101")
    scene = load_external(paths["zc_transbench101_micro.json"], "class_scene", "transnas_scene", "TransNAS-scene")
    cross = tuple(proxy for proxy in EXTERNAL if proxy in NB201)
    return [
        Pair("nb201_tss_to_sss", "original_trap", tss, target_subset(sss), NB201),
        Pair("nb201_sss_to_tss", "original_reverse_null", sss, target_subset(tss), NB201),
        Pair("transnas_scene_to_nb101", "additional_trap", scene, target_subset(nb101), EXTERNAL),
        Pair("transnas_scene_to_nb201", "additional_null", scene, target_subset(tss), cross),
    ]

def pair_context(pair: Pair) -> dict[str, Any]:
    usable = []
    rhos, signs = {}, {}
    for proxy in pair.proxies:
        src, tgt = pair.source.columns[proxy], pair.target.columns[proxy]
        if int(np.isfinite(src).sum()) < 20 or int(np.isfinite(tgt).sum()) < 20:
            continue
        rho = spearman(src, pair.source.accuracy)
        orientation = sign(rho)
        if orientation:
            usable.append(proxy)
            rhos[proxy], signs[proxy] = rho, orientation
    full = [proxy for proxy in usable
            if np.isfinite(pair.source.columns[proxy]).mean() >= FULL_COVERAGE
            and np.isfinite(pair.target.columns[proxy]).mean() >= FULL_COVERAGE]
    if not full or any(proxy not in full for proxy in ENSEMBLE):
        raise RuntimeError(f"{pair.pair_id}: incomplete full-coverage method pool")
    source_best = max(full, key=lambda proxy: abs(rhos[proxy]))
    proxy_codes = {proxy: index for index, proxy in enumerate(pair.proxies)}
    return {"usable": usable, "full": full, "rhos": rhos, "signs": signs,
            "source_best": source_best, "proxy_codes": proxy_codes}

def write_manifest(pair: Pair, context: dict[str, Any], output: Path) -> None:
    target = pair.target
    def json_floats(values: np.ndarray) -> list[float | None]:
        return [float(value) if math.isfinite(float(value)) else None for value in values]
    payload = {
        "pair_id": pair.pair_id, "role": pair.role, "source_axis": pair.source.name,
        "target_axis": target.name, "source_n": len(pair.source.ids), "target_n": len(target.ids),
        "source_best": context["source_best"], "source_rhos": context["rhos"],
        "source_signs": context["signs"], "proxy_order": list(pair.proxies),
        "source_finite_counts": {
            proxy: int(np.isfinite(pair.source.columns[proxy]).sum()) for proxy in context["usable"]
        },
        "full_coverage_proxies": list(context["full"]),
        "target_architecture_ids": target.ids.tolist(), "target_input_rows": target.input_rows.tolist(),
        "target_cached_accuracy": json_floats(target.accuracy),
        "target_proxy_scores": {key: json_floats(value) for key, value in target.columns.items()},
    }
    path = output / "manifests" / f"{pair.pair_id}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as raw:
        raw.write((json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())

def oriented_score(values: np.ndarray, orientation: int) -> np.ndarray:
    score = orientation * values.astype(float)
    return np.where(np.isfinite(score), score, -np.inf)

def ensemble_score(pair: Pair, rows: np.ndarray, signs: dict[str, int]) -> np.ndarray:
    pieces = []
    for proxy in ENSEMBLE:
        rank = ranks(oriented_score(pair.target.columns[proxy][rows], signs[proxy]))
        pieces.append(rank / max(len(rank) - 1, 1))
    return np.mean(np.vstack(pieces), axis=0)

def top_positions(score: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-score, kind="mergesort")[:k]

def membership_bits(rows: np.ndarray, n: int) -> np.ndarray:
    bits = np.zeros(n, dtype=np.uint8)
    bits[np.unique(rows)] = 1
    return np.packbits(bits, bitorder="little")

def split_rows(protocol: str, b: int, s: int, outer_rows: np.ndarray, n: int):
    split_seed = SEED + 50000 + b * N_SPLITS + s
    rng = np.random.RandomState(split_seed)
    if protocol == "round1_compatible":
        perm = rng.permutation(len(outer_rows))
        cut = int(round(CALIB_FRAC * len(perm)))
        return outer_rows[perm[:cut]], outer_rows[perm[cut:]], rng, split_seed
    partition = rng.permutation(n)
    cut = int(round(CALIB_FRAC * n))
    calib_base, eval_base = partition[:cut], partition[cut:]
    boot = np.random.RandomState(SEED + 900000 + b * N_SPLITS + s)
    calib = boot.choice(calib_base, len(calib_base), replace=True)
    evaluation = boot.choice(eval_base, len(eval_base), replace=True)
    return calib, evaluation, rng, split_seed

def run_chunk(pair: Pair, protocol: str, start: int, stop: int, output: Path) -> None:
    context = pair_context(pair)
    n, b_count = len(pair.target.ids), stop - start
    if not (0 <= start < stop <= N_BOOT):
        raise ValueError("invalid bootstrap chunk")
    trace: dict[str, Any] = {
        "bootstrap_ids": np.arange(start, stop, dtype=np.uint16),
        "split_seeds": np.zeros((b_count, N_SPLITS), dtype=np.uint32),
        "calib_membership": np.zeros((b_count, N_SPLITS, math.ceil(n / 8)), dtype=np.uint8),
        "eval_membership": np.zeros((b_count, N_SPLITS, math.ceil(n / 8)), dtype=np.uint8),
        "identity_overlap": np.zeros((b_count, N_SPLITS), dtype=np.uint16),
        "screen_proxy_code": np.full((b_count, N_SPLITS), -1, dtype=np.int16),
    }
    for k in KS:
        trace[f"det_idx_k{k}"] = np.zeros((b_count, N_SPLITS, len(DETERMINISTIC), k), dtype=np.uint16)
        trace[f"det_score_k{k}"] = np.zeros((b_count, N_SPLITS, len(DETERMINISTIC), k), dtype=np.float64)
        trace[f"random_idx_k{k}"] = np.zeros((b_count, N_SPLITS, N_RANDOM, k), dtype=np.uint16)

    bootstrap_rng = np.random.RandomState(SEED + 99)
    all_outer = [bootstrap_rng.randint(0, n, n) for _ in range(stop)] if protocol == "round1_compatible" else [np.arange(n)] * stop
    for local_b, b in enumerate(range(start, stop)):
        outer = all_outer[b]
        for s in range(N_SPLITS):
            calib, evaluation, random_rng, split_seed = split_rows(protocol, b, s, outer, n)
            trace["split_seeds"][local_b, s] = split_seed
            trace["calib_membership"][local_b, s] = membership_bits(calib, n)
            trace["eval_membership"][local_b, s] = membership_bits(evaluation, n)
            trace["identity_overlap"][local_b, s] = len(set(calib.tolist()) & set(evaluation.tolist()))
            if protocol == "identity_disjoint" and trace["identity_overlap"][local_b, s] != 0:
                raise AssertionError("identity-disjoint split has non-zero overlap")
            calib_rhos = {proxy: spearman(pair.target.columns[proxy][calib], pair.target.accuracy[calib])
                          for proxy in context["usable"]}
            survivors = [proxy for proxy in context["full"] if sign(calib_rhos[proxy]) == context["signs"][proxy]]
            screen = max(survivors, key=lambda proxy: abs(context["rhos"][proxy])) if survivors else context["source_best"]
            trace["screen_proxy_code"][local_b, s] = context["proxy_codes"][screen]
            score_by_method = {
                "source_best": oriented_score(pair.target.columns[context["source_best"]][evaluation], context["signs"][context["source_best"]]),
                "params": oriented_score(pair.target.columns["params"][evaluation], context["signs"]["params"]),
                "synflow": oriented_score(pair.target.columns["synflow"][evaluation], context["signs"]["synflow"]),
                "rank_ensemble": ensemble_score(pair, evaluation, context["signs"]),
                "sign_stability_screen": oriented_score(pair.target.columns[screen][evaluation], context["signs"][screen]),
            }
            for k in KS:
                for method_i, method in enumerate(DETERMINISTIC):
                    positions = top_positions(score_by_method[method], k)
                    trace[f"det_idx_k{k}"][local_b, s, method_i] = evaluation[positions]
                    trace[f"det_score_k{k}"][local_b, s, method_i] = score_by_method[method][positions]
                for draw in range(N_RANDOM):
                    positions = random_rng.choice(len(evaluation), k, replace=False)
                    trace[f"random_idx_k{k}"][local_b, s, draw] = evaluation[positions]

    root = output / "traces" / protocol / pair.pair_id
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"chunk_{start:04d}_{stop:04d}.npz"
    temporary = destination.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trace)
    os.replace(temporary, destination)
    metadata = {
        "status": "COMPLETE", "protocol": protocol, "pair_id": pair.pair_id,
        "start": start, "stop": stop, "methods": list(METHODS),
        "deterministic_method_order": list(DETERMINISTIC), "K": list(KS),
        "random_draws_per_split": N_RANDOM, "source_best": context["source_best"],
        "rank_ensemble": list(ENSEMBLE), "trace_sha256": file_sha256(destination),
    }
    (root / f"chunk_{start:04d}_{stop:04d}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))

def run_point(pair: Pair, protocol: str, output: Path) -> None:

    context, n = pair_context(pair), len(pair.target.ids)
    trace: dict[str, Any] = {
        "split_seeds": np.arange(SEED + 1, SEED + 1 + N_SPLITS, dtype=np.uint32),
        "calib_membership": np.zeros((N_SPLITS, math.ceil(n / 8)), dtype=np.uint8),
        "eval_membership": np.zeros((N_SPLITS, math.ceil(n / 8)), dtype=np.uint8),
        "identity_overlap": np.zeros(N_SPLITS, dtype=np.uint16),
        "screen_proxy_code": np.full(N_SPLITS, -1, dtype=np.int16),
    }
    for k in KS:
        trace[f"det_idx_k{k}"] = np.zeros((N_SPLITS, len(DETERMINISTIC), k), dtype=np.uint16)
        trace[f"det_score_k{k}"] = np.zeros((N_SPLITS, len(DETERMINISTIC), k), dtype=np.float64)
        trace[f"random_idx_k{k}"] = np.zeros((N_SPLITS, N_RANDOM, k), dtype=np.uint16)
    for s in range(N_SPLITS):
        rng = np.random.RandomState(SEED + 1 + s)
        partition = rng.permutation(n)
        cut = int(round(CALIB_FRAC * n))
        calib, evaluation = partition[:cut], partition[cut:]
        trace["calib_membership"][s] = membership_bits(calib, n)
        trace["eval_membership"][s] = membership_bits(evaluation, n)
        trace["identity_overlap"][s] = 0
        calib_rhos = {proxy: spearman(pair.target.columns[proxy][calib], pair.target.accuracy[calib])
                      for proxy in context["usable"]}
        survivors = [proxy for proxy in context["full"] if sign(calib_rhos[proxy]) == context["signs"][proxy]]
        screen = max(survivors, key=lambda proxy: abs(context["rhos"][proxy])) if survivors else context["source_best"]
        trace["screen_proxy_code"][s] = context["proxy_codes"][screen]
        score_by_method = {
            "source_best": oriented_score(pair.target.columns[context["source_best"]][evaluation], context["signs"][context["source_best"]]),
            "params": oriented_score(pair.target.columns["params"][evaluation], context["signs"]["params"]),
            "synflow": oriented_score(pair.target.columns["synflow"][evaluation], context["signs"]["synflow"]),
            "rank_ensemble": ensemble_score(pair, evaluation, context["signs"]),
            "sign_stability_screen": oriented_score(pair.target.columns[screen][evaluation], context["signs"][screen]),
        }
        for k in KS:
            for method_i, method in enumerate(DETERMINISTIC):
                positions = top_positions(score_by_method[method], k)
                trace[f"det_idx_k{k}"][s, method_i] = evaluation[positions]
                trace[f"det_score_k{k}"][s, method_i] = score_by_method[method][positions]
            for draw in range(N_RANDOM):
                positions = rng.choice(len(evaluation), k, replace=False)
                trace[f"random_idx_k{k}"][s, draw] = evaluation[positions]
    root = output / "traces" / protocol / pair.pair_id
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "point_trace.npz"
    temporary = destination.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trace)
    os.replace(temporary, destination)
    print(json.dumps({"status": "COMPLETE", "protocol": protocol, "pair_id": pair.pair_id,
                      "trace": destination.name, "sha256": file_sha256(destination)}, sort_keys=True))

def resampling_sign_mass(delta: np.ndarray) -> dict[str, float]:

    n = len(delta)
    return {
        "resampling_positive_fraction": float(np.sum(delta > 0) / n),
        "resampling_zero_fraction": float(np.sum(delta == 0) / n),
        "resampling_negative_fraction": float(np.sum(delta < 0) / n),
    }

def resampling_unit(protocol: str) -> str:
    if protocol == "round1_compatible":
        return "paired_target_position_outer_resample"
    return "identity_partition_then_within_side_resample"

def aggregate_protocol(protocol: str, pairs: list[Pair], output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer_values: dict[tuple[str, int, str], np.ndarray] = {}
    point_values: dict[tuple[str, int, str], float] = {}
    for pair in pairs:
        chunks = sorted((output / "traces" / protocol / pair.pair_id).glob("chunk_*.npz"))
        if len(chunks) != 10:
            raise RuntimeError(f"{protocol}/{pair.pair_id}: expected 10 chunks, found {len(chunks)}")
        acc = pair.target.accuracy
        method_values = {(k, method): np.full(N_BOOT, np.nan) for k in KS for method in METHODS}
        for chunk in chunks:
            with np.load(chunk) as data:
                bootstrap_ids = data["bootstrap_ids"].astype(int)
                for k in KS:
                    det = data[f"det_idx_k{k}"].astype(int)
                    for method_i, method in enumerate(DETERMINISTIC):
                        split_means = np.mean(acc[det[:, :, method_i, :]], axis=-1)
                        method_values[(k, method)][bootstrap_ids] = np.mean(split_means, axis=1)
                    random = data[f"random_idx_k{k}"].astype(int)
                    random_split = np.mean(np.mean(acc[random], axis=-1), axis=-1)
                    method_values[(k, "random")][bootstrap_ids] = np.mean(random_split, axis=1)
        for key, value in method_values.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"{protocol}/{pair.pair_id}/{key}: incomplete outer values")
            outer_values[(pair.pair_id, key[0], key[1])] = value
        point_path = output / "traces" / protocol / pair.pair_id / "point_trace.npz"
        if not point_path.is_file():
            raise RuntimeError(f"missing point trace: {point_path}")
        with np.load(point_path) as point:
            for k in KS:
                det = point[f"det_idx_k{k}"].astype(int)
                for method_i, method in enumerate(DETERMINISTIC):
                    point_values[(pair.pair_id, k, method)] = float(
                        np.mean(np.mean(acc[det[:, method_i, :]], axis=-1))
                    )
                random = point[f"random_idx_k{k}"].astype(int)
                point_values[(pair.pair_id, k, "random")] = float(
                    np.mean(np.mean(acc[random], axis=-1))
                )

    method_rows: list[dict[str, Any]] = []
    for pair in pairs:
        for k in KS:
            for method in METHODS:
                array = outer_values[(pair.pair_id, k, method)]
                method_rows.append({
                    "protocol": protocol, "pair_id": pair.pair_id, "role": pair.role,
                    "K": k, "method": method, "n_outer": len(array),
                    "resampling_unit": resampling_unit(protocol),
                    "point_mean_cached_accuracy": point_values[(pair.pair_id, k, method)],
                    "bootstrap_mean_cached_accuracy": float(np.mean(array)),
                    "ci95_low": float(np.percentile(array, 2.5)),
                    "ci95_high": float(np.percentile(array, 97.5)),
                })
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for k in KS:
            for comparator in COMPARATORS:
                delta = (outer_values[(pair.pair_id, k, "sign_stability_screen")] -
                         outer_values[(pair.pair_id, k, comparator)])
                rows.append({
                    "protocol": protocol, "pair_id": pair.pair_id, "role": pair.role, "K": k,
                    "comparison": f"sign_stability_screen_minus_{comparator}", "comparator": comparator,
                    "n_outer": len(delta),
                    "point_delta": (point_values[(pair.pair_id, k, "sign_stability_screen")] -
                                    point_values[(pair.pair_id, k, comparator)]),
                    "bootstrap_mean_delta": float(np.mean(delta)),
                    "ci95_low": float(np.percentile(delta, 2.5)), "ci95_high": float(np.percentile(delta, 97.5)),
                    "resampling_unit": resampling_unit(protocol),
                    **resampling_sign_mass(delta),
                })
    return rows, method_rows

def command_aggregate(pairs: list[Pair], output: Path) -> None:
    rows, methods = [], []
    for protocol in PROTOCOLS:
        protocol_rows, protocol_methods = aggregate_protocol(protocol, pairs, output)
        rows.extend(protocol_rows); methods.extend(protocol_methods)
    fields = list(rows[0])
    with (output / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    method_fields = list(methods[0])
    with (output / "method_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=method_fields); writer.writeheader(); writer.writerows(methods)
    primary = [row for row in rows if row["comparator"] == "source_best" or row["K"] == 10]
    with (output / "paper_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(primary)
    by_key = {(row["protocol"], row["pair_id"], row["K"], row["comparator"]): row for row in rows}
    tex = [r"\\begin{tabular}{llrrrr}",
           "Protocol & Pair & $K=1$ & $K=5$ & $K=10$ & $K=50$ \\\\", r"\\hline"]
    for protocol in PROTOCOLS:
        for pair in pairs:
            cells = []
            for k in KS:
                row = by_key[(protocol, pair.pair_id, k, "source_best")]
                cells.append(f"{float(row['point_delta']):+.3f} [{float(row['ci95_low']):+.3f},{float(row['ci95_high']):+.3f}]")
            tex.append(f"{protocol.replace('_', r'\\_')} & {pair.pair_id.replace('_', r'\\_')} & " + " & ".join(cells) + " \\\\")
    tex.extend([r"\\end{tabular}", "",
                r"% Round-1-compatible intervals resample target positions before CALIB/EVAL splitting; identities may overlap.",
                r"% Identity-disjoint sensitivity intervals partition identities, then resample within each side.",
                r"% Percentile intervals are algorithmic resampling summaries; no inferential p-values are reported.",
                r"% Screen uses target-CALIB cached labels and is not label-free."])
    (output / "paper_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps([row for row in rows if row["K"] == 10 and row["comparator"] == "source_best"], indent=2, sort_keys=True))

def command_validate(pairs: list[Pair], output: Path) -> None:
    checked, overlap_max = 0, 0
    for protocol in PROTOCOLS:
        for pair in pairs:
            starts = []
            for path in sorted((output / "traces" / protocol / pair.pair_id).glob("chunk_*.npz")):
                with np.load(path) as data:
                    ids = data["bootstrap_ids"].astype(int).tolist()
                    starts.extend(ids)
                    overlap = data["identity_overlap"].astype(int)
                    overlap_max = max(overlap_max, int(overlap.max()))
                    if protocol == "identity_disjoint" and np.any(overlap):
                        raise AssertionError(f"{path}: identity overlap")
                    for k in KS:
                        assert data[f"det_idx_k{k}"].shape[1:] == (N_SPLITS, len(DETERMINISTIC), k)
                        assert data[f"random_idx_k{k}"].shape[1:] == (N_SPLITS, N_RANDOM, k)
                checked += 1
            if starts != list(range(N_BOOT)):
                raise AssertionError(f"{protocol}/{pair.pair_id}: incomplete bootstrap IDs")
    rows = list(csv.DictReader((output / "paired_comparisons.csv").open(newline="", encoding="utf-8")))
    if len(rows) != len(PROTOCOLS) * len(pairs) * len(KS) * len(COMPARATORS):
        raise AssertionError("paired comparison count")
    report = {"status": "PASS", "trace_chunks": checked, "paired_comparisons": len(rows),
              "identity_disjoint_overlap_max": 0, "round1_compatible_overlap_max": overlap_max}
    (output / "validation_nested.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run-point", "run-chunk", "aggregate", "validate"))
    parser.add_argument("--frozen-results", "--round1-results", dest="round1_results", type=Path, required=True)
    parser.add_argument("--official-zc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--pair")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    pairs = build_pairs(args.round1_results, args.official_zc)
    if args.command == "prepare":
        for pair in pairs:
            write_manifest(pair, pair_context(pair), args.output)
        print(json.dumps({"status": "READY", "pairs": [pair.pair_id for pair in pairs]}, indent=2))
    elif args.command == "run-chunk":
        selected = [pair for pair in pairs if pair.pair_id == args.pair]
        if len(selected) != 1 or args.protocol is None or args.start is None or args.stop is None:
            raise RuntimeError("run-chunk requires one valid --pair, --protocol, --start, and --stop")
        run_chunk(selected[0], args.protocol, args.start, args.stop, args.output)
    elif args.command == "run-point":
        selected = [pair for pair in pairs if pair.pair_id == args.pair]
        if len(selected) != 1 or args.protocol is None:
            raise RuntimeError("run-point requires one valid --pair and --protocol")
        run_point(selected[0], args.protocol, args.output)
    elif args.command == "aggregate":
        command_aggregate(pairs, args.output)
    else:
        command_validate(pairs, args.output)

if __name__ == "__main__":
    main()
