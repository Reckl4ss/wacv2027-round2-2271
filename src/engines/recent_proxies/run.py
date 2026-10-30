# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

SSS_GENOTYPE = (
    "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+"
    "|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|"
)
SSS_CHANNEL_CANDIDATES = (8, 16, 24, 32, 40, 48, 56, 64)
CIFAR10_MEAN = np.asarray([0.49139968, 0.48215827, 0.44653124], dtype=np.float32)
CIFAR10_STD = np.asarray([0.24703233, 0.24348505, 0.26158768], dtype=np.float32)
BOOTSTRAP_SEED = 20270825

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def standardize_nonfinite_row(row: dict[str, Any]) -> dict[str, Any]:

    output = dict(row)
    markers = dict(output.get("nonfinite_fields", {}))
    for key, value in list(output.items()):
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            numeric = float(value)
            markers[key] = "nan" if math.isnan(numeric) else (
                "positive_infinity" if numeric > 0 else "negative_infinity"
            )
            output[key] = None
    if markers:
        output["nonfinite_fields"] = markers
        output["nonfinite_encoding"] = "JSON null; exact sentinel type is recorded in nonfinite_fields"
    return output

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_safe) + "\n")

def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open_text(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "a") as handle:
        handle.write(json.dumps(standardize_nonfinite_row(row), sort_keys=True, default=json_safe, allow_nan=False) + "\n")
        handle.flush()

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "w") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    standardize_nonfinite_row(row), sort_keys=True, default=json_safe, allow_nan=False
                )
                + "\n"
            )

def rankdata_average(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    if values.ndim != 1 or np.any(np.isnan(values)):
        raise ValueError("rankdata requires a one-dimensional array without NaN; infinities are rankable sentinels")
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

def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    rx, ry = rankdata_average(x), rankdata_average(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, *, draws: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    n = len(x)
    for draw in range(draws):
        indexes = rng.integers(0, n, size=n)
        samples[draw] = spearman(x[indexes], y[indexes])
    finite = samples[np.isfinite(samples)]
    if len(finite) != draws:
        raise AssertionError(f"non-finite bootstrap draws: {draws - len(finite)}")
    return {
        "draws": draws,
        "seed": seed,
        "ci_percentile": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
        "probability_positive": float(np.mean(finite > 0)),
        "probability_negative": float(np.mean(finite < 0)),
    }

def decode_sss_index(index: int) -> str:
    if not 0 <= index < len(SSS_CHANNEL_CANDIDATES) ** 5:
        raise ValueError(f"invalid NATS-SSS index: {index}")
    digits = [0] * 5
    value = index
    for position in range(4, -1, -1):
        digits[position] = value % 8
        value //= 8
    return ":".join(str(SSS_CHANNEL_CANDIDATES[digit]) for digit in digits)

def load_populations(round1_results: Path) -> dict[str, list[dict[str, Any]]]:
    tss = pd.read_pickle(round1_results / "base_tss_cifar10.pkl")
    sss = pd.read_pickle(round1_results / "base_sss_cifar10.pkl")
    arch_strings = pd.read_pickle(round1_results / "_tss_archstr.pkl")
    if not isinstance(tss, list) or len(tss) != 1000:
        raise AssertionError(f"expected 1000 TSS rows, got {type(tss)} / {len(tss)}")
    if not isinstance(sss, list) or len(sss) != 425:
        raise AssertionError(f"expected 425 SSS rows, got {type(sss)} / {len(sss)}")
    if not isinstance(arch_strings, dict):
        raise AssertionError("TSS architecture map must be a dict")
    populations = {"tss": [], "sss": []}
    for space, rows in (("tss", tss), ("sss", sss)):
        seen: set[int] = set()
        for row in rows:
            architecture_id = int(row["i"])
            if architecture_id in seen:
                raise AssertionError(f"duplicate {space} architecture ID: {architecture_id}")
            seen.add(architecture_id)
            if row["space"] != space or row["dataset"] != "cifar10":
                raise AssertionError(f"population tag mismatch: {row}")
            item = dict(row)
            item["architecture_id"] = architecture_id
            item["architecture_key"] = f"nats-{space}:{architecture_id}"
            item["architecture_string"] = (
                str(arch_strings[architecture_id]) if space == "tss" else decode_sss_index(architecture_id)
            )
            populations[space].append(item)
    return populations

def import_official_sources(swap_repo: Path, az_repo: Path):
    sys.path.insert(0, str(swap_repo))
    from src.metrics.swap import SWAP  
    from src.utils.utilities import network_weight_gaussian_init  

    nb201 = az_repo / "NB201"
    sys.path.insert(0, str(nb201))
    from custom.tss_model import TinyNetwork  
    from xautodl.models.cell_searchs.genotypes import Structure  
    from xautodl.models.shape_infers import DynamicShapeTinyNet  
    from ZeroShotProxy.compute_az_nas_score import compute_nas_score  

    return SWAP, network_weight_gaussian_init, TinyNetwork, Structure, DynamicShapeTinyNet, compute_nas_score

def build_model(space: str, architecture_string: str, official: tuple[Any, ...]):
    _, _, TinyNetwork, Structure, DynamicShapeTinyNet, _ = official
    if space == "tss":
        genotype = Structure.str2structure(architecture_string)
        return TinyNetwork(C=16, N=5, genotype=genotype, num_classes=10)
    if space == "sss":
        channels = tuple(int(value) for value in architecture_string.split(":"))
        genotype = Structure.str2structure(SSS_GENOTYPE)
        model = DynamicShapeTinyNet(channels, genotype, 10)

        def extract_cell_features(self, inputs):
            features = []
            feature = self.stem(inputs)
            if feature.requires_grad:
                feature.retain_grad()
            features.append(feature)
            for cell in self.cells:
                feature = cell(feature)
                if feature.requires_grad:
                    feature.retain_grad()
                features.append(feature)
            return features

        model.extract_cell_features = types.MethodType(extract_cell_features, model)
        return model
    raise ValueError(space)

def actual_parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))

def assert_model_identity(row: dict[str, Any], model: torch.nn.Module) -> int:
    actual = actual_parameter_count(model)
    expected = float(row["params"]) * 1_000_000.0
    if abs(actual - expected) > 0.500001:
        raise AssertionError(
            f"model identity mismatch for {row['architecture_key']}: actual={actual}, cached={expected}"
        )
    return actual

def load_cifar_batch(path: Path) -> torch.Tensor:
    raw = np.load(path, allow_pickle=False)
    if raw.shape != (16, 3, 32, 32) or raw.dtype != np.uint8:
        raise AssertionError(f"invalid CIFAR batch: shape={raw.shape}, dtype={raw.dtype}")
    values = raw.astype(np.float32) / 255.0
    values = (values - CIFAR10_MEAN[None, :, None, None]) / CIFAR10_STD[None, :, None, None]
    return torch.from_numpy(values)

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def base_output(row: dict[str, Any], actual_params: int, method: str, repeat: int, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": method,
        "space": row["space"],
        "dataset": "cifar10",
        "architecture_id": row["architecture_id"],
        "architecture_key": row["architecture_key"],
        "architecture_string": row["architecture_string"],
        "repeat": repeat,
        "seed": seed,
        "cached_accuracy_test_percent": float(row["testacc"]),
        "cached_accuracy_validation_percent": float(row["valacc"]),
        "cached_flops_million": float(row["flops"]),
        "cached_params_million": float(row["params"]),
        "actual_parameter_count": actual_params,
        "model_identity_exact": True,
    }

def run_swap(
    row: dict[str, Any], model: torch.nn.Module, inputs: torch.Tensor, repeats: int,
    raw_path: Path, completed: set[tuple[str, str, int, int]], official: tuple[Any, ...], device: str,
) -> None:
    SWAP, network_weight_gaussian_init, *_ = official
    model = model.to(device)
    actual_params = assert_model_identity(row, model)
    metric = SWAP(model=model, inputs=inputs, device=device, seed=0, regular=False)
    for repeat in range(repeats):
        key = ("swap", row["space"], row["architecture_id"], repeat)
        if key in completed:
            continue
        seed = repeat
        seed_all(seed)
        model.apply(network_weight_gaussian_init)
        metric.reinit()
        started = time.perf_counter()
        score_tensor = metric.forward()
        if score_tensor is None:
            raise RuntimeError(f"SWAP collected no ReLU activations for {row['architecture_key']}")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        score = float(score_tensor.item() if hasattr(score_tensor, "item") else score_tensor)
        output = base_output(row, actual_params, "swap", repeat, seed)
        output.update({"swap_score": score, "elapsed_seconds": elapsed, "status": "completed"})
        append_jsonl(raw_path, output)
        completed.add(key)
        metric.clear()
    del metric, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

def shared_az_input(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(100_000 + seed)
    return torch.randn((64, 3, 32, 32), generator=generator, dtype=torch.float32)

def run_aznas(
    row: dict[str, Any], model: torch.nn.Module, seeds: list[int], raw_path: Path,
    completed: set[tuple[str, str, int, int]], official: tuple[Any, ...], gpu: int,
) -> None:
    *_, compute_nas_score = official
    actual_params = assert_model_identity(row, model)
    for repeat, seed in enumerate(seeds):
        key = ("aznas", row["space"], row["architecture_id"], repeat)
        if key in completed:
            continue
        seed_all(seed)
        fixed_batch = shared_az_input(seed)
        started = time.perf_counter()
        scores = compute_nas_score(
            model=model,
            gpu=gpu,
            trainloader=[(fixed_batch, None)],
            resolution=32,
            batch_size=64,
            init_method="kaiming_norm_fanin",
            fp16=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        output = base_output(row, actual_params, "aznas", repeat, seed)
        output.update(
            {
                "az_expressivity": float(scores["expressivity"]),
                "az_progressivity": float(scores["progressivity"]),
                "az_trainability": float(scores["trainability"]),
                "elapsed_seconds": elapsed,
                "status": "completed",
            }
        )
        append_jsonl(raw_path, output)
        completed.add(key)
        del fixed_batch
    del model
    torch.cuda.empty_cache()

def aggregate(raw_path: Path, output_dir: Path, bootstrap_draws: int) -> dict[str, Any]:
    rows = load_jsonl(raw_path)
    if not rows:
        raise ValueError(f"no raw rows in {raw_path}")
    by_group: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = (row["method"], row["space"], int(row["architecture_id"]))
        by_group.setdefault(key, []).append(row)

    architecture_rows: list[dict[str, Any]] = []
    for (method, space, architecture_id), group in sorted(by_group.items()):
        first = group[0]
        item = {
            key: first[key]
            for key in (
                "method", "space", "dataset", "architecture_id", "architecture_key",
                "architecture_string", "cached_accuracy_test_percent",
                "cached_accuracy_validation_percent", "cached_flops_million",
                "cached_params_million", "actual_parameter_count", "model_identity_exact",
            )
        }
        item["n_repeats"] = len(group)
        if method == "swap":
            values = np.asarray([row["swap_score"] for row in group], dtype=np.float64)
            item["proxy_score"] = float(np.mean(values))
            item["proxy_score_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        elif method == "aznas":
            for component in ("expressivity", "progressivity", "trainability"):
                values = np.asarray([row[f"az_{component}"] for row in group], dtype=np.float64)
                negative_infinity = int(np.sum(np.isneginf(values)))
                positive_infinity = int(np.sum(np.isposinf(values)))
                nan_count = int(np.sum(np.isnan(values)))
                if positive_infinity or nan_count:
                    raise AssertionError(
                        f"unsupported AZ sentinel for {space}/{architecture_id}/{component}: "
                        f"positive_infinity={positive_infinity}, nan={nan_count}"
                    )
                item[f"az_{component}_mean"] = float(np.mean(values))
                finite_values = values[np.isfinite(values)]
                item[f"az_{component}_finite_mean"] = (
                    float(np.mean(finite_values)) if len(finite_values) else None
                )
                item[f"az_{component}_finite_std"] = (
                    float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0
                )
                item[f"az_{component}_negative_infinity_repeats"] = negative_infinity
        architecture_rows.append(item)

    for space in ("tss", "sss"):
        selected = [row for row in architecture_rows if row["method"] == "aznas" and row["space"] == space]
        if selected:
            count = len(selected)
            aggregate_score = np.zeros(count, dtype=np.float64)
            for field in (
                "cached_flops_million", "az_expressivity_mean",
                "az_progressivity_mean", "az_trainability_mean",
            ):
                ranks = rankdata_average([row[field] for row in selected])
                aggregate_score += np.log(ranks / count)
            for row, score in zip(selected, aggregate_score):
                row["proxy_score"] = float(score)

    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_raw_path = output_dir / "raw_scores.jsonl.gz"
    write_jsonl(sanitized_raw_path, rows)
    architecture_path = output_dir / "architecture_scores.jsonl.gz"
    write_jsonl(architecture_path, architecture_rows)

    summary_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {"schema_version": 1, "bootstrap_draws": bootstrap_draws, "methods": {}}
    for method in sorted({row["method"] for row in architecture_rows}):
        results["methods"][method] = {}
        for space in ("tss", "sss"):
            selected = [row for row in architecture_rows if row["method"] == method and row["space"] == space]
            if not selected:
                continue
            scores = np.asarray([row["proxy_score"] for row in selected], dtype=np.float64)
            metrics: dict[str, Any] = {"n": len(selected)}
            for target_name, target_field in (
                ("test_accuracy", "cached_accuracy_test_percent"),
                ("validation_accuracy", "cached_accuracy_validation_percent"),
            ):
                targets = np.asarray([row[target_field] for row in selected], dtype=np.float64)
                rho = spearman(scores, targets)
                boot = bootstrap_spearman(
                    scores, targets, draws=bootstrap_draws,
                    seed=BOOTSTRAP_SEED + (0 if method == "swap" else 10_000) + (0 if space == "tss" else 1_000) + (0 if target_name == "test_accuracy" else 100),
                )
                metrics[target_name] = {"spearman_rho": rho, **boot}
                summary_rows.append(
                    {
                        "method": method,
                        "space": space,
                        "target": target_name,
                        "n": len(selected),
                        "spearman_rho": rho,
                        "ci_low": boot["ci_percentile"][0],
                        "ci_high": boot["ci_percentile"][1],
                        "bootstrap_draws": bootstrap_draws,
                        "bootstrap_seed": boot["seed"],
                    }
                )
            results["methods"][method][space] = metrics
        if all(space in results["methods"][method] for space in ("tss", "sss")):
            tss_rho = results["methods"][method]["tss"]["test_accuracy"]["spearman_rho"]
            sss_rho = results["methods"][method]["sss"]["test_accuracy"]["spearman_rho"]
            results["methods"][method]["test_accuracy_sign_stability"] = {
                "same_nonzero_point_sign": bool(tss_rho != 0 and sss_rho != 0 and np.sign(tss_rho) == np.sign(sss_rho)),
                "definition": "Point-estimate signs agree and neither point estimate is zero; this does not establish universality.",
            }

    import csv
    with (output_dir / "correlations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output_dir / "results.json", results)
    return results

def parse_ids(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item) for item in value.split(",") if item.strip()}

def selected_population(
    populations: dict[str, list[dict[str, Any]]], limit: int | None,
    ids_tss: set[int] | None, ids_sss: set[int] | None,
):
    for space in ("tss", "sss"):
        rows = populations[space]
        requested = ids_tss if space == "tss" else ids_sss
        if requested is not None:
            selected = [row for row in rows if row["architecture_id"] in requested]
            found = {row["architecture_id"] for row in selected}
            if found != requested:
                raise ValueError(f"{space} IDs absent from the locked population: {sorted(requested - found)}")
        else:
            selected = rows if limit is None else rows[:limit]
        yield space, selected

def environment_payload() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("identity", "run", "aggregate", "validate"), required=True)
    parser.add_argument("--method", choices=("swap", "aznas", "both"), default="both")
    parser.add_argument("--frozen-results", "--round1-results", dest="round1_results", type=Path, required=True)
    parser.add_argument("--cifar-first16", type=Path)
    parser.add_argument("--swap-repo", type=Path)
    parser.add_argument("--az-repo", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-file", default="raw_scores.jsonl")
    parser.add_argument("--limit-per-space", type=int)
    parser.add_argument("--ids-tss")
    parser.add_argument("--ids-sss")
    parser.add_argument("--swap-repeats", type=int, default=32)
    parser.add_argument("--az-seeds", default="1,2,3,4,5")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    raw_path = args.output_dir / args.raw_file
    if args.mode == "aggregate":
        results = aggregate(raw_path, args.output_dir, args.bootstrap_draws)
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    populations = load_populations(args.round1_results)
    if args.mode == "validate":
        results = json.loads((args.output_dir / "results.json").read_text())
        architecture_rows = load_jsonl(args.output_dir / "architecture_scores.jsonl.gz")
        expected = {"tss": 1000, "sss": 425}
        for method, method_results in results["methods"].items():
            for space, count in expected.items():
                actual = sum(row["method"] == method and row["space"] == space for row in architecture_rows)
                if actual != count or method_results[space]["n"] != count:
                    raise AssertionError(f"{method}/{space}: expected {count}, got {actual}")
        if not all(bool(row["model_identity_exact"]) for row in architecture_rows):
            raise AssertionError("model identity gate failed")
        print(json.dumps({"status": "PASS", "architecture_rows": len(architecture_rows), "methods": sorted(results["methods"])}, indent=2))
        return

    for required in (args.swap_repo, args.az_repo):
        if required is None or not required.exists():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment_runtime.json", environment_payload())
    official = import_official_sources(args.swap_repo, args.az_repo)
    chosen = list(
        selected_population(
            populations, args.limit_per_space, parse_ids(args.ids_tss), parse_ids(args.ids_sss)
        )
    )
    if args.mode == "identity":
        gate_path = args.output_dir / "identity_gate.jsonl.gz"
        if gate_path.exists():
            gate_path.unlink()
        total = 0
        for space, rows in chosen:
            for row in rows:
                model = build_model(space, row["architecture_string"], official)
                actual = assert_model_identity(row, model)
                append_jsonl(
                    gate_path,
                    {
                        "space": space,
                        "architecture_id": row["architecture_id"],
                        "architecture_key": row["architecture_key"],
                        "architecture_string": row["architecture_string"],
                        "cached_params_million": float(row["params"]),
                        "actual_parameter_count": actual,
                        "model_identity_exact": True,
                    },
                )
                total += 1
                del model
        summary = {"status": "PASS", "architectures": total, "tss": len(chosen[0][1]), "sss": len(chosen[1][1])}
        write_json(args.output_dir / "identity_gate_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("formal E6 scoring requires CUDA")
    if args.cifar_first16 is None or not args.cifar_first16.exists():
        raise FileNotFoundError(args.cifar_first16)
    inputs = load_cifar_batch(args.cifar_first16)
    previous = load_jsonl(raw_path)
    completed = {
        (row["method"], row["space"], int(row["architecture_id"]), int(row["repeat"]))
        for row in previous if row.get("status") == "completed"
    }
    methods = ("swap", "aznas") if args.method == "both" else (args.method,)
    az_seeds = [int(value) for value in args.az_seeds.split(",") if value]
    print(json.dumps({"mode": "run", "methods": methods, "limit_per_space": args.limit_per_space, "already_completed": len(completed), "environment": environment_payload()}, sort_keys=True))
    for space, rows in chosen:
        for position, row in enumerate(rows, start=1):
            for method in methods:
                started = time.perf_counter()
                seed_all(0)
                model = build_model(space, row["architecture_string"], official)
                if method == "swap":
                    run_swap(row, model, inputs, args.swap_repeats, raw_path, completed, official, "cuda:0")
                else:
                    run_aznas(row, model, az_seeds, raw_path, completed, official, 0)
                print(json.dumps({"method": method, "space": space, "architecture_id": row["architecture_id"], "position": position, "total": len(rows), "wall_seconds": time.perf_counter() - started}, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
