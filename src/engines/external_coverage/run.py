# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "1.1.0"
AVAILABLE_PROXY_FIELDS = (
    "epe_nas",
    "fisher",
    "flops",
    "grad_norm",
    "grasp",
    "jacov",
    "l2_norm",
    "nwot",
    "params",
    "plain",
    "snip",
    "synflow",
    "zen",
)
REQUIRED_PROXY_COUNT = 21

REQUIRED_PROXY_FIELDS: tuple[str, ...] | None = None
REQUIRED_PROXY_FIELDS_STATUS = "blocked_not_frozen_in_authoritative_round1_source"
EXPECTED_SHA256 = {
    "zc_nasbench101.json": "aaf67baf526661c84b9c2bd449926cd0ff41d9451fe0aeaeba3450013ec6801f",
    "zc_nasbench301.json": "8951464c59220053b5468a694604e209b9fe9692ef024b00cc0db12ef2f11536",
    "zc_transbench101_micro.json": "60934abb09b682abe0e1ca44acf08a1162e3001f34960ac78fda9f8867525c52",
    "nasbench_only108.pkl": "992337e87be79cfc379195f1b8ba1560741a15501b9a1a36ad315da8decbafcb",
    "nb301_full_training.pickle": "1f658483ff85b92a6318fdaf4478c60ec42ea3b96af580498b1fb80cdf11fb92",
    "transnas-bench_v10141024.pth": "1974b0ba21872494fabb541616e9bdae740242eecd6df5c5893747291364bc10",
}
NB101_OPS = [
    "input",
    "output",
    "maxpool3x3",
    "conv1x1-bn-relu",
    "conv3x3-bn-relu",
]
TRANSNAS_METRICS = {
    "class_scene": ("train_top1", "valid_top1"),
    "class_object": ("train_top1", "valid_top1"),
    "jigsaw": ("train_top1", "valid_top1"),
    "room_layout": ("train_neg_loss", "valid_neg_loss"),
    "segmentsemantic": ("train_acc", "valid_acc"),
    "normal": ("train_ssim", "valid_ssim"),
    "autoencoder": ("train_ssim", "valid_ssim"),
}
EXACT_PAPER_T_METRICS = {"train_top1", "train_acc"}
EXACT_FINAL_ACCURACY_METRICS = {"valid_top1", "valid_acc"}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def stable_id(prefix: str, raw: str) -> str:
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"

def score_values(row: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for proxy in AVAILABLE_PROXY_FIELDS:
        value = row.get(proxy)
        if isinstance(value, dict):
            value = value.get("score")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = math.nan
        out[proxy] = number if math.isfinite(number) else None
    return out

def first5_slope(values: Any) -> float | None:
    if not isinstance(values, (list, tuple)) or len(values) < 5:
        return None
    y = np.asarray(values[:5], dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    x = np.arange(5, dtype=float)
    return float(np.polyfit(x, y, 1)[0])

def exact_float(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError, OverflowError):
        return False

def gzip_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(gz, encoding="utf-8", newline="\n")

def write_records(path: Path, records: Iterable[dict[str, Any]]) -> Counter:
    tmp = path.with_suffix(path.suffix + ".part")
    counts: Counter = Counter()
    with gzip_writer(tmp) as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
            f.write("\n")
            counts["rows"] += 1
            for key in (
                "curve_joined",
                "q_13_complete",
                "capacity_finite",
                "erank_finite",
                "task_native_early_metric_finite",
                "exact_paper_t_finite",
                "task_native_final_metric_exact_match",
                "exact_final_accuracy_match",
                "eq1_eligible",
            ):
                counts[key] += int(bool(record[key]))
    os.replace(tmp, path)
    return counts

def common_record(
    benchmark: str,
    task: str,
    canonical_id: str,
    raw_key: str,
    zc_row: dict[str, Any],
) -> dict[str, Any]:
    q = score_values(zc_row)
    q_count = sum(value is not None for value in q.values())
    required_names_auditable = REQUIRED_PROXY_FIELDS is not None
    required_complete = bool(
        required_names_auditable
        and all(q.get(name) is not None for name in REQUIRED_PROXY_FIELDS)
    )
    capacity = q["params"]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "task": task,
        "canonical_architecture_id": canonical_id,
        "raw_zc_architecture_key": raw_key,
        "zc_row_id": zc_row.get("id"),
        "q_values": q,
        "q_available_fields": list(AVAILABLE_PROXY_FIELDS),
        "q_finite_count": q_count,
        "q_13_complete": q_count == len(AVAILABLE_PROXY_FIELDS),
        "q_required_count": REQUIRED_PROXY_COUNT,
        "q_required_count_gap": max(0, REQUIRED_PROXY_COUNT - q_count),
        "q_required_fields": list(REQUIRED_PROXY_FIELDS) if REQUIRED_PROXY_FIELDS else None,
        "q_required_fields_status": REQUIRED_PROXY_FIELDS_STATUS,
        "q_required_missing_fields": (
            sorted(set(REQUIRED_PROXY_FIELDS) - {name for name, value in q.items() if value is not None})
            if REQUIRED_PROXY_FIELDS else None
        ),
        "q_required_names_auditable": required_names_auditable,
        "q_required_21_complete": required_complete,
        "capacity_name": "params",
        "capacity_value": capacity,
        "capacity_finite": capacity is not None,
        "erank_value": None,
        "erank_finite": False,
        "task_native_early_metric_name": None,
        "task_native_early_metric_value": None,
        "task_native_early_metric_finite": False,
        "exact_paper_t_name": "first5_train_accuracy_ols_slope",
        "exact_paper_t_value": None,
        "exact_paper_t_finite": False,
        "zc_final_metric_name": "val_accuracy",
        "zc_final_metric_value": float(zc_row["val_accuracy"]),
        "task_native_final_metric_name": None,
        "task_native_final_metric_value": None,
        "task_native_final_metric_exact_match": False,
        "exact_final_accuracy_name": None,
        "exact_final_accuracy_value": None,
        "exact_final_accuracy_match": False,
        "curve_joined": False,
        "join_status": "unattempted",
        "missing_fields": [],
        "eq1_eligible": False,
    }

def nb101_signature(fixed: dict[str, Any]) -> tuple[int, ...]:
    matrix = tuple(int(x) for x in np.asarray(fixed["module_adjacency"]).flatten())
    operations = tuple(NB101_OPS.index(op) for op in fixed["module_operations"])
    return matrix + operations

def audit_nb101(zc_path: Path, history_path: Path, output: Path) -> tuple[Counter, dict[str, Any]]:
    zc = json.loads(zc_path.read_text(encoding="utf-8"))["cifar10"]
    with history_path.open("rb") as f:
        history = pickle.load(f)
    signature_map: dict[tuple[int, ...], tuple[str, Any]] = {}
    for module_hash, value in history.items():
        signature = nb101_signature(value[0])
        if signature in signature_map:
            raise RuntimeError("duplicate NB101 canonical signature")
        signature_map[signature] = (module_hash, value)

    def records():
        for raw_key, zc_row in zc.items():
            signature = tuple(int(x) for x in ast.literal_eval(raw_key))
            joined = signature_map.get(signature)
            module_hash = joined[0] if joined else None
            record = common_record(
                "NAS-Bench-101",
                "cifar10",
                f"nb101:{module_hash}" if module_hash else stable_id("nb101_unmatched", raw_key),
                raw_key,
                zc_row,
            )
            if joined:
                fixed, computed = joined[1]
                repeats = computed.get(108, [])
                first_final = (
                    float(repeats[0]["final_validation_accuracy"]) * 100.0
                    if repeats
                    else None
                )
                record.update(
                    {
                        "curve_joined": True,
                        "join_status": "exact_structural_signature",
                        "official_module_hash": module_hash,
                        "official_capacity_name": "trainable_parameters",
                        "official_capacity_value": int(fixed["trainable_parameters"]),
                        "capacity_scale_exact_match": (
                            record["capacity_value"] is not None
                            and round(float(record["capacity_value"]) * 1_000_000.0)
                            == int(fixed["trainable_parameters"])
                        ),
                        "exact_paper_t_blocker": (
                            "only108 pickle stores halfway/final metrics, not epochs 1-5"
                        ),
                        "task_native_final_metric_name": "repeat0_final_validation_accuracy_percent",
                        "task_native_final_metric_value": first_final,
                        "task_native_final_metric_exact_match": exact_float(
                            record["zc_final_metric_value"], first_final
                        ),
                        "exact_final_accuracy_name": "repeat0_final_validation_accuracy_percent",
                        "exact_final_accuracy_value": first_final,
                        "exact_final_accuracy_match": exact_float(
                            record["zc_final_metric_value"], first_final
                        ),
                    }
                )
            missing = []
            if record["q_required_count_gap"]:
                missing.append(f"{record['q_required_count_gap']}_required_proxy_count_gap")
            if not record["q_required_names_auditable"]:
                missing.append("required_proxy_name_set_not_frozen")
            if not record["erank_finite"]:
                missing.append("erank")
            if not record["exact_paper_t_finite"]:
                missing.append("exact_first5_train_accuracy_T")
            if not record["exact_final_accuracy_match"]:
                missing.append("exact_final_accuracy_match")
            record["missing_fields"] = missing
            record["eq1_eligible"] = not missing
            yield record

    counts = write_records(output / "join_records" / "nb101_cifar10.jsonl.gz", records())
    return counts, {"zc_rows": len(zc), "official_history_rows": len(history)}

def audit_nb301(zc_path: Path, history_path: Path, output: Path) -> tuple[Counter, dict[str, Any]]:
    zc = json.loads(zc_path.read_text(encoding="utf-8"))["cifar10"]
    with history_path.open("rb") as f:
        history = pickle.load(f)

    def records():
        for raw_key, zc_row in zc.items():
            genotype = ast.literal_eval(raw_key)
            curve = history.get(genotype)
            record = common_record(
                "NAS-Bench-301",
                "cifar10",
                stable_id("nb301", raw_key),
                raw_key,
                zc_row,
            )
            if curve is not None:
                val_curve = curve.get("val_accuracies", [])
                final = float(val_curve[-1]) if val_curve else None
                record.update(
                    {
                        "curve_joined": True,
                        "join_status": "exact_genotype_tuple",
                        "exact_paper_t_blocker": (
                            "history stores train_losses and val_accuracies, not train_accuracy"
                        ),
                        "available_non_equivalent_first5_train_loss": (
                            first5_slope(curve.get("train_losses"))
                        ),
                        "task_native_final_metric_name": "final_val_accuracy",
                        "task_native_final_metric_value": final,
                        "task_native_final_metric_exact_match": exact_float(
                            record["zc_final_metric_value"], final
                        ),
                        "exact_final_accuracy_name": "final_val_accuracy",
                        "exact_final_accuracy_value": final,
                        "exact_final_accuracy_match": exact_float(
                            record["zc_final_metric_value"], final
                        ),
                    }
                )
            missing = []
            if record["q_required_count_gap"]:
                missing.append(f"{record['q_required_count_gap']}_required_proxy_count_gap")
            if not record["q_required_names_auditable"]:
                missing.append("required_proxy_name_set_not_frozen")
            if not record["erank_finite"]:
                missing.append("erank")
            if not record["exact_paper_t_finite"]:
                missing.append("exact_first5_train_accuracy_T")
            if not record["exact_final_accuracy_match"]:
                missing.append("exact_final_accuracy_match")
            record["missing_fields"] = missing
            record["eq1_eligible"] = not missing
            yield record

    counts = write_records(output / "join_records" / "nb301_cifar10.jsonl.gz", records())
    return counts, {"zc_rows": len(zc), "official_history_rows": len(history)}

def transnas_arch(raw_key: str) -> str:
    values = tuple(int(x) for x in ast.literal_eval(raw_key))
    if len(values) != 6:
        raise ValueError(f"TransNAS micro encoding length {len(values)}")
    return "64-41414-{}_{}{}_{}{}{}".format(*values)

def audit_transnas(zc_path: Path, api_path: Path, output: Path) -> list[tuple[str, Counter, dict[str, Any]]]:
    import torch

    zc_all = json.loads(zc_path.read_text(encoding="utf-8"))
    database = torch.load(api_path, map_location="cpu", weights_only=False)
    micro = database["data"]["micro"]
    official_order = {arch: index for index, arch in enumerate(micro)}
    results = []
    for task, (train_metric, final_metric) in TRANSNAS_METRICS.items():
        zc = zc_all[task]

        def records(task=task, train_metric=train_metric, final_metric=final_metric, zc=zc):
            for raw_key, zc_row in zc.items():
                arch = transnas_arch(raw_key)
                official = micro.get(arch, {}).get(task)
                record = common_record(
                    "TransNAS-Bench-101/micro",
                    task,
                    f"transnas_micro:{arch}",
                    raw_key,
                    zc_row,
                )
                if official is not None:
                    metrics = official["metrics"]
                    train_curve = metrics.get(train_metric)
                    final_curve = metrics.get(final_metric)
                    t_value = first5_slope(train_curve)
                    final_value = float(final_curve[-1]) if final_curve else None
                    exact_t = train_metric in EXACT_PAPER_T_METRICS
                    exact_accuracy = final_metric in EXACT_FINAL_ACCURACY_METRICS
                    final_matches = exact_float(record["zc_final_metric_value"], final_value)
                    record.update(
                        {
                            "curve_joined": True,
                            "join_status": "exact_official_architecture_string",
                            "official_architecture_string": arch,
                            "official_row_index": official_order[arch],
                            "zc_id_equals_official_row_index": (
                                int(zc_row["id"]) == official_order[arch]
                            ),
                            "task_native_early_metric_name": f"first5_{train_metric}_ols_slope",
                            "task_native_early_metric_value": t_value,
                            "task_native_early_metric_finite": t_value is not None,
                            "exact_paper_t_name": (
                                f"first5_{train_metric}_ols_slope" if exact_t else None
                            ),
                            "exact_paper_t_value": t_value if exact_t else None,
                            "exact_paper_t_finite": bool(exact_t and t_value is not None),
                            "exact_paper_t_blocker": (
                                None if exact_t else f"{train_metric} is task-native, not training accuracy"
                            ),
                            "task_native_final_metric_name": f"final_{final_metric}",
                            "task_native_final_metric_value": final_value,
                            "task_native_final_metric_exact_match": final_matches,
                            "exact_final_accuracy_name": (
                                f"final_{final_metric}" if exact_accuracy else None
                            ),
                            "exact_final_accuracy_value": final_value if exact_accuracy else None,
                            "exact_final_accuracy_match": bool(exact_accuracy and final_matches),
                            "exact_final_accuracy_blocker": (
                                None if exact_accuracy else f"{final_metric} is task-native, not accuracy"
                            ),
                            "official_capacity_name": "model_params",
                            "official_capacity_value": official["model_info"].get("model_params"),
                        }
                    )
                missing = []
                if record["q_required_count_gap"]:
                    missing.append(f"{record['q_required_count_gap']}_required_proxy_count_gap")
                if not record["q_required_names_auditable"]:
                    missing.append("required_proxy_name_set_not_frozen")
                if not record["erank_finite"]:
                    missing.append("erank")
                if not record["exact_paper_t_finite"]:
                    missing.append("exact_first5_train_accuracy_T")
                if not record["exact_final_accuracy_match"]:
                    missing.append("exact_final_accuracy_match")
                record["missing_fields"] = missing
                record["eq1_eligible"] = not missing
                yield record

        counts = write_records(
            output / "join_records" / f"transnas_micro_{task}.jsonl.gz", records()
        )
        results.append(
            (
                task,
                counts,
                {
                    "zc_rows": len(zc),
                    "official_micro_rows": len(micro),
                    "train_metric": train_metric,
                    "final_metric": final_metric,
                },
            )
        )
    return results

def verify_inputs(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "zc_nasbench101.json": args.zc_dir / "zc_nasbench101.json",
        "zc_nasbench301.json": args.zc_dir / "zc_nasbench301.json",
        "zc_transbench101_micro.json": args.zc_dir / "zc_transbench101_micro.json",
        "nasbench_only108.pkl": args.nb101,
        "nb301_full_training.pickle": args.nb301,
        "transnas-bench_v10141024.pth": args.transnas,
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256[name]:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {digest}")
        actual[name] = digest
    return actual

def command_prepare(args: argparse.Namespace) -> None:
    actual = verify_inputs(args)
    report = {"schema_version": SCHEMA_VERSION, "status": "PASS", "input_sha256": actual}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "input_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))

def command_audit(args: argparse.Namespace) -> None:
    verify_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    join_rows: list[dict[str, Any]] = []
    nb101_counts, nb101_meta = audit_nb101(
        args.zc_dir / "zc_nasbench101.json", args.nb101, args.output
    )
    q_contract = {
        "q_required_count": REQUIRED_PROXY_COUNT,
        "q_required_count_gap": REQUIRED_PROXY_COUNT - len(AVAILABLE_PROXY_FIELDS),
        "q_required_names_auditable": 0,
        "q_required_fields_status": REQUIRED_PROXY_FIELDS_STATUS,
    }
    rows.append({"benchmark": "NAS-Bench-101", "task": "cifar10", **nb101_counts, **q_contract})
    join_rows.append({"benchmark": "NAS-Bench-101", "task": "cifar10", **nb101_meta})
    nb301_counts, nb301_meta = audit_nb301(
        args.zc_dir / "zc_nasbench301.json", args.nb301, args.output
    )
    rows.append({"benchmark": "NAS-Bench-301", "task": "cifar10", **nb301_counts, **q_contract})
    join_rows.append({"benchmark": "NAS-Bench-301", "task": "cifar10", **nb301_meta})
    for task, counts, meta in audit_transnas(
        args.zc_dir / "zc_transbench101_micro.json", args.transnas, args.output
    ):
        rows.append(
            {"benchmark": "TransNAS-Bench-101/micro", "task": task, **counts, **q_contract}
        )
        join_rows.append({"benchmark": "TransNAS-Bench-101/micro", "task": task, **meta})

    coverage_fields = [
        "benchmark",
        "task",
        "rows",
        "curve_joined",
        "q_13_complete",
        "q_required_count",
        "q_required_count_gap",
        "q_required_names_auditable",
        "q_required_fields_status",
        "capacity_finite",
        "erank_finite",
        "task_native_early_metric_finite",
        "exact_paper_t_finite",
        "task_native_final_metric_exact_match",
        "exact_final_accuracy_match",
        "eq1_eligible",
    ]
    with (args.output / "field_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(rows)
    join_fields = sorted({key for row in join_rows for key in row})
    with (args.output / "join_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=join_fields)
        writer.writeheader()
        writer.writerows(join_rows)
    blockers = {
        "schema_version": SCHEMA_VERSION,
        "eq1_completed": False,
        "global_blockers": [
            "Official ZC inputs contain 13 finite proxy fields versus a required count of 21; the count gap is 8.",
            "The authoritative Round-1 sources do not freeze an unambiguous 21-name contract, so the names/set difference for the eight-field count gap are blocked rather than invented.",
            "No aligned external effective-rank field is present for any benchmark.",
        ],
        "benchmark_blockers": {
            "NAS-Bench-101": [
                "only108 histories contain halfway/final metrics, not epochs 1-5 train accuracy"
            ],
            "NAS-Bench-301": [
                "histories contain train_losses and val_accuracies but no train_accuracy; loss is not substituted"
            ],
            "TransNAS-Bench-101": [
                "native first-five training metrics are available, but train_neg_loss/train_ssim are not exact training-accuracy T",
                "native final metrics are available, but valid_neg_loss/valid_ssim are not exact final accuracy",
                "the eight-field proxy count gap, unaudited required-name set, and erank remain blocked"
            ],
        },
    }
    (args.output / "blockers.json").write_text(
        json.dumps(blockers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, sort_keys=True))

def command_validate(args: argparse.Namespace) -> None:
    expected = {
        "nb101_cifar10.jsonl.gz": 9781,
        "nb301_cifar10.jsonl.gz": 11221,
        **{f"transnas_micro_{task}.jsonl.gz": 4096 for task in TRANSNAS_METRICS},
    }
    errors = []
    observed = {}
    for name, n_expected in expected.items():
        path = args.output / "join_records" / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        count = 0
        counts: Counter = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                count += 1
                for field in (
                    "curve_joined",
                    "q_13_complete",
                    "capacity_finite",
                    "erank_finite",
                    "task_native_early_metric_finite",
                    "exact_paper_t_finite",
                    "task_native_final_metric_exact_match",
                    "exact_final_accuracy_match",
                    "eq1_eligible",
                ):
                    counts[field] += int(bool(row[field]))
                if row["q_required_count_gap"] != 8:
                    errors.append(f"unexpected Q count gap in {name}")
                if row["q_required_names_auditable"]:
                    errors.append(f"unfrozen Q21 names claimed auditable in {name}")
                if row["eq1_eligible"]:
                    errors.append(f"unexpected eligible row in {name}")
        observed[name] = count
        if count != n_expected:
            errors.append(f"{name}: {count} != {n_expected}")
        is_transnas = name.startswith("transnas_")
        exact_accuracy_task = any(
            token in name
            for token in ("class_scene", "class_object", "jigsaw", "segmentsemantic")
        )
        expected_native_early = n_expected if is_transnas else 0
        expected_exact_t = n_expected if exact_accuracy_task else 0
        expected_exact_final = n_expected if (not is_transnas or exact_accuracy_task) else 0
        expected_counts = {
            "curve_joined": n_expected,
            "q_13_complete": n_expected,
            "capacity_finite": n_expected,
            "erank_finite": 0,
            "task_native_early_metric_finite": expected_native_early,
            "exact_paper_t_finite": expected_exact_t,
            "task_native_final_metric_exact_match": n_expected,
            "exact_final_accuracy_match": expected_exact_final,
            "eq1_eligible": 0,
        }
        for field, expected_value in expected_counts.items():
            if counts[field] != expected_value:
                errors.append(f"{name}/{field}: {counts[field]} != {expected_value}")
    if errors:
        raise RuntimeError("E5 validation failed:\n" + "\n".join(errors[:100]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "eq1_completed": False,
        "eq1_eligible_population_count": 0,
        "join_record_counts": observed,
    }
    (args.output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))

def command_decompose(args: argparse.Namespace) -> None:
    validation = json.loads((args.output / "validation.json").read_text(encoding="utf-8"))
    if validation["eq1_eligible_population_count"] == 0:
        print("BLOCKED: no architecture population has an audited named-Q21 set + erank + exact training-accuracy T + exact final accuracy")
        return
    raise NotImplementedError("eligible population exists; decomposition implementation required")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "audit", "validate", "decompose"))
    parser.add_argument("--zc-dir", type=Path)
    parser.add_argument("--nb101", type=Path)
    parser.add_argument("--nb301", type=Path)
    parser.add_argument("--transnas", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eligible-only", action="store_true")
    args = parser.parse_args()
    if args.command in {"prepare", "audit"}:
        for name in ("zc_dir", "nb101", "nb301", "transnas"):
            if getattr(args, name) is None:
                parser.error(f"{args.command} requires --{name.replace('_', '-')}")
    return args

def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "audit":
        command_audit(args)
    elif args.command == "validate":
        command_validate(args)
    elif args.command == "decompose":
        command_decompose(args)

if __name__ == "__main__":
    main()
