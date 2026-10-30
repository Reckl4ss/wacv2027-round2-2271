# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

EXPECTED = {
    "nb101_cifar10.jsonl.gz": 9781,
    "nb301_cifar10.jsonl.gz": 11221,
    "transnas_micro_class_scene.jsonl.gz": 4096,
    "transnas_micro_class_object.jsonl.gz": 4096,
    "transnas_micro_jigsaw.jsonl.gz": 4096,
    "transnas_micro_room_layout.jsonl.gz": 4096,
    "transnas_micro_segmentsemantic.jsonl.gz": 4096,
    "transnas_micro_normal.jsonl.gz": 4096,
    "transnas_micro_autoencoder.jsonl.gz": 4096,
}
EXACT_ACCURACY_TASK_FILES = {
    "transnas_micro_class_scene.jsonl.gz",
    "transnas_micro_class_object.jsonl.gz",
    "transnas_micro_jigsaw.jsonl.gz",
    "transnas_micro_segmentsemantic.jsonl.gz",
}
AVAILABLE_PROXY_FIELDS = {
    "epe_nas", "fisher", "flops", "grad_norm", "grasp", "jacov", "l2_norm",
    "nwot", "params", "plain", "snip", "synflow", "zen",
}

def verify(output: Path) -> dict:
    observed: dict[str, dict] = {}
    total = 0
    for name, expected_count in EXPECTED.items():
        ids = set()
        counts = Counter()
        missing = Counter()
        path = output / "join_records" / name
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                arch_id = row["canonical_architecture_id"]
                if arch_id in ids:
                    raise AssertionError(f"duplicate canonical ID in {name}: {arch_id}")
                ids.add(arch_id)
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
                    counts[key] += int(bool(row[key]))
                counts["capacity_scale_exact_match"] += int(
                    bool(row.get("capacity_scale_exact_match", False))
                )
                missing.update(row["missing_fields"])
                if row["q_finite_count"] != 13 or row["q_required_21_complete"]:
                    raise AssertionError(f"unexpected proxy coverage in {name}")
                if set(row["q_available_fields"]) != AVAILABLE_PROXY_FIELDS:
                    raise AssertionError(f"unexpected available proxy fields in {name}")
                if row["q_required_count"] != 21 or row["q_required_count_gap"] != 8:
                    raise AssertionError(f"unexpected Q21 count contract in {name}")
                if row["q_required_names_auditable"] or row["q_required_fields"] is not None:
                    raise AssertionError(f"unfrozen Q21 name set claimed auditable in {name}")
                if row["q_required_missing_fields"] is not None:
                    raise AssertionError(f"invented Q21 set difference in {name}")
                if row["erank_value"] is not None or row["erank_finite"]:
                    raise AssertionError(f"unexpected erank in {name}")
                if not row["task_native_final_metric_exact_match"]:
                    raise AssertionError(f"final metric mismatch in {name}: {arch_id}")
                if row["eq1_eligible"]:
                    raise AssertionError(f"unexpected Eq. (1) eligible row in {name}")
        if counts["rows"] != expected_count or len(ids) != expected_count:
            raise AssertionError(f"row/ID count mismatch in {name}: {counts['rows']}")
        is_transnas = name.startswith("transnas_")
        exact_accuracy_task = name in EXACT_ACCURACY_TASK_FILES
        if counts["task_native_early_metric_finite"] != (expected_count if is_transnas else 0):
            raise AssertionError(f"unexpected task-native early-metric count in {name}")
        if counts["exact_paper_t_finite"] != (expected_count if exact_accuracy_task else 0):
            raise AssertionError(f"unexpected exact paper-T count in {name}")
        if counts["task_native_final_metric_exact_match"] != expected_count:
            raise AssertionError(f"unexpected task-native final-metric match count in {name}")
        expected_exact_accuracy = expected_count if (not is_transnas or exact_accuracy_task) else 0
        if counts["exact_final_accuracy_match"] != expected_exact_accuracy:
            raise AssertionError(f"unexpected exact final-accuracy count in {name}")
        if name == "nb101_cifar10.jsonl.gz" and counts["capacity_scale_exact_match"] != expected_count:
            raise AssertionError("NB101 capacity integer-scale equality did not hold for all rows")
        observed[name] = {"counts": dict(counts), "missing_fields": dict(missing)}
        total += expected_count

    with (output / "field_coverage.csv").open(newline="", encoding="utf-8") as handle:
        coverage = list(csv.DictReader(handle))
    if len(coverage) != len(EXPECTED):
        raise AssertionError("field_coverage.csv row count mismatch")
    for row in coverage:
        if int(row["q_required_count"]) != 21 or int(row["q_required_count_gap"]) != 8:
            raise AssertionError("field_coverage.csv Q count contract mismatch")
        if int(row["q_required_names_auditable"]) != 0:
            raise AssertionError("field_coverage.csv invents an auditable Q21 name set")
        if row["q_required_fields_status"] != "blocked_not_frozen_in_authoritative_round1_source":
            raise AssertionError("field_coverage.csv required-name blocker mismatch")
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    if validation != {
        "eq1_completed": False,
        "eq1_eligible_population_count": 0,
        "join_record_counts": {name: EXPECTED[name] for name in sorted(EXPECTED)},
        "schema_version": "1.1.0",
        "status": "PASS",
    }:
        raise AssertionError("validation.json does not match independently expected contract")
    return {
        "status": "PASS",
        "total_join_records": total,
        "files_verified": len(EXPECTED),
        "observed": observed,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.output)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")

if __name__ == "__main__":
    main()
