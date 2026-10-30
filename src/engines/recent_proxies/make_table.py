# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, label in (("swap", "SWAP"), ("aznas", "AZ-NAS")):
        item = {"method": label}
        for space in ("tss", "sss"):
            metric = results["methods"][method][space]["test_accuracy"]
            item[f"{space}_n"] = results["methods"][method][space]["n"]
            item[f"{space}_rho"] = metric["spearman_rho"]
            item[f"{space}_ci_low"] = metric["ci_percentile"][0]
            item[f"{space}_ci_high"] = metric["ci_percentile"][1]
        item["same_sign"] = results["methods"][method]["test_accuracy_sign_stability"]["same_nonzero_point_sign"]
        rows.append(item)
    fields = list(rows[0])
    with (args.output_dir / "paper_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Proxy & TSS $\\rho$ [95\\% CI] & SSS $\\rho$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    for row in rows:
        tss = f"{row['tss_rho']:.3f} [{row['tss_ci_low']:.3f}, {row['tss_ci_high']:.3f}]"
        sss = f"{row['sss_rho']:.3f} [{row['sss_ci_low']:.3f}, {row['sss_ci_high']:.3f}]"
        lines.append(f"{row['method']} & {tss} & {sss} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (args.output_dir / "paper_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows)}, indent=2))

if __name__ == "__main__":
    main()
