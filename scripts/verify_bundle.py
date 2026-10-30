"""Verify the exported measurement/result bytes before executing analyses."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def verify():
    manifest = json.loads((ROOT / 'data/provenance/export_manifest.json').read_text())
    failures = []
    checked = 0
    for record in manifest:
        # Adapted source is tracked separately by the final distribution checksum.
        if record['transformed']:
            continue
        path = ROOT / record['destination']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            failures.append(record['destination'])
        checked += 1
    result = {'status': 'PASS' if not failures else 'FAIL', 'byte_identical_files_checked': checked, 'failures': failures}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)
    return result

if __name__ == '__main__':
    verify()
