# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    root = args.payload_root.resolve()
    rows: list[tuple[str, str]] = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        rows.append((expected.lower(), rel.strip()))

    failures: list[str] = []
    for expected, rel in rows:
        target = root / rel
        if not target.is_file():
            failures.append(f"MISSING\t{rel}")
            continue
        actual = sha256(target)
        if actual != expected:
            failures.append(f"HASH\t{rel}\t{expected}\t{actual}")

    print(f"manifest_entries={len(rows)}")
    print(f"verified={len(rows) - len(failures)}")
    print(f"failures={len(failures)}")
    for failure in failures:
        print(failure)
    return 1 if failures or len(rows) != 179 else 0


if __name__ == "__main__":
    raise SystemExit(main())
