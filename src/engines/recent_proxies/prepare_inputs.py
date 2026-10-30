# Numerical engine; see docs/protocols.md for interpretation and provenance.
from __future__ import annotations

import argparse
import hashlib
import io
import pickle
import tarfile
from pathlib import Path

import numpy as np

ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
DATA_BATCH_1_MD5 = "c99cafc152244af753f735de768cd75f"
OUTPUT_SHA256 = "3ff731b7248a0e538ce114bcb6868bf1be292468801c48d723f8266b4e26fedc"
MEMBER_SUFFIX = "cifar-10-batches-py/data_batch_1"

def digest(data: bytes, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    value.update(data)
    return value.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archive_bytes = args.archive.read_bytes()
    archive_md5 = digest(archive_bytes, "md5")
    if archive_md5 != ARCHIVE_MD5:
        raise RuntimeError(f"CIFAR-10 archive MD5 mismatch: {archive_md5}")

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        matches = [member for member in archive.getmembers() if member.name.endswith(MEMBER_SUFFIX)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {MEMBER_SUFFIX} member, found {len(matches)}")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise RuntimeError("could not read CIFAR-10 data_batch_1")
        batch_bytes = handle.read()

    batch_md5 = digest(batch_bytes, "md5")
    if batch_md5 != DATA_BATCH_1_MD5:
        raise RuntimeError(f"CIFAR-10 data_batch_1 MD5 mismatch: {batch_md5}")

    payload = pickle.loads(batch_bytes, encoding="bytes")
    raw = np.asarray(payload[b"data"], dtype=np.uint8)
    if raw.shape != (10000, 3072):
        raise RuntimeError(f"unexpected CIFAR-10 data shape: {raw.shape}")
    first16 = np.ascontiguousarray(raw.reshape(10000, 3, 32, 32)[:16])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.save(output, first16, allow_pickle=False)
    output_sha256 = digest(args.output.read_bytes(), "sha256")
    if output_sha256 != OUTPUT_SHA256:
        raise RuntimeError(f"first16 NPY SHA-256 mismatch: {output_sha256}")

    print(
        f"PASS archive_md5={archive_md5} data_batch_1_md5={batch_md5} "
        f"output_sha256={output_sha256} shape={first16.shape} dtype={first16.dtype}"
    )

if __name__ == "__main__":
    main()
