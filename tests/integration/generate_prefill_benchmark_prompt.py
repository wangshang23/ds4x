#!/usr/bin/env python3
"""Generate the deterministic 6,000,000-byte DS4X prefill fixture."""

import argparse
import hashlib
from pathlib import Path


TEXT = (
    "DS4X reproducible prefill benchmark. The quick brown fox jumps over the "
    "lazy dog. Compressed sparse attention and mixture of experts are measured "
    "on NVIDIA GB10.\n"
).encode("ascii")
SIZE = 6_000_000
SHA256 = "ee9ef79daa0f0adbe3972bebe6a404c19a65c829f19e1c34d84fb6f12c4ff645"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = (TEXT * ((SIZE + len(TEXT) - 1) // len(TEXT)))[:SIZE]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"fixture hash mismatch: {digest}")
    args.output.write_bytes(payload)
    print(f"{args.output}: {len(payload)} bytes sha256={digest}")


if __name__ == "__main__":
    main()
