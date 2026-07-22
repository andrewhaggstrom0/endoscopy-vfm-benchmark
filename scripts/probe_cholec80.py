"""Inspect the Cholec80 distribution before writing a loader against it.

Annotation formats and directory layout differ between the official release
and mirrors. Guessing here is how a reproduction gate fails for a week.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else f"{os.environ['BIGDIR']}/endoscopy/raw/cholec80")
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return 1

    print(f"root  {root}\n")

    print("top level:")
    for p in sorted(root.iterdir())[:20]:
        kind = "dir " if p.is_dir() else "file"
        print(f"  {kind} {p.name}")

    ext = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
    print(f"\nextensions: {dict(ext.most_common(10))}")

    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".avi"))
    print(f"\nvideos: {len(vids)}")
    for v in vids[:3]:
        print(f"  {v.relative_to(root)}  {v.stat().st_size / 1e9:.2f} GB")

    txts = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".txt", ".csv"))
    print(f"\nannotation files: {len(txts)}")
    for t in txts[:6]:
        print(f"  {t.relative_to(root)}")

    for pattern in ("phase", "tool", "timestamp"):
        match = next((t for t in txts if pattern in t.name.lower()), None)
        if match:
            print(f"\n--- {match.name} (first 5 lines) ---")
            with open(match) as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(f"  {line.rstrip()!r}")
            with open(match) as f:
                print(f"  total lines: {sum(1 for _ in f)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
