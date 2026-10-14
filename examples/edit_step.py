#!/usr/bin/env python3
"""
Example: Edit a STEP file — translate geometry and re-serialize.

No external dependencies needed — uses pure Python parser.

Usage:
    python examples/edit_step.py input.stp output.stp
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cadtokens


def main():
    if len(sys.argv) < 3:
        print("Usage: python edit_step.py <input.stp> <output.stp>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Loading: {in_path}")
    sf = cadtokens.load(in_path)
    print(f"  Entities: {len(sf.entities)}")

    # ── Edit ────────────────────────────────────────────────────────────
    editor = cadtokens.edit(sf)

    # Translate all geometry by (100, 0, 0)
    editor.translate(100.0, 0.0, 0.0)

    # Renumber entities to be contiguous
    editor.renumber()

    print(f"  Translated by (100, 0, 0)")
    print(f"  Renumbered entities")

    # ── Save ────────────────────────────────────────────────────────────
    cadtokens.save(sf, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
