#!/usr/bin/env python3
"""
Example: Extract PMI (GD&T) data from a STEP file.

PMI = Product Manufacturing Information: tolerances, dimensions,
datums, surface finish. Extracted from AP242 STEP files.

No external dependencies needed — uses pure Python parser.

Usage:
    python examples/extract_pmi.py path/to/part_with_pmi.stp
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cadtokens


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_pmi.py <file.stp>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Extracting PMI from: {path}\n")

    # ── Load + extract ──────────────────────────────────────────────────
    sf = cadtokens.load(path)
    pmi = cadtokens.extract_pmi(sf)

    # ── Tolerances ──────────────────────────────────────────────────────
    print(f"Tolerances ({len(pmi.tolerances)}):")
    for t in pmi.tolerances:
        cat = cadtokens.tolerance_category(t)
        print(f"  [{cat}] {t.tolerance_type}: {t.value}")
        if t.face_ids:
            print(f"    → Faces: {t.face_ids}")
        if t.datum_labels:
            print(f"    → Datums: {t.datum_labels}")

    # ── Dimensions ──────────────────────────────────────────────────────
    print(f"\nDimensions ({len(pmi.dimensions)}):")
    for d in pmi.dimensions:
        print(f"  {d.dimension_type}: {d.nominal_value}")
        if d.face_ids:
            print(f"    → Faces: {d.face_ids}")

    # ── Datums ──────────────────────────────────────────────────────────
    print(f"\nDatums ({len(pmi.datums)}):")
    for d in pmi.datums:
        print(f"  {d.label}")
        if d.face_ids:
            print(f"    → Faces: {d.face_ids}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  {len(pmi.tolerances)} tolerances")
    print(f"  {len(pmi.dimensions)} dimensions")
    print(f"  {len(pmi.datums)} datums")
    print(f"  {len(pmi.datum_systems)} datum systems")

    # ── JSON output ─────────────────────────────────────────────────────
    if "--json" in sys.argv:
        result = {
            "tolerances": [
                {"type": t.tolerance_type, "value": t.value,
                 "faces": t.face_ids}
                for t in pmi.tolerances
            ],
            "dimensions": [
                {"type": d.dimension_type, "value": d.nominal_value,
                 "faces": d.face_ids}
                for d in pmi.dimensions
            ],
            "datums": [
                {"label": d.label, "faces": d.face_ids}
                for d in pmi.datums
            ],
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
