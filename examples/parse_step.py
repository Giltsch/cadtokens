#!/usr/bin/env python3
"""
Example: Parse a STEP file and explore its structure.

This uses only the pure Python parser — no external dependencies needed.

Usage:
    python examples/parse_step.py path/to/part.stp
"""

import sys
import json
from pathlib import Path

# Add parent dir so this works without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cadtokens


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_step.py <file.stp>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Parsing: {path}\n")

    # ── Load ────────────────────────────────────────────────────────────
    sf = cadtokens.load(path)
    print(f"Entities: {len(sf.entities)}")
    print(f"Header:   {sf.header}")

    # ── Reference graph ─────────────────────────────────────────────────
    g = cadtokens.graph(sf)
    counts = g.entity_count_by_type()

    print(f"\nTop 10 entity types:")
    for etype, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {etype}: {count}")

    # ── B-Rep extraction ────────────────────────────────────────────────
    brep = cadtokens.BRepExtractor(g)
    faces = brep.get_faces()
    edges = brep.get_edge_curves()
    points = brep.get_points()

    print(f"\nB-Rep topology:")
    print(f"  Faces:  {len(faces)}")
    print(f"  Edges:  {len(edges)}")
    print(f"  Points: {len(points)}")

    # ── Face details ────────────────────────────────────────────────────
    if faces:
        print(f"\nFirst 5 faces:")
        for fid in faces[:5]:
            surface = brep.get_face_surface(fid)
            stype = surface.type_name if surface else "unknown"
            print(f"  #{fid}: {stype}")


if __name__ == "__main__":
    main()
