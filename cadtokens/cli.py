#!/usr/bin/env python3
"""
cadtokens CLI — Talk to your STEP files.

Usage:
    cadtokens parse   part.stp              # Parse → JSON
    cadtokens pmi     part.stp              # Extract PMI/GD&T
    cadtokens chat    part.stp              # Interactive chat (CLI)
    cadtokens ui      part.stp              # Web UI (Gradio)
    cadtokens info    part.stp              # Quick geometry summary
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_parse(args):
    """Parse a STEP file and output structured data."""
    from cadtokens.step_engine import parse_file
    from cadtokens.step_engine.graph import ReferenceGraph, BRepExtractor

    sf = parse_file(args.step_file)
    g = ReferenceGraph(sf)
    brep = BRepExtractor(g)

    result = {
        "file": args.step_file,
        "entities": len(sf.entities),
        "faces": len(brep.get_faces()),
        "edges": len(brep.get_edge_curves()),
        "points": len(brep.get_points()),
        "topology": {
            etype: count
            for etype, count in sorted(
                g.entity_count_by_type().items(), key=lambda x: -x[1]
            )[:20]
        },
    }

    if args.full:
        faces = brep.get_faces()
        result["face_details"] = []
        for i, fid in enumerate(faces):
            surface = brep.get_face_surface(fid)
            stype = surface.type_name if surface else "unknown"
            result["face_details"].append({
                "id": i,
                "entity_id": fid,
                "surface_type": stype,
            })

    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_pmi(args):
    """Extract PMI (GD&T) from a STEP file."""
    from cadtokens.step_engine import parse_file, extract_pmi

    sf = parse_file(args.step_file)
    pmi = extract_pmi(sf)

    result = {
        "file": args.step_file,
        "tolerances": [
            {
                "type": t.tolerance_type,
                "value": t.value,
                "faces": t.face_ids,
                "datum_labels": t.datum_labels,
            }
            for t in pmi.tolerances
        ],
        "dimensions": [
            {
                "type": d.dimension_type,
                "nominal_value": d.nominal_value,
                "faces": d.face_ids,
            }
            for d in pmi.dimensions
        ],
        "datums": [
            {"label": d.label, "faces": d.face_ids}
            for d in pmi.datums
        ],
        "summary": {
            "num_tolerances": len(pmi.tolerances),
            "num_dimensions": len(pmi.dimensions),
            "num_datums": len(pmi.datums),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_chat(args):
    """Interactive CLI chat with a STEP file."""
    try:
        from cadtokens.chat.cad_chat import CADChatAgent
    except ImportError:
        print("Chat requires: pip install ollama pythonocc-core")
        sys.exit(1)

    agent = CADChatAgent(model_name=args.model)
    agent.run(args.step_file)


def cmd_ui(args):
    """Launch Gradio web UI."""
    try:
        from cadtokens.chat.chat_ui import build_ui
    except ImportError as e:
        print(f"Web UI requires: pip install ollama pythonocc-core gradio\n{e}")
        sys.exit(1)

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port)


def cmd_info(args):
    """Quick geometry summary using OCC."""
    try:
        from cadtokens.tools.geometry_tools import GeometryTools
    except ImportError:
        print("Info requires: pip install pythonocc-core")
        sys.exit(1)

    tools = GeometryTools()
    summary = tools.get_summary(args.step_file)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def main():
    parser = argparse.ArgumentParser(
        prog="cadtokens",
        description="Talk to your STEP files — parse, extract PMI, chat.",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # parse
    p = sub.add_parser("parse", help="Parse STEP → JSON")
    p.add_argument("step_file", help="Path to .stp/.step file")
    p.add_argument("--full", action="store_true", help="Include face details")
    p.set_defaults(func=cmd_parse)

    # pmi
    p = sub.add_parser("pmi", help="Extract PMI/GD&T")
    p.add_argument("step_file", help="Path to .stp/.step file")
    p.set_defaults(func=cmd_pmi)

    # chat
    p = sub.add_parser("chat", help="Interactive CLI chat")
    p.add_argument("step_file", help="Path to .stp/.step file")
    p.add_argument("--model", default="qwen3:14b", help="Ollama model name")
    p.set_defaults(func=cmd_chat)

    # ui
    p = sub.add_parser("ui", help="Launch Gradio web UI")
    p.add_argument("--port", type=int, default=7860, help="Server port")
    p.set_defaults(func=cmd_ui)

    # info
    p = sub.add_parser("info", help="Quick geometry summary (OCC)")
    p.add_argument("step_file", help="Path to .stp/.step file")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
