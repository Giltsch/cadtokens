#!/usr/bin/env python3
"""
cadtokens CLI — Talk to your STEP files.

Usage:
    cadtokens parse   part.stp              # Parse → JSON
    cadtokens pmi     part.stp              # Extract PMI/GD&T
    cadtokens characteristics part.stp      # Inspection characteristic list
    cadtokens report  part.stp              # Ballooned inspection report (PDF)
    cadtokens mcp                           # MCP server for AI agents
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
    from dataclasses import asdict
    from cadtokens.step_engine import parse_file, extract_pmi

    sf = parse_file(args.step_file)
    pmi = extract_pmi(sf)

    result = {
        "file": args.step_file,
        "length_unit": pmi.length_unit,
        "warnings": pmi.warnings,
        "tolerances": [
            {
                "type": t.tolerance_type,
                "name": t.name,
                "value": t.value,
                "unit": t.unit,
                "diametral_zone": t.is_diametral,
                "modifiers": t.modifiers,
                "datums": [asdict(r) for r in t.datum_references],
                "projected_length": t.projected_length,
                "unit_basis": t.unit_basis,
                "feature_count": t.feature_count,
                "all_around": t.all_around,
                "whole_part": t.whole_part,
                "faces": t.face_ids,
                "edges": t.edge_ids,
                "entity_id": t.entity_id,
            }
            for t in pmi.tolerances
        ],
        "dimensions": [
            {
                "type": d.dimension_type,
                "semantic": d.semantic_name,
                "nominal_value": d.nominal_value,
                "upper_tolerance": d.upper_tolerance,
                "lower_tolerance": d.lower_tolerance,
                "upper_limit": d.upper_limit,
                "lower_limit": d.lower_limit,
                "fit": d.fit,
                "qualifier": d.qualifier,
                "unit": d.unit,
                "feature_count": d.feature_count,
                "faces": d.face_ids,
                "entity_id": d.entity_id,
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


def cmd_characteristics(args):
    """Inspection characteristic list (FAI / EMPB) from semantic PMI."""
    from cadtokens.step_engine import parse_file, extract_pmi
    from cadtokens.characteristics import build_characteristics, to_csv, to_json, to_table

    pmi = extract_pmi(parse_file(args.step_file))
    chars = build_characteristics(pmi, include_untoleranced=not args.toleranced_only)

    if args.format == "pdf":
        if not args.output:
            print("PDF needs an output file: -o report.pdf", file=sys.stderr)
            sys.exit(2)
        from cadtokens.report import write_report
        info = write_report(args.step_file, args.output, pmi=pmi, chars=chars)
        text = ""
        print(f"{info['characteristics']} characteristics ({info['ballooned']} ballooned), "
              f"{info['pages']} pages → {args.output}")
    elif args.format == "csv":
        text = to_csv(chars, args.output, delimiter=args.delimiter,
                      decimal_comma=args.decimal_comma)
    elif args.format == "json":
        text = to_json(chars, pmi, source=args.step_file)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
    else:
        text = to_table(chars)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")

    if args.format == "pdf":
        pass
    elif args.output:
        print(f"{len(chars)} characteristics written to {args.output}")
    else:
        print(text)
    if pmi.warnings:
        print(f"\n{len(pmi.warnings)} data-quality warning(s):", file=sys.stderr)
        for w in pmi.warnings:
            print(f"  ! {w}", file=sys.stderr)
    if not chars:
        print("No semantic PMI found in this file.", file=sys.stderr)


def cmd_chat(args):
    """Interactive CLI chat with a STEP file."""
    try:
        from cadtokens.chat.cad_chat import CADChatAgent
    except ImportError:
        print("Chat requires: pip install ollama pythonocc-core")
        sys.exit(1)

    from cadtokens.chat.cad_chat import interactive_mode
    agent = CADChatAgent(model=args.model)
    agent.set_step_file(args.step_file)
    interactive_mode(agent)


def cmd_report(args):
    """Ballooned inspection report PDF."""
    from cadtokens.report import write_report

    out = args.output or str(Path(args.step_file).with_suffix("")) + "_inspection.pdf"
    info = write_report(args.step_file, out)
    print(f"{info['characteristics']} characteristics ({info['ballooned']} ballooned), "
          f"{info['pages']} pages → {out}")
    for w in info["warnings"]:
        print(f"  ! {w}", file=sys.stderr)


def cmd_mcp(args):
    """Run the MCP server over stdio."""
    try:
        from cadtokens.mcp_server import main as mcp_main
    except ImportError as e:
        print(f"MCP server requires: pip install 'cadtokens[mcp]'\n{e}", file=sys.stderr)
        sys.exit(1)
    mcp_main()


def cmd_ui(args):
    """Launch Gradio web UI."""
    try:
        from cadtokens.chat.chat_ui import build_ui
    except ImportError as e:
        print(f"Web UI requires: pip install ollama pythonocc-core gradio\n{e}")
        sys.exit(1)

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)


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
        description="STEP AP242 PMI extraction and inspection characteristic lists.",
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

    # report
    p = sub.add_parser("report", help="Ballooned inspection report (PDF)")
    p.add_argument("step_file", help="Path to .stp/.step file (AP242 with semantic PMI)")
    p.add_argument("-o", "--output", help="PDF file (default: <step name>_inspection.pdf)")
    p.set_defaults(func=cmd_report)

    # characteristics
    p = sub.add_parser("characteristics", aliases=["chars"],
                       help="Inspection characteristic list (FAI / EMPB)")
    p.add_argument("step_file", help="Path to .stp/.step file (AP242 with semantic PMI)")
    p.add_argument("-f", "--format", choices=["table", "csv", "json", "pdf"], default="table",
                   help="pdf = ballooned inspection report (needs -o)")
    p.add_argument("-o", "--output", help="Write to file instead of stdout")
    p.add_argument("--delimiter", default=",", help="CSV delimiter (use ';' for German Excel)")
    p.add_argument("--decimal-comma", action="store_true", help="CSV numbers as 0,15")
    p.add_argument("--toleranced-only", action="store_true",
                   help="Skip dimensions without tolerance (basic / general tolerance)")
    p.set_defaults(func=cmd_characteristics)

    # mcp
    p = sub.add_parser("mcp", help="Run the MCP server (stdio) for AI agents")
    p.set_defaults(func=cmd_mcp)

    # chat
    p = sub.add_parser("chat", help="Interactive CLI chat (experimental)")
    p.add_argument("step_file", help="Path to .stp/.step file")
    p.add_argument("--model", default="qwen3:14b", help="Local Ollama model name")
    p.set_defaults(func=cmd_chat)

    # ui
    p = sub.add_parser("ui", help="Launch Gradio web UI")
    p.add_argument("--port", type=int, default=7860, help="Server port")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (default: localhost only; 0.0.0.0 exposes it on the network)")
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
