"""
MCP server: STEP AP242 PMI and inspection characteristics for AI agents.

Lets any MCP client (Claude Desktop/Code, local agents via Ollama bridges,
IDE assistants) query the semantic PMI of a STEP file: tolerances, datums,
characteristic lists and which faces they apply to. Runs locally over stdio;
STEP files are read from disk and never uploaded by this server.

    pip install 'cadtokens[mcp]'
    cadtokens mcp                     # stdio transport

Claude Code:
    claude mcp add cadtokens -- cadtokens mcp
"""

import functools
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import cadtokens
from cadtokens.characteristics import build_characteristics, to_csv, to_json
from cadtokens.step_engine.pmi import PMIData

INSTRUCTIONS = """\
cadtokens reads semantic PMI (GD&T) from STEP AP242 files fully locally.
Start with pmi_summary to see units, counts and data-quality warnings, then
list_characteristics for the numbered inspection characteristic list.
Face and edge IDs are STEP entity IDs (#123 in the file). Always state the
unit of every value, and surface warnings to the user: they mark places where
the file itself is inconsistent or incomplete. Never invent limits for ISO
fits (e.g. H7) — they are flagged as not computed."""

STEP_EXTENSIONS = ('.stp', '.step', '.p21', '.stpz')

_cache: Dict[str, Tuple[Tuple[float, int], PMIData]] = {}


# ======================================================================
# Loading
# ======================================================================

def _load_pmi(step_path: str) -> PMIData:
    """Parse + extract PMI, cached per file path and modification time."""
    path = os.path.realpath(os.path.expanduser(step_path))
    if not path.lower().endswith(STEP_EXTENSIONS):
        raise ValueError(f"Not a STEP file (expected {', '.join(STEP_EXTENSIONS)}): {step_path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    stat = os.stat(path)
    key = (stat.st_mtime, stat.st_size)
    cached = _cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    pmi = cadtokens.extract_pmi(cadtokens.load(path))
    _cache[path] = (key, pmi)
    return pmi


def _compact(char) -> Dict[str, Any]:
    return {
        'number': char.number,
        'characteristic': char.characteristic,
        'specification': char.specification,
        'min': char.lower_limit,
        'max': char.upper_limit,
        'unit': char.unit,
        'applies_to': char.applies_to,
        'faces': char.face_ids,
        **({'notes': char.notes} if char.notes else {}),
    }


# ======================================================================
# Tools (plain functions, registered in build_server)
# ======================================================================

def pmi_summary(step_path: str) -> Dict[str, Any]:
    """Overview of the semantic PMI in a STEP AP242 file.

    Returns counts, the dominant length unit, datum reference frames used,
    and data-quality warnings (mixed units, unlinked or valueless PMI).
    """
    pmi = _load_pmi(step_path)
    frames = sorted({' | '.join(r.label for r in t.datum_references)
                     for t in pmi.tolerances if t.datum_references})
    return {
        'file': os.path.basename(step_path),
        'has_semantic_pmi': pmi.has_pmi,
        'length_unit': pmi.length_unit,
        'geometric_tolerances': len(pmi.tolerances),
        'dimensions': len(pmi.dimensions),
        'datums': sorted({d.label for d in pmi.datums if d.label}),
        'datum_reference_frames': frames,
        'annotated_faces': len(pmi.annotated_faces()),
        'warnings': pmi.warnings,
    }


def list_characteristics(step_path: str, toleranced_only: bool = False,
                         kind: Optional[str] = None,
                         max_tolerance: Optional[float] = None) -> Dict[str, Any]:
    """Numbered inspection characteristic list (basis for FAI / EMPB reports).

    Args:
        step_path: path to the STEP AP242 file
        toleranced_only: skip dimensions without tolerance (basic / general)
        kind: 'dimension' or 'geometric_tolerance' to filter
        max_tolerance: only characteristics whose tolerance zone/band is at
            most this value (in the file's unit) — finds the tight ones
    """
    pmi = _load_pmi(step_path)
    chars = build_characteristics(pmi, include_untoleranced=not toleranced_only)
    if kind:
        chars = [c for c in chars if c.kind == kind]
    if max_tolerance is not None:
        chars = [c for c in chars
                 if c.lower_limit is not None and c.upper_limit is not None
                 and (c.upper_limit - c.lower_limit) <= max_tolerance]
    return {
        'length_unit': pmi.length_unit,
        'count': len(chars),
        'characteristics': [_compact(c) for c in chars],
        'warnings': pmi.warnings,
    }


def get_characteristic(step_path: str, number: int) -> Dict[str, Any]:
    """Full detail of one characteristic by its balloon number."""
    for c in build_characteristics(_load_pmi(step_path)):
        if c.number == number:
            return asdict(c)
    raise ValueError(f"No characteristic number {number}")


def face_pmi(step_path: str, face_id: int) -> Dict[str, Any]:
    """All PMI that applies to one B-Rep face (STEP entity ID of an ADVANCED_FACE)."""
    pmi = _load_pmi(step_path)
    ann = pmi.face_annotations(face_id)
    chars = [c for c in build_characteristics(pmi)
             if face_id in c.face_ids or c.applies_to == 'whole part']
    return {
        'face_id': face_id,
        'characteristics': [_compact(c) for c in chars],
        'datum_features': [d.label for d in ann['datums']],
    }


def datum_reference_frames(step_path: str) -> Dict[str, Any]:
    """Datums (with the faces that define them) and every datum reference frame in use."""
    pmi = _load_pmi(step_path)
    frames: Dict[str, List[str]] = {}
    for t in pmi.tolerances:
        if t.datum_references:
            key = ' | '.join(
                r.label + (' ' + ' '.join(r.modifiers) if r.modifiers else '')
                for r in t.datum_references)
            frames.setdefault(key, []).append(t.name or f"#{t.entity_id}")
    return {
        'datums': [{'label': d.label, 'faces': d.face_ids, 'edges': d.edge_ids}
                   for d in pmi.datums],
        'frames': [{'frame': k, 'used_by': v} for k, v in sorted(frames.items())],
    }


def export_characteristics(step_path: str, output_path: str, format: str = 'csv',
                           delimiter: str = ',', decimal_comma: bool = False,
                           overwrite: bool = False) -> Dict[str, Any]:
    """Write the characteristic list to CSV (Excel), JSON, or a ballooned PDF report.

    Use delimiter=';' and decimal_comma=True for German Excel.
    format='pdf' writes an inspection report with a ballooned part view.
    Refuses to overwrite an existing file unless overwrite=True.
    """
    if format not in ('csv', 'json', 'pdf'):
        raise ValueError("format must be 'csv', 'json' or 'pdf'")
    out = os.path.realpath(os.path.expanduser(output_path))
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f"{output_path} exists — pass overwrite=True to replace it")
    pmi = _load_pmi(step_path)
    chars = build_characteristics(pmi)
    if format == 'pdf':
        from cadtokens.report import write_report
        write_report(os.path.realpath(os.path.expanduser(step_path)), out, pmi=pmi, chars=chars)
    elif format == 'csv':
        to_csv(chars, out, delimiter=delimiter, decimal_comma=decimal_comma)
    else:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(to_json(chars, pmi, source=step_path))
    return {'written': out, 'count': len(chars), 'warnings': pmi.warnings}


READ_ONLY_TOOLS = [pmi_summary, list_characteristics, get_characteristic,
                   face_pmi, datum_reference_frames]


# ======================================================================
# Server
# ======================================================================

def build_server():
    """Create the MCP server (supports mcp SDK 2.x and 1.x)."""
    try:
        from mcp.server.mcpserver import MCPServer as Server   # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server       # mcp 1.x
    try:
        from mcp.types import ToolAnnotations
    except ImportError:
        ToolAnnotations = None

    server = Server(name="cadtokens", instructions=INSTRUCTIONS)

    try:
        from mcp.server.mcpserver.exceptions import ToolError            # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp.exceptions import ToolError              # mcp 1.x

    def user_errors(fn):
        """Report expected failures (bad path, unknown number) as clean tool errors."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (ValueError, FileNotFoundError, FileExistsError) as e:
                raise ToolError(str(e)) from e
        return wrapper

    def annotations(read_only: bool):
        if ToolAnnotations is None:
            return {}
        return {'annotations': ToolAnnotations(readOnlyHint=read_only,
                                               destructiveHint=False,
                                               openWorldHint=False)}

    for fn in READ_ONLY_TOOLS:
        server.tool(**annotations(True))(user_errors(fn))
    server.tool(**annotations(False))(user_errors(export_characteristics))
    return server


def main():
    build_server().run("stdio")


if __name__ == "__main__":
    main()
