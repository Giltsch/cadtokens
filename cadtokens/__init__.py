"""
cadtokens — read and check the GD&T in STEP AP242 files.

Pure Python, zero dependencies, fully local.

Quick start:
    import cadtokens
    from cadtokens.characteristics import build_characteristics
    from cadtokens.report import write_report

    pmi = cadtokens.extract_pmi(cadtokens.load("part.stp"))
    print(pmi.warnings)                         # data-quality issues in the file
    chars = build_characteristics(pmi)          # inspection characteristic list
    write_report("part.stp", "part_inspection.pdf")
"""

__version__ = "0.2.0"

# ── Feature 1: STEP Parser + PMI (always available, zero deps) ──────────────
from cadtokens.step_engine import (
    # Core
    load, save, edit, graph,
    # Data model
    StepFile, Entity, Ref, EnumValue, TypedValue,
    # Parser
    parse, parse_file, ParseError,
    # Serializer
    serialize,
    # Graph
    ReferenceGraph, BRepExtractor,
    # Editor
    StepEditor,
    # PMI
    PMIExtractor, PMIWriter, PMIData,
    ToleranceInfo, DimensionInfo, DatumInfo, DatumSystemInfo,
    extract_pmi, tolerance_category,
)


__all__ = [
    # Version
    "__version__",
    # STEP Parser
    "load", "save", "edit", "graph",
    "StepFile", "Entity", "Ref", "EnumValue", "TypedValue",
    "parse", "parse_file", "ParseError", "serialize",
    "ReferenceGraph", "BRepExtractor", "StepEditor",
    # PMI
    "PMIExtractor", "PMIWriter", "PMIData",
    "ToleranceInfo", "DimensionInfo", "DatumInfo", "DatumSystemInfo",
    "extract_pmi", "tolerance_category",
]
