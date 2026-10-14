"""
STEP Engine - Pure Python STEP file parser, editor, and serializer.

No CAD kernel required. Handles ISO 10303-21 (Part 21) format.
Parse, query, edit, transform, and serialize STEP files using only
the graph structure — no geometric computation needed.

Usage:
    import step_engine

    # Load
    sf = step_engine.load("part.step")

    # Query
    g = step_engine.graph(sf)
    faces = g.find_by_type('ADVANCED_FACE')
    points = step_engine.BRepExtractor(g).get_points()

    # Edit
    e = step_engine.edit(sf)
    e.translate(10, 0, 0)
    e.renumber()

    # Save
    step_engine.save(sf, "modified.step")
"""

from .model import (
    StepFile, Entity, EntityPart, HeaderSection,
    Ref, EnumValue, TypedValue, Unset, Derived, UNSET, DERIVED,
    StepParam,
)
from .parser import parse, parse_file, ParseError
from .serializer import serialize
from .graph import ReferenceGraph, BRepExtractor
from .editor import StepEditor
from .pmi import (
    PMIExtractor, PMIWriter, PMIData,
    ToleranceInfo, DimensionInfo, DatumInfo, DatumSystemInfo,
    extract_pmi, tolerance_category,
)


def load(path: str) -> StepFile:
    """Load and parse a STEP file from disk."""
    return parse_file(path)


def save(step_file: StepFile, path: str, line_width: int = 80):
    """Serialize and save a StepFile to disk."""
    with open(path, 'w') as f:
        f.write(serialize(step_file, line_width=line_width))


def edit(step_file: StepFile) -> StepEditor:
    """Create an editor for a StepFile."""
    return StepEditor(step_file)


def graph(step_file: StepFile) -> ReferenceGraph:
    """Build the reference graph for a StepFile."""
    return ReferenceGraph(step_file)


__all__ = [
    # Data model
    'StepFile', 'Entity', 'EntityPart', 'HeaderSection',
    'Ref', 'EnumValue', 'TypedValue', 'Unset', 'Derived', 'UNSET', 'DERIVED',
    'StepParam',
    # Parser
    'parse', 'parse_file', 'ParseError',
    # Serializer
    'serialize',
    # Graph
    'ReferenceGraph', 'BRepExtractor',
    # Editor
    'StepEditor',
    # PMI
    'PMIExtractor', 'PMIWriter', 'PMIData',
    'ToleranceInfo', 'DimensionInfo', 'DatumInfo', 'DatumSystemInfo',
    'extract_pmi', 'tolerance_category',
    # Convenience
    'load', 'save', 'edit', 'graph',
]
