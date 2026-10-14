"""
cadtokens.tools — OCC-based geometry analysis tools for LLM integration.

Provides structured tool definitions that an LLM can call via function calling.

Requires: pythonocc-core
"""
from cadtokens.tools.geometry_tools import GeometryTools, TOOL_DEFINITIONS
from cadtokens.tools.step_analyzer import analyze_step, AnalysisResult

__all__ = ["GeometryTools", "TOOL_DEFINITIONS", "analyze_step", "AnalysisResult"]
