"""
cadtokens — Talk to your STEP files.

Parse STEP geometry + PMI, chat with CAD models, detect manufacturing features.

Quick start:
    import cadtokens

    # Parse a STEP file (zero dependencies)
    sf = cadtokens.load("part.stp")
    pmi = cadtokens.extract_pmi(sf)

    # Chat (requires: pythonocc-core, ollama)
    cadtokens.chat("part.stp")
"""

__version__ = "0.1.0"

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


# ── Feature 2: Chat (lazy import — needs ollama + pythonocc) ────────────────
def chat(step_path: str, model: str = "qwen3:14b", **kwargs):
    """Launch interactive CLI chat with a STEP file.

    Requires: pip install ollama pythonocc-core
    """
    from cadtokens.chat.cad_chat import CADChatAgent, interactive_mode
    agent = CADChatAgent(model=model)
    agent.set_step_file(step_path)
    interactive_mode(agent)


def chat_ui(step_path: str = None, **kwargs):
    """Launch Gradio web UI for CAD chat.

    Requires: pip install ollama pythonocc-core gradio
    """
    from cadtokens.chat.chat_ui import build_ui
    build_ui().launch(server_name=kwargs.get("host", "127.0.0.1"),
                      server_port=kwargs.get("port", 7860))


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
    # Chat
    "chat", "chat_ui",
]
