# cadtokens — Talk to your STEP files

Parse STEP geometry and PMI, chat with CAD models in plain English,
detect manufacturing features with a GNN.

Runs **100% locally**. Your data never leaves your machine.

---

## Demo

```bash
# Parse any STEP file — no dependencies needed
$ cadtokens parse my_part.stp
{
  "entities": 962,
  "faces": 13,
  "edges": 22,
  "points": 403
}

# Extract PMI / GD&T
$ cadtokens pmi my_part.stp

# Chat with your CAD model (requires ollama + pythonocc)
$ cadtokens chat my_part.stp
> How many faces does this part have?
> What surface types are present?
```

## Features

| Feature | Status | Dependencies |
|---|---|---|
| **STEP Parser + PMI** | ✅ Stable | None (pure Python) |
| **Chat Interface** | ✅ Working | pythonocc, ollama |
| **GNN Feature Recognition** | ⚠️ Experimental | torch, torch-geometric, pythonocc |

### STEP Parser + PMI Extraction

Pure Python parser for ISO 10303-21 (STEP Part 21) files.
Zero external dependencies.

- Full entity parsing, reference graph, B-Rep topology extraction
- **PMI/GD&T extraction** — tolerances, dimensions, datums, surface finish
- Face-level PMI mapping — know exactly which tolerance applies to which face
- Edit + re-serialize STEP files (translate, renumber, modify entities)
- Supports AP203, AP214, AP242

### Chat with your CAD files

Natural language Q&A powered by a local LLM (Ollama).
Tool-calling architecture: the LLM queries parsed geometry data, runs OCC
analysis, and returns human-readable answers.

- CLI and Gradio web UI
- No API keys, no cloud, no data leaves your machine
- Extensible — add your own tools

### GNN Feature Recognition ⚠️

12-class manufacturing feature classification on B-Rep faces
(holes, pockets, bosses, fillets, chamfers, etc.).

- BRepGNN with ~4.7M parameters
- 78% inference accuracy on real STEP parts
- Early stage — contributions welcome

---

## Installation

```bash
git clone https://github.com/giltsch/cadtokens.git
cd cadtokens
pip install -e .
```

The core (STEP parser + PMI) has **zero dependencies** — it works with just the clone.

### Optional dependencies

For the chat interface and other features, install extras:

```bash
# Chat (CLI) — needs pythonocc + ollama
pip install -e ".[chat]"

# Web UI — adds Gradio
pip install -e ".[ui]"

# GNN feature recognition (experimental)
pip install -e ".[gnn]"

# Everything
pip install -e ".[all]"
```

**Chat also requires [Ollama](https://ollama.ai) running locally:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:14b
```

---

## Quick Start

### Parse a STEP file (no dependencies needed)

```python
import cadtokens

# Load and parse
sf = cadtokens.load("part.stp")

# Explore topology
g = cadtokens.graph(sf)
print(f"Entity types: {g.entity_count_by_type()}")

brep = cadtokens.BRepExtractor(g)
print(f"Faces: {len(brep.get_faces())}")
print(f"Edges: {len(brep.get_edge_curves())}")
```

### Extract PMI / GD&T

```python
import cadtokens

sf = cadtokens.load("part_with_pmi.stp")
pmi = cadtokens.extract_pmi(sf)

for tol in pmi.tolerances:
    print(f"{tol.tolerance_type}: {tol.value}")
    print(f"  Applied to faces: {tol.face_ids}")
    print(f"  Datum labels: {tol.datum_labels}")

for dim in pmi.dimensions:
    print(f"{dim.dimension_type}: {dim.nominal_value}")
```

### Edit a STEP file

```python
import cadtokens

sf = cadtokens.load("part.stp")
editor = cadtokens.edit(sf)
editor.translate(10.0, 0.0, 0.0)
editor.renumber()
cadtokens.save(sf, "moved_part.stp")
```

### CLI

```bash
# Parse to JSON
cadtokens parse part.stp
cadtokens parse part.stp --full    # include face details

# Extract PMI
cadtokens pmi part.stp

# Interactive chat
cadtokens chat part.stp
cadtokens chat part.stp --model ministral:3b

# Web UI
cadtokens ui
cadtokens ui --port 8080

# Quick geometry summary (requires pythonocc)
cadtokens info part.stp
```

---

## Architecture

```
STEP file
  │
  ├─► step_engine (pure Python)
  │     ├── Parser → StepFile (entities, refs, topology)
  │     ├── ReferenceGraph → face/edge/point extraction
  │     ├── PMIExtractor → tolerances, dimensions, datums
  │     └── StepEditor → modify + re-serialize
  │
  ├─► tools (pythonocc)
  │     ├── GeometryTools → areas, volumes, curvature, normals
  │     ├── StepAnalyzer → detailed face/edge analysis
  │     └── StepValidator → geometry checks
  │
  ├─► chat (ollama + gradio)
  │     ├── CADChatAgent → LLM + tool-calling loop
  │     └── chat_ui → Gradio web interface
  │
  └─► gnn (torch + torch-geometric)  ⚠️ experimental
        ├── BRepGNN → 12-class face segmentation
        └── GNNTools → inference wrapper
```

---

## Package Structure

```
cadtokens/
├── __init__.py          # Top-level API: load, parse, extract_pmi, chat, ...
├── cli.py               # CLI entry point (cadtokens parse/pmi/chat/ui/info)
│
├── step_engine/         # Pure Python STEP parser + PMI (zero deps)
│   ├── model.py         # Data model: StepFile, Entity, Ref, ...
│   ├── tokenizer.py     # Lexer for ISO 10303-21
│   ├── parser.py        # Recursive descent parser
│   ├── serializer.py    # STEP file writer
│   ├── graph.py         # Reference graph + B-Rep extraction
│   ├── editor.py        # Edit entities (translate, renumber, ...)
│   └── pmi.py           # PMI/GD&T extraction (tolerances, datums, ...)
│
├── tools/               # OCC-based geometry analysis
│   ├── geometry_tools.py
│   ├── cad_renderer.py
│   ├── step_analyzer.py
│   └── step_validator.py
│
├── chat/                # LLM chat interface
│   ├── cad_chat.py      # CLI agent (tool-calling loop)
│   └── chat_ui.py       # Gradio web UI
│
└── gnn/                 # GNN feature recognition (experimental)
    ├── brep_gnn.py      # BRepGNN model architecture
    ├── gnn_tools.py     # Inference wrapper
    ├── shape_to_graph.py
    └── cad_graph_dataset.py
```

---

## Related Work

This project builds on research in GNN-based CAD understanding:

- [BrepMFR](https://github.com/zhangshuming0668/BrepMFR) — GNN for machining feature recognition
- [AAGNet](https://github.com/whjdark/AAGNet) — Multi-task GNN for B-Rep
- [Hierarchical CADNet](https://github.com/xupeiwust/hierarchical-cadnet) — Hierarchical GNN from CAD

What makes cadtokens different: **natural language interface**.
Those repos classify faces. This one lets you **ask questions**.

---

## Contributing

Contributions welcome! Especially for:

- GNN accuracy improvements (currently ~78%)
- Additional STEP schema support
- New tool functions for the chat agent
- Documentation and examples

---

## License

MIT
