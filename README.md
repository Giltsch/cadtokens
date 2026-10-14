# cadtokens — GD&T out of STEP files. Locally.

Read the semantic PMI (GD&T) of STEP AP242 files, turn it into an
inspection characteristic list, and let AI agents query it through MCP.

- **Pure Python, zero dependencies** for parsing, PMI and characteristic lists
- **Runs 100% locally** — no cloud, no upload, works air-gapped
- **Validated** against the NIST MBE PMI test models

---

## What it does

### 1. PMI extraction from STEP AP242 — mapped to faces

Every tolerance, dimension and datum is resolved to the B-Rep faces (or
edges, axes, or the whole part) it applies to. Values carry their unit;
nothing is assumed.

| Covered | |
|---|---|
| Geometric tolerances | all 15 types, value, unit, ⌀/S⌀ zone, Ⓜ Ⓛ Ⓕ Ⓣ ⟨ST⟩ modifiers, projected zone Ⓟ, unequally disposed Ⓤ, per-unit length/area (0.2/15), maximum tolerance, all around, whole part |
| Datum reference frames | ordered compartments, common datums (A-B), datum modifiers (MMB/LMB) |
| Dimensions | size, location, angular; ± and unilateral tolerances, limit dimensions, ISO fits (H7), MAX/MIN, decimal format, pattern count (4X) |
| Data-quality checks | mixed units, PMI not linked to geometry, dimensions without a semantic value |

### 2. Inspection characteristic list (FAI / EMPB)

A numbered list — the starting point for an AS9102 Form 3, VDA 2 / EMPB or
PPAP report. Characteristics on the same feature stay together, as on a
ballooned drawing.

```
$ cadtokens characteristics nist_ftc_10_asme1_ap242-e2.stp
#   Characteristic    Specification                 Min   Max    Unit  Applies to
--  ----------------  ----------------------------  ----  -----  ----  ---------------------
1   Surface profile   ⌓ | 1.5 | A | B | C           0     1.5    mm    whole part
2   Flatness          ⏥ | 0.2                       0     0.2    mm    1 faces
3   Diameter          ⌀6.00 ±0.15                   5.85  6.15   mm    2 faces
4   Perpendicularity  ⟂ | ⌀0.25 | A                 0     0.25   mm    2 faces
...
10  Diameter          8X ⌀2.50 ±0.15                2.35  2.65   mm    16 faces
11  Position          8X ⌖ | ⌀0.75 | A | B | C      0     0.75   mm    16 faces
...
36  Position          ⌖ | ⌀0.8 | D | E-F-G Ⓜ        0     0.8    mm    4 faces
49  Position          2X ⌖ | ⌀0.8 Ⓟ3.2 | D | E | B  0     0.8    mm    supplemental geometry
```

Export for Excel (`;` and decimal comma for German Excel) or JSON:

```bash
cadtokens chars part.stp -f csv -o part_merkmale.csv --delimiter ";" --decimal-comma
cadtokens chars part.stp -f json -o part_characteristics.json
cadtokens chars part.stp --toleranced-only     # skip basic / general-tolerance dims
```

When the file itself is inconsistent, you are told instead of getting a
silently wrong list:

```
1 data-quality warning(s):
  ! Mixed length units in PMI: tolerances use ['mm'], dimensions use ['in'].
    The file may be exported inconsistently — check values against the drawing.
```

### 3. MCP server — let AI agents read GD&T

Any MCP client (Claude Code, Claude Desktop, local agents) can ask
*"which features have a position tolerance tighter than 0.1 mm, and to which
datums?"* and get exact answers from the file, not a guess from a drawing image.

```bash
pip install -e ".[mcp]"
claude mcp add cadtokens -- cadtokens mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "cadtokens": { "command": "cadtokens", "args": ["mcp"] } } }
```

| Tool | |
|---|---|
| `pmi_summary` | units, counts, datum reference frames, data-quality warnings |
| `list_characteristics` | numbered characteristic list; filter by kind or tolerance band |
| `get_characteristic` | full detail of one characteristic |
| `face_pmi` | all PMI on one face |
| `datum_reference_frames` | datums, their faces, and where each frame is used |
| `export_characteristics` | write CSV/JSON (never overwrites unless asked) |

---

## Validation

Tested against the 17 AP242 files of the
[NIST MBE PMI test models](https://www.nist.gov/el/systems-integration-division-73400/mbe-pmi-validation-and-conformance-testing/download-free-cad)
(CTC, FTC, STC — AP242 editions 1–3, mm and inch, files from several CAD systems):

| | |
|---|---|
| Geometric tolerances found | 362 / 362 |
| … linked to geometry | 360 / 362 — the other 2 have no geometry link in the file (flagged) |
| Dimensions linked to geometry | 268 / 268 |
| Feature control frames vs. NIST FTC-10 drawing | match (spot-checked in `tests/test_nist.py`) |
| Parse + extract | ~5 MB/s, pure Python |

```bash
python scripts/fetch_nist.py      # downloads the NIST models (sha256-pinned)
pytest
```

## Not yet supported

Be aware of these before using the list for a real inspection:

- **ISO 286 fit limits** are not computed — `⌀3.5 G6` is flagged, not guessed
- **Graphical-only PMI** (no semantic representation) cannot be read; such
  dimensions are flagged
- Datum targets, thread designations (M3x0.5-6g), surface texture, notes
- Assemblies (PMI is read per file, without assembly transforms)
- Very large files: memory use is roughly 45× file size

---

## Installation

```bash
git clone https://github.com/giltsch/cadtokens.git
cd cadtokens
pip install -e .              # core: no dependencies
pip install -e ".[mcp]"       # + MCP server
```

## Python API

```python
import cadtokens
from cadtokens.characteristics import build_characteristics, to_csv

pmi = cadtokens.extract_pmi(cadtokens.load("part.stp"))

for t in pmi.tolerances:
    print(t.tolerance_type, t.value, t.unit, t.datum_labels, t.face_ids)

for w in pmi.warnings:
    print("!", w)

chars = build_characteristics(pmi)
to_csv(chars, "part_characteristics.csv")
```

The parser also exposes the STEP entity graph, B-Rep topology
(`cadtokens.BRepExtractor`), a STEP editor and a PMI writer
(`cadtokens.PMIWriter`) for attaching tolerances to faces.

---

## Experimental

Not part of the core, not validated to the same standard:

- **Chat** (`cadtokens chat part.stp`, `cadtokens ui`) — local LLM via Ollama
  over OCC geometry tools. Requires `pythonocc-core` (conda) and Ollama.
  Defaults to a local model and warns if an Ollama `-cloud` model is chosen.
  The web UI binds to localhost.
- **GNN face classification** (`cadtokens/gnn`) — 12 classes derived from
  Fusion 360 Gallery operation labels (e.g. CutSide + cylinder → hole wall),
  ~78% on those derived labels. Not a validated machining-feature recognizer.

## License

MIT
