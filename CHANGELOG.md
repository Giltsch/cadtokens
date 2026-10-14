# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] — unreleased

Focus: read and check the semantic PMI (GD&T) of STEP AP242 files.

### Added
- Semantic PMI: units per value, diametral/spherical zones, projected zones,
  unequally disposed zones, per-unit-length/area tolerances, maximum
  tolerance, all around, whole-part tolerances, common datums (A-B), datum
  modifiers (MMB/LMB), limit dimensions, ISO fits, MAX/MIN qualifiers,
  pattern counts (4X), geometry links to faces/edges/supplemental geometry.
- Data-quality warnings (`PMIData.warnings`): mixed length units, PMI not
  linked to geometry, dimensions without a semantic value.
- Inspection characteristic list (`cadtokens characteristics`): table, CSV
  (incl. German Excel), JSON.
- ISO 286-1 limits for tolerance classes C–S / c–s up to 500 mm.
- Ballooned inspection report PDF (`cadtokens report`), pure Python.
- MCP server (`cadtokens mcp`) for AI agents.
- Regression suite against the NIST MBE PMI test models
  (`scripts/fetch_nist.py`).

### Fixed
- Simple-form tolerances lost their datum references.
- Limit dimensions reported a limit as the nominal value.
- Dimensions with a datum feature were not linked to geometry.
- `pip install -e .` failed (invalid build backend).

### Removed
- Chat UI, Ollama agent, pythonocc-based geometry tools and the experimental
  GNN. The last state is preserved under the git tag `archive/chat-gnn`.

### Changed
- License changed from MIT to Apache-2.0.
- Requires Python 3.10 or newer.

## [0.1.0] — 2026-02

Initial version: STEP parser, PMI extraction, editor, chat and GNN prototypes.
