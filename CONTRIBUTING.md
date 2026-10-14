# Contributing to cadtokens

Thanks for helping. cadtokens deliberately does a few things well:

1. read the semantic PMI (GD&T) of STEP AP242 files, mapped to geometry,
2. check it for data-quality problems,
3. turn it into inspection characteristic lists and ballooned reports
   (plus an MCP server so agents can query it).

Contributions that make these more correct or more complete are very
welcome. Features outside this scope (CAD editing, rendering, AI chat) are
likely to be declined — please open an issue to discuss first.

## Reporting a file that is read incorrectly

This is the most valuable contribution. Please include:

- what you expected (e.g. the drawing's feature control frame) and what
  `cadtokens pmi file.stp` or `cadtokens chars file.stp` returned,
- the CAD system and version that exported the file,
- **a file you are allowed to share.** Never attach customer or
  export-controlled data. A small test part that reproduces the issue, or
  the relevant entity lines (`#123=...`) with geometry removed, is ideal.

## Development setup

```bash
git clone https://github.com/giltsch/cadtokens.git
cd cadtokens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/fetch_nist.py      # NIST AP242 test models → tests/data/nist
pytest
```

The core (`cadtokens/`) must stay free of third-party dependencies; only
the optional MCP server may import `mcp`.

## Pull requests

- Add a test for every behaviour change. For PMI semantics, prefer a test
  against a NIST model with the expected value taken from the NIST drawing
  (see `tests/test_nist.py`).
- Never guess values that end up in an inspection document. If something
  cannot be determined from the file, flag it (note or warning) instead.
- Keep the code style of the surrounding module; update `CHANGELOG.md`.

## Support

cadtokens is maintained on a best-effort basis. Issues are usually answered
within a couple of weeks. There is no commercial support.
