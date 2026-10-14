"""
STEP Engine - Tokenizer

Fast regex-based tokenizer for ISO 10303-21 data sections.
Handles all Part 21 token types including complex entities,
scientific notation, escaped strings, and enums.
"""

import re
from typing import List, NamedTuple


class Token(NamedTuple):
    type: str
    value: str
    pos: int


# Combined regex pattern - order matters (more specific first).
# Uses named groups for O(1) type identification via lastgroup.
_TOKEN_RE = re.compile(r"""
    (?P<COMMENT>/\*.*?\*/)                                          |
    (?P<STRING>'(?:[^']|'')*')                                      |
    (?P<ENTITY_REF>\#\d+)                                           |
    (?P<ENUM>\.[A-Z_][A-Z0-9_]*\.)                                  |
    (?P<REAL>[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?)             |
    (?P<INTEGER>[-+]?\d+)                                           |
    (?P<KEYWORD>[A-Z_][A-Z0-9_]*)                                   |
    (?P<LPAREN>\()                                                  |
    (?P<RPAREN>\))                                                  |
    (?P<COMMA>,)                                                    |
    (?P<SEMI>;)                                                     |
    (?P<EQUALS>=)                                                   |
    (?P<DOLLAR>\$)                                                  |
    (?P<STAR>\*)                                                    |
    (?P<WS>\s+)
""", re.VERBOSE | re.DOTALL)

# Token types to skip
_SKIP = frozenset(('WS', 'COMMENT'))


def tokenize(text: str) -> List[Token]:
    """Tokenize STEP text into a list of tokens.

    Skips whitespace and comments. Returns a list for fast indexed access
    during parsing. Uses compiled regex with named groups for speed.
    """
    tokens = []
    append = tokens.append  # local ref for speed
    for m in _TOKEN_RE.finditer(text):
        ttype = m.lastgroup
        if ttype not in _SKIP:
            append(Token(ttype, m.group(), m.start()))
    return tokens
