"""
STEP Engine - Parser

Recursive descent parser for ISO 10303-21 (Part 21) files.
Converts raw text into a structured StepFile with typed entities and parameters.
"""

import re
from typing import Dict, List, Optional, Tuple

from .model import (
    StepFile, Entity, EntityPart, HeaderSection, StepParam,
    Ref, EnumValue, TypedValue, Unset, Derived, UNSET, DERIVED,
)
from .tokenizer import Token, tokenize


class ParseError(Exception):
    """Error during STEP file parsing."""
    pass


class StepParser:
    """Recursive descent parser for STEP Part 21 data sections."""

    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0

    def parse(self, text: str) -> StepFile:
        """Parse a complete STEP file string into a StepFile."""
        header_text, data_text = _split_sections(text)
        header = _parse_header(header_text)

        self.tokens = tokenize(data_text)
        self.pos = 0
        entities = self._parse_entities()

        return StepFile(header=header, entities=entities)

    # ------------------------------------------------------------------
    # Entity parsing
    # ------------------------------------------------------------------

    def _parse_entities(self) -> Dict[int, Entity]:
        entities = {}
        n = len(self.tokens)
        while self.pos < n:
            tok = self.tokens[self.pos]
            if tok.type == 'ENTITY_REF':
                entity = self._parse_entity()
                entities[entity.entity_id] = entity
            else:
                self.pos += 1  # skip stray tokens (section markers etc.)
        return entities

    def _parse_entity(self) -> Entity:
        """Parse: #ID = body ;"""
        ref_tok = self._expect('ENTITY_REF')
        entity_id = int(ref_tok.value[1:])
        self._expect('EQUALS')

        tok = self._peek()
        if tok.type == 'LPAREN':
            # Might be complex entity: ( TYPE1(...) TYPE2(...) )
            # Look ahead: ( followed by KEYWORD = complex entity
            next_tok = self._peek_at(1)
            if next_tok and next_tok.type == 'KEYWORD':
                parts = self._parse_complex_body()
            else:
                # Shouldn't happen in valid STEP, but handle gracefully
                parts = [EntityPart("_UNKNOWN", [self._parse_param()])]
        elif tok.type == 'KEYWORD':
            parts = [self._parse_simple_body()]
        else:
            raise ParseError(
                f"Expected KEYWORD or '(' after '=' at pos {tok.pos}, "
                f"got {tok.type} '{tok.value}'"
            )

        self._expect('SEMI')
        return Entity(entity_id=entity_id, parts=parts)

    def _parse_simple_body(self) -> EntityPart:
        """Parse: TYPE(param, param, ...)"""
        type_tok = self._expect('KEYWORD')
        self._expect('LPAREN')
        params = self._parse_param_list()
        self._expect('RPAREN')
        return EntityPart(type_name=type_tok.value, params=params)

    def _parse_complex_body(self) -> List[EntityPart]:
        """Parse: ( TYPE1(params) TYPE2(params) ... )"""
        self._expect('LPAREN')
        parts = []
        while self.pos < len(self.tokens) and self._peek().type == 'KEYWORD':
            parts.append(self._parse_simple_body())
        self._expect('RPAREN')
        return parts

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def _parse_param_list(self) -> List[StepParam]:
        """Parse comma-separated parameters until ')' is reached."""
        params = []
        if self._peek().type == 'RPAREN':
            return params
        params.append(self._parse_param())
        while self.pos < len(self.tokens) and self._peek().type == 'COMMA':
            self.pos += 1  # consume comma
            params.append(self._parse_param())
        return params

    def _parse_param(self) -> StepParam:
        """Parse a single parameter value."""
        tok = self._peek()

        if tok.type == 'INTEGER':
            self.pos += 1
            return int(tok.value)

        elif tok.type == 'REAL':
            self.pos += 1
            return float(tok.value)

        elif tok.type == 'STRING':
            self.pos += 1
            # Unescape: strip outer quotes, replace '' with '
            return tok.value[1:-1].replace("''", "'")

        elif tok.type == 'ENTITY_REF':
            self.pos += 1
            return Ref(int(tok.value[1:]))

        elif tok.type == 'ENUM':
            self.pos += 1
            return EnumValue(tok.value[1:-1])  # strip dots

        elif tok.type == 'DOLLAR':
            self.pos += 1
            return UNSET

        elif tok.type == 'STAR':
            self.pos += 1
            return DERIVED

        elif tok.type == 'LPAREN':
            # List: (item, item, ...)
            self.pos += 1  # consume (
            items = self._parse_param_list()
            self._expect('RPAREN')
            return items

        elif tok.type == 'KEYWORD':
            # Typed value: TYPE(value) e.g. LENGTH_MEASURE(1.0)
            self.pos += 1
            self._expect('LPAREN')
            inner = self._parse_param_list()
            self._expect('RPAREN')
            if len(inner) == 0:
                val = None
            elif len(inner) == 1:
                val = inner[0]
            else:
                val = inner
            return TypedValue(type_name=tok.value, value=val)

        else:
            raise ParseError(
                f"Unexpected token {tok.type} '{tok.value}' at pos {tok.pos}"
            )

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        if self.pos >= len(self.tokens):
            raise ParseError("Unexpected end of token stream")
        return self.tokens[self.pos]

    def _peek_at(self, offset: int) -> Optional[Token]:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def _expect(self, expected_type: str) -> Token:
        tok = self._peek()
        if tok.type != expected_type:
            raise ParseError(
                f"Expected {expected_type} at pos {tok.pos}, "
                f"got {tok.type} '{tok.value}'"
            )
        self.pos += 1
        return tok


# ======================================================================
# Section splitting and header parsing
# ======================================================================

def _split_sections(text: str) -> Tuple[str, str]:
    """Split a STEP file into header text and data text."""
    data_start = text.find('DATA;')
    if data_start == -1:
        # Try DATA with whitespace
        m = re.search(r'DATA\s*;', text)
        if m:
            data_start = m.start()
        else:
            raise ParseError("No DATA section found in STEP file")

    # Header: between HEADER; and first ENDSEC;
    header_start = text.find('HEADER;')
    if header_start == -1:
        header_text = ""
    else:
        header_end = text.find('ENDSEC;', header_start)
        header_text = text[header_start:header_end + 7] if header_end != -1 else ""

    # Data: between DATA; and next ENDSEC;
    content_start = text.index(';', data_start) + 1
    data_end = text.find('ENDSEC;', content_start)
    if data_end == -1:
        data_end = text.find('END-ISO-10303-21;', content_start)
    if data_end == -1:
        data_end = len(text)
    data_text = text[content_start:data_end]

    return header_text, data_text


def _parse_header(header_text: str) -> HeaderSection:
    """Parse the HEADER section into a HeaderSection."""
    header = HeaderSection(raw=header_text)

    # FILE_DESCRIPTION
    m = re.search(r"FILE_DESCRIPTION\s*\(\s*\(([^)]*)\)", header_text, re.DOTALL)
    if m:
        header.file_description = m.group(1).replace("'", "").strip()

    # FILE_NAME
    m = re.search(r"FILE_NAME\s*\(\s*'([^']*)'", header_text)
    if m:
        header.file_name = m.group(1).strip()

    # FILE_SCHEMA
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(([^)]*)\)\s*\)", header_text)
    if m:
        schemas = re.findall(r"'([^']*)'", m.group(1))
        header.file_schema = schemas if schemas else None

    return header


# ======================================================================
# Public API
# ======================================================================

def parse(text: str) -> StepFile:
    """Parse a STEP file string into a StepFile."""
    return StepParser().parse(text)


def parse_file(path: str) -> StepFile:
    """Parse a STEP file from disk."""
    with open(path, 'r', errors='replace') as f:
        return parse(f.read())
