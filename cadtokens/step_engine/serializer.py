"""
STEP Engine - Serializer

Converts a StepFile back into valid ISO 10303-21 text.
Handles formatting, line wrapping, and proper STEP encoding.
"""

from typing import List

from .model import (
    StepFile, Entity, EntityPart, HeaderSection, StepParam,
    Ref, EnumValue, TypedValue, Unset, Derived,
)


class StepSerializer:
    """Serialize a StepFile to ISO 10303-21 text."""

    def __init__(self, line_width: int = 80):
        self.line_width = line_width

    def serialize(self, step_file: StepFile) -> str:
        parts = []
        parts.append("ISO-10303-21;")
        parts.append(self._serialize_header(step_file.header))
        parts.append("DATA;")

        for eid in sorted(step_file.entities.keys()):
            entity = step_file.entities[eid]
            parts.append(self._serialize_entity(entity))

        parts.append("ENDSEC;")
        parts.append("END-ISO-10303-21;")
        return "\n".join(parts) + "\n"

    def _serialize_header(self, header: HeaderSection) -> str:
        """Use raw header text if available, otherwise reconstruct."""
        if header.raw:
            return header.raw

        lines = ["HEADER;"]
        desc = header.file_description or "Open CASCADE Model"
        lines.append(f"FILE_DESCRIPTION(('{desc}'),'2;1');")

        name = header.file_name or "step_engine_output"
        lines.append(
            f"FILE_NAME('{name}','2026-01-01T00:00:00',(''),(''),"
            f"'step_engine','step_engine','');"
        )

        schemas = header.file_schema or ["AUTOMOTIVE_DESIGN"]
        schema_str = ",".join(f"'{s}'" for s in schemas)
        lines.append(f"FILE_SCHEMA(({schema_str}));")
        lines.append("ENDSEC;")
        return "\n".join(lines)

    def _serialize_entity(self, entity: Entity) -> str:
        if entity.is_complex:
            part_strs = []
            for part in entity.parts:
                pstr = ",".join(self._fmt(p) for p in part.params)
                part_strs.append(f"{part.type_name}({pstr})")
            body = "( " + " ".join(part_strs) + " )"
        else:
            part = entity.parts[0]
            pstr = ",".join(self._fmt(p) for p in part.params)
            body = f"{part.type_name}({pstr})"

        line = f"#{entity.entity_id} = {body};"

        if len(line) > self.line_width:
            return self._wrap(line)
        return line

    def _fmt(self, param: StepParam) -> str:
        """Format a single parameter to STEP text."""
        if isinstance(param, int):
            return str(param)
        elif isinstance(param, float):
            return self._fmt_float(param)
        elif isinstance(param, str):
            escaped = param.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(param, Ref):
            return f"#{param.entity_id}"
        elif isinstance(param, EnumValue):
            return f".{param.value}."
        elif isinstance(param, TypedValue):
            if param.value is None:
                return f"{param.type_name}()"
            elif isinstance(param.value, list):
                inner = ",".join(self._fmt(p) for p in param.value)
            else:
                inner = self._fmt(param.value)
            return f"{param.type_name}({inner})"
        elif isinstance(param, list):
            items = ",".join(self._fmt(p) for p in param)
            return f"({items})"
        elif isinstance(param, Unset):
            return "$"
        elif isinstance(param, Derived):
            return "*"
        else:
            return str(param)

    def _fmt_float(self, val: float) -> str:
        """Format float in STEP style (always includes decimal point)."""
        if val == 0.0:
            # Preserve sign of negative zero
            if str(val).startswith('-'):
                return "-0."
            return "0."

        # Use enough precision to round-trip
        s = f"{val:.15G}"

        # Ensure decimal point exists
        if '.' not in s and 'E' not in s:
            s += '.'

        # STEP uses uppercase E
        s = s.replace('e', 'E')

        # If scientific notation without decimal: 1E5 -> 1.E5
        if 'E' in s and '.' not in s.split('E')[0]:
            parts = s.split('E')
            s = parts[0] + '.E' + parts[1]

        return s

    def _wrap(self, line: str) -> str:
        """Wrap a long line at commas with indentation."""
        # Find the = sign for alignment
        eq_idx = line.find('=')
        if eq_idx == -1:
            return line

        prefix = line[:eq_idx + 2]  # "#ID = "
        rest = line[eq_idx + 2:]
        indent = "  "

        result_lines = []
        current = prefix
        depth = 0
        in_string = False
        i = 0

        while i < len(rest):
            ch = rest[i]

            if ch == "'" and not in_string:
                in_string = True
            elif ch == "'" and in_string:
                # Check for escaped quote ''
                if i + 1 < len(rest) and rest[i + 1] == "'":
                    current += ch
                    i += 1
                    current += rest[i]
                    i += 1
                    continue
                in_string = False

            if not in_string:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1

            current += ch

            # Wrap after comma if line is getting long
            if ch == ',' and not in_string and len(current) > self.line_width - 5:
                result_lines.append(current)
                current = indent

            i += 1

        if current.strip():
            result_lines.append(current)

        return "\n".join(result_lines)


# ======================================================================
# Public API
# ======================================================================

def serialize(step_file: StepFile, line_width: int = 80) -> str:
    """Serialize a StepFile to ISO 10303-21 text."""
    return StepSerializer(line_width=line_width).serialize(step_file)
