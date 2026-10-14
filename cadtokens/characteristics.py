"""
Inspection characteristic list from STEP AP242 semantic PMI.

Turns extracted PMI into a numbered list of characteristics — the input for
a first article inspection report (AS9102 Form 3, VDA 2 / EMPB, PPAP
ballooned drawing). Each row carries the specification as printed on a
drawing, numeric acceptance limits, datums, modifiers and the B-Rep faces
it applies to.

Zero dependencies (stdlib only).

Usage:
    import cadtokens
    from cadtokens.characteristics import build_characteristics, to_csv

    pmi = cadtokens.extract_pmi(cadtokens.load("part.stp"))
    chars = build_characteristics(pmi)
    to_csv(chars, "part_characteristics.csv")
"""

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union

from cadtokens.iso286 import UnsupportedFit, fit_deviations
from cadtokens.step_engine.pmi import (
    DatumReference, DimensionInfo, PMIData, ToleranceInfo, tolerance_category,
)


# ======================================================================
# Symbols
# ======================================================================

GDT_SYMBOLS = {
    'STRAIGHTNESS': '⏤', 'FLATNESS': '⏥', 'ROUNDNESS': '○', 'CYLINDRICITY': '⌭',
    'LINE_PROFILE': '⌒', 'SURFACE_PROFILE': '⌓',
    'PERPENDICULARITY': '⟂', 'PARALLELISM': '∥', 'ANGULARITY': '∠',
    'POSITION': '⌖', 'CONCENTRICITY': '◎', 'COAXIALITY': '◎', 'SYMMETRY': '⌯',
    'CIRCULAR_RUNOUT': '↗', 'TOTAL_RUNOUT': '⌰',
}

MODIFIER_SYMBOLS = {
    'MAXIMUM_MATERIAL_REQUIREMENT': 'Ⓜ',
    'LEAST_MATERIAL_REQUIREMENT': 'Ⓛ',
    'FREE_STATE': 'Ⓕ',
    'TANGENT_PLANE': 'Ⓣ',
    'STATISTICAL_TOLERANCE': '⟨ST⟩',
    'SEPARATE_REQUIREMENT': 'SEP REQT',
    'COMMON_ZONE': 'CZ',
    'EACH_RADIAL_ELEMENT': 'EACH RADIAL ELEMENT',
    'RECIPROCITY_REQUIREMENT': 'Ⓡ',
    'MAXIMUM_MATERIAL_REQUIREMENT_FOR_DATUM': 'Ⓜ',
}

_DIMENSION_PREFIX = {
    'diameter': '⌀', 'radius': 'R', 'spherical diameter': 'S⌀',
    'spherical radius': 'SR', 'controlled radius': 'CR', 'square': '□',
}

_QUALIFIER_TEXT = {'maximum': 'MAX', 'minimum': 'MIN', 'average': 'AVG'}


# ======================================================================
# Data model
# ======================================================================

@dataclass
class Characteristic:
    """One row of the inspection characteristic list."""
    number: int                         # balloon number
    kind: str                           # 'dimension' | 'geometric_tolerance'
    characteristic: str                 # 'Diameter', 'Position', ...
    specification: str                  # '4X ⌀6 ±0.15', '⌖ ⌀0.5 Ⓜ | A | B'
    nominal: Optional[float] = None
    lower_limit: Optional[float] = None  # lowest acceptable measured value
    upper_limit: Optional[float] = None  # highest acceptable measured value
    unit: Optional[str] = None
    category: str = ''                  # size/location/angle or form/orientation/...
    datums: List[str] = field(default_factory=list)       # ['A', 'B Ⓜ']
    modifiers: List[str] = field(default_factory=list)
    feature_count: int = 1
    applies_to: str = 'faces'           # faces | edges | supplemental geometry | whole part | unlinked
    face_ids: List[int] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    source_entity_id: int = 0
    source_name: str = ''
    notes: List[str] = field(default_factory=list)

    @property
    def is_toleranced(self) -> bool:
        return self.upper_limit is not None or self.lower_limit is not None


# ======================================================================
# Formatting helpers
# ======================================================================

def _num(value: Optional[float], decimals: Optional[int] = None) -> str:
    if value is None:
        return '?'
    if decimals is not None:
        return f"{value:.{decimals}f}"
    text = f"{value:.6f}".rstrip('0').rstrip('.')
    return text if text not in ('-0', '') else '0'


def _signed(value: float, decimals: Optional[int] = None, zero_sign: str = '+') -> str:
    """Tolerance with explicit sign: +0.003, -0.001, and -0.000 for a zero lower bound."""
    if abs(value) < 1e-12:
        return zero_sign + _num(0.0, decimals)
    return ('+' if value > 0 else '-') + _num(abs(value), decimals)


def _modifier_text(modifiers: Iterable[str]) -> List[str]:
    return [MODIFIER_SYMBOLS.get(m, m.replace('_', ' ').lower()) for m in modifiers]


def _datum_text(ref: DatumReference) -> str:
    mods = ' '.join(_modifier_text(ref.modifiers))
    return f"{ref.label} {mods}" if mods else ref.label


def _applies_to(face_ids, edge_ids, other_ids, whole_part) -> str:
    if whole_part:
        return 'whole part'
    if face_ids:
        return 'faces'
    if edge_ids:
        return 'edges'
    if other_ids:
        return 'supplemental geometry'
    return 'unlinked'


# ======================================================================
# Builders
# ======================================================================

_MM_PER_UNIT = {'mm': 1.0, 'in': 25.4, 'm': 1000.0, 'cm': 10.0}


def _fit_deviations(d: DimensionInfo) -> Tuple[float, float]:
    """ISO 286 deviations in the dimension's own unit."""
    scale = _MM_PER_UNIT.get(d.unit or '')
    if scale is None:
        raise UnsupportedFit(f"unit {d.unit!r} unknown")
    lo, up = fit_deviations(d.nominal_value * scale, d.fit, d.fit_kind)
    return round(lo / scale, 7), round(up / scale, 7)


def _dimension_row(d: DimensionInfo) -> Characteristic:
    semantic = (d.semantic_name or '').strip().lower()
    angular = 'ANGULAR' in d.dimension_type
    unit = 'deg' if angular and d.unit in (None, 'deg') else d.unit
    prefix = _DIMENSION_PREFIX.get(semantic, '')
    suffix = '°' if unit == 'deg' else ''
    count = f"{d.feature_count}X " if d.feature_count > 1 else ''
    dec = d.decimals
    notes: List[str] = []

    lower = d.min_value
    upper = d.max_value

    if d.upper_limit is not None or d.lower_limit is not None:
        spec = f"{count}{prefix}{_num(d.upper_limit, dec)}{suffix}/{_num(d.lower_limit, dec)}{suffix}"
    elif d.nominal_value is None:
        spec = f"{count}{prefix}?"
        notes.append('No semantic value in the file (graphical PMI only?) — read from drawing.')
    else:
        spec = f"{count}{prefix}{_num(d.nominal_value, dec)}{suffix}"
        lo, up = d.lower_tolerance, d.upper_tolerance
        if d.fit:
            spec += f" {d.fit}"
            try:
                lo_dev, up_dev = _fit_deviations(d)
            except UnsupportedFit as e:
                notes.append(f"ISO 286 fit {d.fit} — limits not computed ({e}).")
            else:
                limit_dec = None if dec is None else max(dec, 3 if unit == 'mm' else 4)
                spec += f" ({_signed(up_dev, limit_dec)}/{_signed(lo_dev, limit_dec)})"
                lower = round(d.nominal_value + lo_dev, 6)
                upper = round(d.nominal_value + up_dev, 6)
                notes.append(f"Limits per ISO 286-1 ({d.fit_kind or 'hole/shaft by letter case'}).")
        elif lo is not None and up is not None:
            if abs(up + lo) < 1e-12 and up > 0:
                spec += f" ±{_num(up, dec)}"
            else:
                spec += f" {_signed(up, dec)}/{_signed(lo, dec, zero_sign='-')}"
        elif d.qualifier in _QUALIFIER_TEXT:
            spec += f" {_QUALIFIER_TEXT[d.qualifier]}"
            if d.qualifier == 'maximum':
                upper = d.nominal_value
            elif d.qualifier == 'minimum':
                lower = d.nominal_value
        else:
            notes.append('No tolerance in PMI — basic dimension or general tolerance (e.g. ISO 2768) applies.')

    if d.qualifier and d.qualifier not in _QUALIFIER_TEXT and d.qualifier != 'designed':
        notes.append(f"Qualifier: {d.qualifier}")

    location = d.dimension_type in ('DIMENSIONAL_LOCATION', 'DIRECTED_DIMENSIONAL_LOCATION',
                                    'DIMENSIONAL_LOCATION_WITH_PATH', 'ANGULAR_LOCATION')
    return Characteristic(
        number=0,
        kind='dimension',
        characteristic=(d.semantic_name or d.dimension_type.replace('_', ' ').lower()).capitalize(),
        specification=spec,
        nominal=d.nominal_value,
        lower_limit=lower,
        upper_limit=upper,
        unit=unit,
        category='angle' if angular else ('location' if location else 'size'),
        feature_count=d.feature_count,
        applies_to=_applies_to(d.face_ids, d.edge_ids, d.other_geometry_ids, False),
        face_ids=list(d.face_ids),
        edge_ids=list(d.edge_ids),
        source_entity_id=d.entity_id,
        source_name=d.name,
        notes=notes,
    )


def _tolerance_row(t: ToleranceInfo) -> Characteristic:
    symbol = GDT_SYMBOLS.get(t.tolerance_type, t.tolerance_type)
    count = f"{t.feature_count}X " if t.feature_count > 1 else ''
    zone = ('S⌀' if t.zone_form and 'spherical' in t.zone_form else '⌀') if t.is_diametral else ''
    parts = [f"{zone}{_num(t.value)}"]

    if t.unit_basis:
        if t.unit_basis_area_type == 'CIRCULAR':
            parts[0] += f"/⌀{_num(t.unit_basis[0])}"
        else:
            parts[0] += '/' + '×'.join(_num(v) for v in t.unit_basis)
    parts += _modifier_text(t.modifiers)
    if t.unequal_displacement is not None:
        parts.append(f"Ⓤ {_num(t.unequal_displacement)}")
    if t.projected_length is not None:
        parts.append(f"Ⓟ{_num(t.projected_length)}")
    if t.max_value is not None:
        parts.append(f"{_num(t.max_value)} MAX")

    frame = [symbol, ' '.join(parts)] + [_datum_text(r) for r in t.datum_references]
    spec = count + ' | '.join(frame)

    notes: List[str] = []
    if t.all_around:
        notes.append('All around.')
    if t.whole_part:
        notes.append('Applies to the whole part (e.g. profile on all untoleranced surfaces).')
    if 'MAXIMUM_MATERIAL_REQUIREMENT' in t.modifiers or 'LEAST_MATERIAL_REQUIREMENT' in t.modifiers:
        notes.append('Material condition modifier — bonus tolerance depends on actual feature size.')
    if any(r.modifiers for r in t.datum_references):
        notes.append('Datum feature modifier — datum shift may apply.')

    return Characteristic(
        number=0,
        kind='geometric_tolerance',
        characteristic=t.tolerance_type.replace('_', ' ').capitalize(),
        specification=spec,
        nominal=0.0 if t.value is not None else None,
        lower_limit=0.0 if t.value is not None else None,
        upper_limit=t.value,
        unit=t.unit,
        category=tolerance_category(t.tolerance_type),
        datums=[_datum_text(r) for r in t.datum_references],
        modifiers=list(t.modifiers),
        feature_count=t.feature_count,
        applies_to=_applies_to(t.face_ids, t.edge_ids, t.other_geometry_ids, t.whole_part),
        face_ids=list(t.face_ids),
        edge_ids=list(t.edge_ids),
        source_entity_id=t.entity_id,
        source_name=t.name,
        notes=notes,
    )


def build_characteristics(pmi: PMIData, include_untoleranced: bool = True) -> List[Characteristic]:
    """Build a numbered characteristic list from extracted PMI.

    Characteristics on the same feature are kept together, dimensions
    before geometric tolerances — the order a drawing is usually ballooned.
    Within that, file order is preserved, so numbering is deterministic.

    Args:
        pmi: output of cadtokens.extract_pmi()
        include_untoleranced: keep dimensions without tolerance (basic or
            general-tolerance dimensions). Set False for a pure FAI list.
    """
    rows: List[Tuple[Tuple, Characteristic]] = []
    for d in pmi.dimensions:
        rows.append(((tuple(sorted(d.face_ids)) or ('d', d.entity_id)), _dimension_row(d)))
    for t in pmi.tolerances:
        rows.append(((tuple(sorted(t.face_ids)) or ('t', t.entity_id)), _tolerance_row(t)))

    if not include_untoleranced:
        rows = [(k, c) for k, c in rows if c.is_toleranced or c.nominal is None]

    # Group by feature (same face set), groups ordered by first source entity
    groups: Dict[Tuple, List[Characteristic]] = {}
    for key, char in sorted(rows, key=lambda r: r[1].source_entity_id):
        groups.setdefault(key, []).append(char)
    ordered = []
    for chars in groups.values():
        ordered += sorted(chars, key=lambda c: (c.kind != 'dimension', c.source_entity_id))

    for i, char in enumerate(ordered, start=1):
        char.number = i
    return ordered


# ======================================================================
# Export
# ======================================================================

CSV_COLUMNS = [
    'number', 'characteristic', 'specification', 'nominal', 'lower_limit',
    'upper_limit', 'unit', 'feature_count', 'datums', 'kind', 'category',
    'applies_to', 'face_ids', 'source_entity_id', 'source_name', 'notes',
]


def _csv_value(value) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return _num(value)
    if isinstance(value, list):
        return ' '.join(str(v) for v in value) if value and not isinstance(value[0], str) \
            else '; '.join(value)
    return str(value)


def to_csv(chars: List[Characteristic], target: Union[str, TextIO, None] = None,
           delimiter: str = ',', decimal_comma: bool = False) -> str:
    """Write the characteristic list as CSV (UTF-8 with BOM, opens in Excel).

    Args:
        target: file path, open text file, or None to only return the text
        delimiter: ';' for German Excel
        decimal_comma: write 0,15 instead of 0.15 (German Excel)
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator='\n')
    writer.writerow(CSV_COLUMNS)
    for c in chars:
        row = []
        for col in CSV_COLUMNS:
            text = _csv_value(getattr(c, col))
            if decimal_comma and isinstance(getattr(c, col), float):
                text = text.replace('.', ',')
            row.append(text)
        writer.writerow(row)
    text = buf.getvalue()

    if isinstance(target, str):
        with open(target, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    elif target is not None:
        target.write(text)
    return text


def to_dict(chars: List[Characteristic], pmi: Optional[PMIData] = None,
            source: Optional[str] = None) -> Dict:
    """JSON-ready structure with metadata and data-quality warnings."""
    return {
        'source': source,
        'length_unit': pmi.length_unit if pmi else None,
        'count': len(chars),
        'warnings': list(pmi.warnings) if pmi else [],
        'characteristics': [asdict(c) for c in chars],
    }


def to_json(chars: List[Characteristic], pmi: Optional[PMIData] = None,
            source: Optional[str] = None, indent: int = 2) -> str:
    return json.dumps(to_dict(chars, pmi, source), indent=indent, ensure_ascii=False)


def to_table(chars: List[Characteristic]) -> str:
    """Plain-text table for the terminal."""
    header = ('#', 'Characteristic', 'Specification', 'Min', 'Max', 'Unit', 'Applies to')
    rows = [header] + [
        (str(c.number), c.characteristic, c.specification,
         _num(c.lower_limit) if c.lower_limit is not None else '',
         _num(c.upper_limit) if c.upper_limit is not None else '',
         c.unit or '',
         f"{len(c.face_ids)} faces" if c.applies_to == 'faces' else c.applies_to)
        for c in chars
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = ['  '.join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows]
    lines.insert(1, '  '.join('-' * w for w in widths))
    return '\n'.join(lines)
