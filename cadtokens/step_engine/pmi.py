"""
STEP Engine - PMI (Product Manufacturing Information)

Extract, query, and write GD&T annotations (tolerances, dimensions, datums)
from/to STEP AP242 files. Resolves the full entity chain from PMI to geometry
(ADVANCED_FACE) using only graph traversal — no CAD kernel required.

Supports:
  - Geometric tolerances (flatness, perpendicularity, position, profile, etc.)
    with tolerance modifiers, diametral zones, projected zones, unequally
    disposed zones, unit-basis and maximum tolerances
  - Datum reference frames with common datums (A-B) and datum modifiers
  - Dimensions (size, location, angular) with plus/minus tolerances,
    limit dimensions and ISO fits (e.g. H7)
  - Units per value (mm, inch, degree) — never assumed
  - Full PMI-to-geometry resolution via SHAPE_ASPECT → GISU/IIRU → faces,
    edges, supplemental geometry, or the whole part
  - Writing new PMI and attaching it to existing faces

Validated against the NIST MBE PMI test models (CTC/FTC/STC, AP242 e1–e3).
"""

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from .model import (
    StepFile, Entity, EntityPart, Ref, EnumValue, TypedValue,
    Unset, Derived, UNSET, DERIVED, StepParam,
)
from .graph import ReferenceGraph


# ======================================================================
# Data Structures
# ======================================================================

@dataclass
class DatumReference:
    """One compartment of a feature control frame, e.g. 'A', 'A-B', 'B(M)'."""
    label: str                      # 'A', or common datum 'A-B'
    modifiers: List[str] = field(default_factory=list)  # MAXIMUM_MATERIAL_REQUIREMENT, ...


@dataclass
class ToleranceInfo:
    """A geometric tolerance (GD&T) annotation."""
    entity_id: int
    name: str
    tolerance_type: str             # FLATNESS, PERPENDICULARITY, POSITION, ...
    value: Optional[float] = None   # tolerance zone size
    face_ids: List[int] = field(default_factory=list)  # ADVANCED_FACE entity IDs
    datum_labels: List[str] = field(default_factory=list)  # ['A', 'B', 'C'] / ['A-B']
    datum_system_id: Optional[int] = None
    modifiers: List[str] = field(default_factory=list)  # MAXIMUM_MATERIAL_REQUIREMENT, etc.
    shape_aspect_id: Optional[int] = None
    value_entity_id: Optional[int] = None
    is_complex: bool = False        # True if multi-type entity
    unit: Optional[str] = None      # 'mm', 'in', ...
    zone_form: Optional[str] = None  # 'cylindrical or circular', 'spherical', ...
    projected_length: Optional[float] = None
    unequal_displacement: Optional[float] = None
    max_value: Optional[float] = None       # GEOMETRIC_TOLERANCE_WITH_MAXIMUM_TOLERANCE
    unit_basis: List[float] = field(default_factory=list)  # per-unit length/area
    unit_basis_area_type: Optional[str] = None
    datum_references: List[DatumReference] = field(default_factory=list)
    all_around: bool = False
    whole_part: bool = False        # applies to the entire part (e.g. profile all over)
    edge_ids: List[int] = field(default_factory=list)
    other_geometry_ids: List[int] = field(default_factory=list)  # axes, points, curve sets
    feature_count: int = 1          # 4 for a '4X' pattern

    @property
    def is_diametral(self) -> bool:
        """True if the tolerance zone is cylindrical/circular (⌀) or spherical (S⌀)."""
        return bool(self.zone_form) and (
            'cylindrical' in self.zone_form or 'circular' in self.zone_form
            or 'spherical' in self.zone_form)


@dataclass
class DimensionInfo:
    """A dimensional annotation (size, location, angular).

    For DIMENSIONAL_SIZE: all faces in face_ids (single feature).
    For DIMENSIONAL_LOCATION: from_face_ids and to_face_ids are the two
    features whose distance is measured. face_ids = from + to combined.

    Tolerance is given either as nominal ± (upper/lower_tolerance), as limit
    values (upper/lower_limit, nominal may be None), or as an ISO fit.
    """
    entity_id: int
    name: str
    dimension_type: str             # DIMENSIONAL_SIZE, DIMENSIONAL_LOCATION, ...
    semantic_name: str = ""         # 'diameter', 'distance', etc.
    nominal_value: Optional[float] = None
    upper_tolerance: Optional[float] = None
    lower_tolerance: Optional[float] = None
    face_ids: List[int] = field(default_factory=list)       # all faces (union)
    from_face_ids: List[int] = field(default_factory=list)  # first shape_aspect
    to_face_ids: List[int] = field(default_factory=list)    # second shape_aspect (location dims)
    shape_aspect_id: Optional[int] = None
    shape_aspect_id_2: Optional[int] = None                 # second SA for location dims
    plus_minus_id: Optional[int] = None
    unit: Optional[str] = None
    upper_limit: Optional[float] = None     # limit dimensioning
    lower_limit: Optional[float] = None
    fit: Optional[str] = None               # ISO 286 fit, e.g. 'H7'
    fit_kind: Optional[str] = None          # 'hole' / 'shaft'
    qualifier: Optional[str] = None         # 'maximum', 'minimum', 'average', ...
    decimals: Optional[int] = None          # display precision from the value format
    edge_ids: List[int] = field(default_factory=list)
    other_geometry_ids: List[int] = field(default_factory=list)
    feature_count: int = 1

    @property
    def min_value(self) -> Optional[float]:
        """Lowest acceptable value, if the dimension carries a numeric tolerance."""
        if self.lower_limit is not None:
            return self.lower_limit
        if self.nominal_value is not None and self.lower_tolerance is not None:
            return self.nominal_value + self.lower_tolerance
        return None

    @property
    def max_value(self) -> Optional[float]:
        """Highest acceptable value, if the dimension carries a numeric tolerance."""
        if self.upper_limit is not None:
            return self.upper_limit
        if self.nominal_value is not None and self.upper_tolerance is not None:
            return self.nominal_value + self.upper_tolerance
        return None


@dataclass
class DatumInfo:
    """A datum reference."""
    entity_id: int                  # DATUM entity ID
    label: str                      # 'A', 'B', 'C', ...
    datum_feature_id: Optional[int] = None  # DATUM_FEATURE entity ID
    face_ids: List[int] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    other_geometry_ids: List[int] = field(default_factory=list)


@dataclass
class DatumSystemInfo:
    """A datum reference frame (ordered set of datums)."""
    entity_id: int
    name: str
    datum_labels: List[str] = field(default_factory=list)  # ordered: [primary, secondary, ...]
    compartment_ids: List[int] = field(default_factory=list)
    datum_references: List[DatumReference] = field(default_factory=list)


@dataclass
class PMIData:
    """All PMI extracted from a STEP file."""
    tolerances: List[ToleranceInfo] = field(default_factory=list)
    dimensions: List[DimensionInfo] = field(default_factory=list)
    datums: List[DatumInfo] = field(default_factory=list)
    datum_systems: List[DatumSystemInfo] = field(default_factory=list)
    length_unit: Optional[str] = None   # dominant length unit of the PMI values

    @property
    def has_pmi(self) -> bool:
        return bool(self.tolerances or self.dimensions or self.datums)

    def face_annotations(self, face_id: int) -> Dict:
        """Get all PMI annotations for a specific face.

        Tolerances that apply to the whole part (e.g. profile all over)
        are included for every face.
        """
        result = {
            'tolerances': [t for t in self.tolerances
                           if face_id in t.face_ids or t.whole_part],
            'dimensions': [d for d in self.dimensions if face_id in d.face_ids],
            'datums': [d for d in self.datums if face_id in d.face_ids],
        }
        return result

    def summary(self) -> Dict:
        return {
            'tolerances': len(self.tolerances),
            'dimensions': len(self.dimensions),
            'datums': len(self.datums),
            'datum_systems': len(self.datum_systems),
            'annotated_faces': len(self.annotated_faces()),
            'length_unit': self.length_unit,
        }

    def annotated_faces(self) -> Set[int]:
        """All face IDs that have at least one PMI annotation."""
        faces = set()
        for t in self.tolerances:
            faces.update(t.face_ids)
        for d in self.dimensions:
            faces.update(d.face_ids)
        for d in self.datums:
            faces.update(d.face_ids)
        return faces


# ======================================================================
# Tolerance type classification
# ======================================================================

# Simple tolerance types (standalone entity with standard params)
_SIMPLE_TOLERANCE_TYPES = {
    'FLATNESS_TOLERANCE', 'STRAIGHTNESS_TOLERANCE', 'ROUNDNESS_TOLERANCE',
    'CYLINDRICITY_TOLERANCE',
}

# Tolerance types that may appear standalone or in complex entities
_ALL_TOLERANCE_SUBTYPES = {
    'FLATNESS_TOLERANCE', 'STRAIGHTNESS_TOLERANCE', 'ROUNDNESS_TOLERANCE',
    'CYLINDRICITY_TOLERANCE', 'PERPENDICULARITY_TOLERANCE', 'PARALLELISM_TOLERANCE',
    'ANGULARITY_TOLERANCE', 'POSITION_TOLERANCE', 'CONCENTRICITY_TOLERANCE',
    'SYMMETRY_TOLERANCE', 'SURFACE_PROFILE_TOLERANCE', 'LINE_PROFILE_TOLERANCE',
    'CIRCULAR_RUNOUT_TOLERANCE', 'TOTAL_RUNOUT_TOLERANCE',
    'COAXIALITY_TOLERANCE',
}

_TOLERANCE_CATEGORIES = {
    # Form tolerances
    'FLATNESS_TOLERANCE': 'form',
    'STRAIGHTNESS_TOLERANCE': 'form',
    'ROUNDNESS_TOLERANCE': 'form',
    'CYLINDRICITY_TOLERANCE': 'form',
    # Orientation tolerances
    'PERPENDICULARITY_TOLERANCE': 'orientation',
    'PARALLELISM_TOLERANCE': 'orientation',
    'ANGULARITY_TOLERANCE': 'orientation',
    # Location tolerances
    'POSITION_TOLERANCE': 'location',
    'CONCENTRICITY_TOLERANCE': 'location',
    'SYMMETRY_TOLERANCE': 'location',
    'COAXIALITY_TOLERANCE': 'location',
    # Profile tolerances
    'SURFACE_PROFILE_TOLERANCE': 'profile',
    'LINE_PROFILE_TOLERANCE': 'profile',
    # Runout tolerances
    'CIRCULAR_RUNOUT_TOLERANCE': 'runout',
    'TOTAL_RUNOUT_TOLERANCE': 'runout',
}

_DIMENSION_TYPES = (
    'DIMENSIONAL_SIZE', 'DIMENSIONAL_LOCATION',
    'ANGULAR_SIZE', 'ANGULAR_LOCATION',
    'DIRECTED_DIMENSIONAL_LOCATION',
    'DIMENSIONAL_SIZE_WITH_DATUM_FEATURE',
    'DIMENSIONAL_SIZE_WITH_PATH', 'DIMENSIONAL_LOCATION_WITH_PATH',
)
_LOCATION_DIMENSION_TYPES = {
    'DIMENSIONAL_LOCATION', 'DIRECTED_DIMENSIONAL_LOCATION',
    'ANGULAR_LOCATION', 'DIMENSIONAL_LOCATION_WITH_PATH',
}

# Geometry classes a shape aspect can be bound to
_FACE_TYPES = {'ADVANCED_FACE', 'FACE_SURFACE', 'COMPLEX_TRIANGULATED_FACE',
               'TRIANGULATED_FACE'}
_EDGE_TYPES = {'EDGE_CURVE', 'ORIENTED_EDGE', 'SUBEDGE'}
_SOLID_TYPES = {'MANIFOLD_SOLID_BREP', 'BREP_WITH_VOIDS', 'CLOSED_SHELL',
                'SHELL_BASED_SURFACE_MODEL', 'TESSELLATED_SOLID', 'TESSELLATED_SHELL'}

_SI_PREFIXES = {
    'MILLI': 'm', 'CENTI': 'c', 'DECI': 'd', 'MICRO': 'µ', 'NANO': 'n', 'KILO': 'k',
}
_SI_UNIT_SYMBOLS = {'METRE': 'm', 'RADIAN': 'rad', 'STERADIAN': 'sr'}
_CONVERSION_UNIT_SYMBOLS = {
    'inch': 'in', 'foot': 'ft', 'millimetre': 'mm', 'millimeter': 'mm',
    'metre': 'm', 'meter': 'm', 'degree': 'deg', 'radian': 'rad',
}


class _Targets(NamedTuple):
    """Geometry a shape aspect resolves to."""
    faces: List[int]
    edges: List[int]
    others: List[int]
    whole_part: bool


def _dedupe(ids: List[int]) -> List[int]:
    seen: Set[int] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def _param(part: Optional[EntityPart], idx: int) -> StepParam:
    if part is None or len(part.params) <= idx:
        return None
    return part.params[idx]


# ======================================================================
# PMI Extractor
# ======================================================================

class PMIExtractor:
    """Extract PMI data from a parsed STEP file.

    Resolves the full entity chain:
        Tolerance/Dimension → SHAPE_ASPECT → GISU/IIRU → ADVANCED_FACE

    Usage:
        sf = step_engine.load("part_with_pmi.step")
        g = step_engine.graph(sf)
        pmi = PMIExtractor(g).extract()
        for t in pmi.tolerances:
            print(f"{t.tolerance_type}: {t.value} on faces {t.face_ids}")
    """

    def __init__(self, graph: ReferenceGraph):
        self.g = graph
        self.sf = graph.step_file

        # Precomputed lookup tables (built in _build_lookups)
        self._sa_items: Dict[int, List[int]] = {}           # shape aspect → geometry items
        self._sa_children: Dict[int, List[int]] = {}        # parent SA → child SAs
        self._datum_feature_to_datum: Dict[int, int] = {}
        self._datum_to_feature: Dict[int, int] = {}
        self._plus_minus: Dict[int, int] = {}               # dimension → PLUS_MINUS_TOLERANCE
        self._dim_reps: Dict[int, int] = {}                 # dimension → SHAPE_DIMENSION_REPRESENTATION
        self._zone_form: Dict[int, str] = {}                # tolerance → zone form name
        self._zone_of_tolerance: Dict[int, int] = {}        # tolerance → TOLERANCE_ZONE
        self._projected_length: Dict[int, int] = {}         # TOLERANCE_ZONE → length measure
        self._unit_cache: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> PMIData:
        """Extract all PMI from the STEP file."""
        self._build_lookups()
        tolerances = self._extract_tolerances()
        dimensions = self._extract_dimensions()
        return PMIData(
            tolerances=tolerances,
            dimensions=dimensions,
            datums=self._extract_datums(),
            datum_systems=self._extract_datum_systems(),
            length_unit=self._dominant_length_unit(tolerances, dimensions),
        )

    # ------------------------------------------------------------------
    # Lookup table construction
    # ------------------------------------------------------------------

    def _entity(self, ref: StepParam) -> Optional[Entity]:
        if isinstance(ref, Ref):
            return self.sf.entities.get(ref.entity_id)
        return None

    def _build_lookups(self):
        """Build all linking tables in one pass per entity type."""
        sf, g = self.sf, self.g

        # SHAPE_ASPECT → geometry via GEOMETRIC_ITEM_SPECIFIC_USAGE and
        # ITEM_IDENTIFIED_REPRESENTATION_USAGE:
        #   (name, desc, definition=SHAPE_ASPECT, used_rep, identified_item)
        for usage_type in ('GEOMETRIC_ITEM_SPECIFIC_USAGE',
                           'ITEM_IDENTIFIED_REPRESENTATION_USAGE'):
            for eid in g.find_by_type(usage_type):
                part = sf[eid].get_part(usage_type) or sf[eid].parts[0]
                sa_ref, item = _param(part, 2), _param(part, 4)
                if isinstance(sa_ref, Ref):
                    items = self._sa_items.setdefault(sa_ref.entity_id, [])
                    items.extend(self._item_refs(item))

        # SHAPE_ASPECT_RELATIONSHIP(name, desc, relating=parent, related=child)
        for eid in g.find_by_type('SHAPE_ASPECT_RELATIONSHIP'):
            params = sf[eid].params
            if len(params) < 4:
                continue
            parent, child = params[2], params[3]
            if not (isinstance(parent, Ref) and isinstance(child, Ref)):
                continue
            src, tgt = self._entity(parent), self._entity(child)
            if src and tgt and src.has_type('DATUM_FEATURE') and tgt.has_type('DATUM'):
                self._datum_feature_to_datum[parent.entity_id] = child.entity_id
                self._datum_to_feature.setdefault(child.entity_id, parent.entity_id)
            else:
                self._sa_children.setdefault(parent.entity_id, []).append(child.entity_id)

        # PLUS_MINUS_TOLERANCE(range, toleranced_dimension)
        for eid in g.find_by_type('PLUS_MINUS_TOLERANCE'):
            params = sf[eid].params
            if len(params) >= 2 and isinstance(params[1], Ref):
                self._plus_minus.setdefault(params[1].entity_id, eid)

        # DIMENSIONAL_CHARACTERISTIC_REPRESENTATION(dimension, representation)
        for eid in g.find_by_type('DIMENSIONAL_CHARACTERISTIC_REPRESENTATION'):
            params = sf[eid].params
            if len(params) >= 2 and isinstance(params[0], Ref) and isinstance(params[1], Ref):
                self._dim_reps.setdefault(params[0].entity_id, params[1].entity_id)

        # TOLERANCE_ZONE(name, desc, of_shape, product_definitional,
        #                defining_tolerance (list), form)
        for eid in g.find_by_type('TOLERANCE_ZONE'):
            part = sf[eid].get_part('TOLERANCE_ZONE')
            defining, form_ref = _param(part, 4), _param(part, 5)
            form = self._entity(form_ref)
            form_name = form.params[0] if form and form.params and isinstance(form.params[0], str) else None
            for tol_ref in (defining if isinstance(defining, list) else [defining]):
                if isinstance(tol_ref, Ref):
                    self._zone_of_tolerance.setdefault(tol_ref.entity_id, eid)
                    if form_name:
                        self._zone_form.setdefault(tol_ref.entity_id, form_name)

        # PROJECTED_ZONE_DEFINITION(zone, boundaries, projection_end, projected_length)
        for eid in g.find_by_type('PROJECTED_ZONE_DEFINITION'):
            params = sf[eid].params
            if len(params) >= 4 and isinstance(params[0], Ref) and isinstance(params[3], Ref):
                self._projected_length[params[0].entity_id] = params[3].entity_id

    def _item_refs(self, item: StepParam) -> List[int]:
        """Flatten an identified_item (Ref, typed SET, or list) to entity IDs."""
        if isinstance(item, Ref):
            return [item.entity_id]
        if isinstance(item, TypedValue):
            return self._item_refs(item.value)
        if isinstance(item, list):
            out: List[int] = []
            for v in item:
                out.extend(self._item_refs(v))
            return out
        return []

    # ------------------------------------------------------------------
    # Resolve geometry for a shape_aspect
    # ------------------------------------------------------------------

    def _resolve_targets(self, shape_aspect_id: int,
                         _visited: Optional[Set[int]] = None) -> _Targets:
        """Resolve a SHAPE_ASPECT (possibly composite) to the geometry it names."""
        visited = _visited if _visited is not None else set()
        faces: List[int] = []
        edges: List[int] = []
        others: List[int] = []
        whole_part = False

        if shape_aspect_id in visited:
            return _Targets(faces, edges, others, whole_part)
        visited.add(shape_aspect_id)

        sa = self.sf.entities.get(shape_aspect_id)
        if sa is not None and sa.has_type('PRODUCT_DEFINITION_SHAPE'):
            # Tolerance applied to the product shape itself, e.g. profile all over
            return _Targets(faces, edges, others, True)

        for item_id in self._sa_items.get(shape_aspect_id, []):
            item = self.sf.entities.get(item_id)
            if item is None:
                continue
            types = set(item.type_names)
            if types & _FACE_TYPES:
                faces.append(item_id)
            elif types & _EDGE_TYPES:
                edges.append(item_id)
            elif types & _SOLID_TYPES:
                whole_part = True
            else:
                others.append(item_id)

        for child_id in self._sa_children.get(shape_aspect_id, []):
            child = self._resolve_targets(child_id, visited)
            faces.extend(child.faces)
            edges.extend(child.edges)
            others.extend(child.others)
            whole_part = whole_part or child.whole_part

        return _Targets(_dedupe(faces), _dedupe(edges), _dedupe(others), whole_part)

    def _resolve_faces(self, shape_aspect_id: int) -> List[int]:
        """Resolve a SHAPE_ASPECT (possibly composite) to face IDs."""
        return self._resolve_targets(shape_aspect_id).faces

    def _feature_count(self, shape_aspect_id: Optional[int]) -> int:
        """Number of features in a pattern ('4X'), 1 for a single feature."""
        sa = self.sf.entities.get(shape_aspect_id) if shape_aspect_id else None
        if sa is None or not sa.has_type('COMPOSITE_GROUP_SHAPE_ASPECT'):
            return 1
        part = sa.get_part('COMPOSITE_GROUP_SHAPE_ASPECT') or sa.get_part('SHAPE_ASPECT') or sa.parts[0]
        desc = _param(part, 1)
        if isinstance(desc, str) and 'pattern' in desc.lower():
            return max(1, len(self._sa_children.get(shape_aspect_id, [])))
        return 1

    # ------------------------------------------------------------------
    # Units and measures
    # ------------------------------------------------------------------

    def _unit_label(self, unit_id: int) -> Optional[str]:
        """Short unit symbol ('mm', 'in', 'deg', ...) for a unit entity."""
        if unit_id in self._unit_cache:
            return self._unit_cache[unit_id]
        label = None
        unit = self.sf.entities.get(unit_id)
        if unit is not None:
            cbu = unit.get_part('CONVERSION_BASED_UNIT')
            si = unit.get_part('SI_UNIT')
            if cbu and cbu.params and isinstance(cbu.params[0], str):
                name = cbu.params[0].strip().lower()
                label = _CONVERSION_UNIT_SYMBOLS.get(name, name)
            elif si and len(si.params) >= 2:
                prefix, name = si.params[-2], si.params[-1]
                symbol = _SI_UNIT_SYMBOLS.get(name.value if isinstance(name, EnumValue) else '', None)
                if symbol:
                    pre = _SI_PREFIXES.get(prefix.value, '') if isinstance(prefix, EnumValue) else ''
                    label = pre + symbol
        self._unit_cache[unit_id] = label
        return label

    def _measure(self, entity_id: Optional[int]) -> Tuple[Optional[float], Optional[str]]:
        """Numeric value and unit of a measure entity (MEASURE_WITH_UNIT and kin)."""
        entity = self.sf.entities.get(entity_id) if entity_id else None
        if entity is None:
            return None, None

        for part in entity.parts:
            if not part.type_name.endswith('MEASURE_WITH_UNIT') or len(part.params) < 2:
                continue
            raw, unit_ref = part.params[0], part.params[1]
            value = raw.value if isinstance(raw, TypedValue) else raw
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                unit = self._unit_label(unit_ref.entity_id) if isinstance(unit_ref, Ref) else None
                return float(value), unit

        # Fallback: first TypedValue or numeric in any part
        for part in entity.parts:
            for p in part.params:
                if isinstance(p, TypedValue) and isinstance(p.value, (int, float)):
                    return float(p.value), None
                if isinstance(p, (int, float)) and not isinstance(p, bool):
                    return float(p), None
        return None, None

    def _extract_measure_value(self, entity_id: int) -> Optional[float]:
        """Extract a numeric value from a MEASURE_WITH_UNIT or similar entity."""
        return self._measure(entity_id)[0]

    @staticmethod
    def _dominant_length_unit(tolerances, dimensions) -> Optional[str]:
        counts: Dict[str, int] = {}
        for item in list(tolerances) + [d for d in dimensions if 'ANGULAR' not in d.dimension_type]:
            if item.unit and item.unit not in ('deg', 'rad'):
                counts[item.unit] = counts.get(item.unit, 0) + 1
        return max(counts, key=counts.get) if counts else None

    # ------------------------------------------------------------------
    # Extract geometric tolerances
    # ------------------------------------------------------------------

    def _extract_tolerances(self) -> List[ToleranceInfo]:
        candidates: Set[int] = set(self.g.find_by_type('GEOMETRIC_TOLERANCE'))
        for ttype in _ALL_TOLERANCE_SUBTYPES:
            candidates.update(self.g.find_by_type(ttype))

        tolerances = []
        for eid in sorted(candidates):
            info = self._extract_tolerance(self.sf[eid])
            if info:
                tolerances.append(info)
        return tolerances

    def _extract_tolerance(self, entity: Entity) -> Optional[ToleranceInfo]:
        """Extract a geometric tolerance in simple or complex (multi-type) form.

        Simple:  POSITION_TOLERANCE(name, desc, magnitude, toleranced_aspect[, datum_system])
        Complex: ( GEOMETRIC_TOLERANCE(name, desc, magnitude, toleranced_aspect)
                   GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE((datum_system))
                   GEOMETRIC_TOLERANCE_WITH_MODIFIERS((.MAXIMUM_MATERIAL_REQUIREMENT.))
                   POSITION_TOLERANCE() ... )
        """
        ttype = 'GEOMETRIC'
        for type_name in entity.type_names:
            if type_name in _ALL_TOLERANCE_SUBTYPES:
                ttype = type_name.replace('_TOLERANCE', '')
                break

        core = entity.get_part('GEOMETRIC_TOLERANCE') if entity.is_complex else entity.parts[0]
        if core is None or len(core.params) < 4:
            return None

        name = core.params[0] if isinstance(core.params[0], str) else ''
        value_ref = core.params[2] if isinstance(core.params[2], Ref) else None
        sa_ref = core.params[3] if isinstance(core.params[3], Ref) else None
        value, unit = self._measure(value_ref.entity_id) if value_ref else (None, None)

        # Datum reference frame: complex GTWDR part, or 5th param of simple form
        ds_param = None
        gtdr = entity.get_part('GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE')
        if gtdr and gtdr.params:
            ds_param = gtdr.params[0]
        elif not entity.is_complex and len(core.params) >= 5:
            ds_param = core.params[4]
        datum_system_id = None
        datum_refs: List[DatumReference] = []
        for ds_ref in (ds_param if isinstance(ds_param, list) else [ds_param]):
            if isinstance(ds_ref, Ref):
                datum_system_id = ds_ref.entity_id
                datum_refs = self._resolve_datum_system(ds_ref.entity_id)
                break

        modifiers: List[str] = []
        gtwm = entity.get_part('GEOMETRIC_TOLERANCE_WITH_MODIFIERS')
        if gtwm and gtwm.params and isinstance(gtwm.params[0], list):
            modifiers = [m.value for m in gtwm.params[0] if isinstance(m, EnumValue)]

        unit_basis: List[float] = []
        area_type = None
        gtdu = entity.get_part('GEOMETRIC_TOLERANCE_WITH_DEFINED_UNIT')
        if gtdu and gtdu.params and isinstance(gtdu.params[0], Ref):
            size = self._measure(gtdu.params[0].entity_id)[0]
            if size is not None:
                unit_basis.append(size)
        gtda = entity.get_part('GEOMETRIC_TOLERANCE_WITH_DEFINED_AREA_UNIT')
        if gtda and gtda.params:
            if isinstance(gtda.params[0], EnumValue):
                area_type = gtda.params[0].value
            if len(gtda.params) > 1 and isinstance(gtda.params[1], Ref):
                second = self._measure(gtda.params[1].entity_id)[0]
                if second is not None:
                    unit_basis.append(second)

        max_value = None
        gtmax = entity.get_part('GEOMETRIC_TOLERANCE_WITH_MAXIMUM_TOLERANCE')
        if gtmax and gtmax.params and isinstance(gtmax.params[0], Ref):
            max_value = self._measure(gtmax.params[0].entity_id)[0]

        unequal = None
        udgt = entity.get_part('UNEQUALLY_DISPOSED_GEOMETRIC_TOLERANCE')
        if udgt and udgt.params and isinstance(udgt.params[0], Ref):
            unequal = self._measure(udgt.params[0].entity_id)[0]

        projected = None
        zone_id = self._zone_of_tolerance.get(entity.entity_id)
        if zone_id in self._projected_length:
            projected = self._measure(self._projected_length[zone_id])[0]

        targets = self._resolve_targets(sa_ref.entity_id) if sa_ref else _Targets([], [], [], False)
        sa_entity = self._entity(sa_ref)

        return ToleranceInfo(
            entity_id=entity.entity_id,
            name=name,
            tolerance_type=ttype,
            value=value,
            face_ids=targets.faces,
            datum_labels=[d.label for d in datum_refs],
            datum_system_id=datum_system_id,
            modifiers=modifiers,
            shape_aspect_id=sa_ref.entity_id if sa_ref else None,
            value_entity_id=value_ref.entity_id if value_ref else None,
            is_complex=entity.is_complex,
            unit=unit,
            zone_form=self._zone_form.get(entity.entity_id),
            projected_length=projected,
            unequal_displacement=unequal,
            max_value=max_value,
            unit_basis=unit_basis,
            unit_basis_area_type=area_type,
            datum_references=datum_refs,
            all_around=bool(sa_entity and sa_entity.has_type('ALL_AROUND_SHAPE_ASPECT')),
            whole_part=targets.whole_part,
            edge_ids=targets.edges,
            other_geometry_ids=targets.others,
            feature_count=self._feature_count(sa_ref.entity_id if sa_ref else None),
        )

    # Kept for backwards compatibility with code calling the old helpers
    def _extract_simple_tolerance(self, entity: Entity, ttype: str) -> Optional[ToleranceInfo]:
        return self._extract_tolerance(entity)

    def _extract_complex_tolerance(self, entity: Entity) -> Optional[ToleranceInfo]:
        return self._extract_tolerance(entity)

    # ------------------------------------------------------------------
    # Extract dimensions
    # ------------------------------------------------------------------

    def _extract_dimensions(self) -> List[DimensionInfo]:
        # A complex entity can carry several dimension types — extract it once
        dims: Dict[int, str] = {}
        for dtype in _DIMENSION_TYPES:
            for eid in self.g.find_by_type(dtype):
                dims.setdefault(eid, dtype)

        dimensions = []
        for eid in sorted(dims):
            info = self._extract_dimension(self.sf[eid], dims[eid])
            if info:
                dimensions.append(info)
        return dimensions

    def _extract_dimension(self, entity: Entity,
                           dtype: str) -> Optional[DimensionInfo]:
        """Extract info from a dimensional entity."""
        part = entity.get_part(dtype) or entity.parts[0]
        params = part.params
        if not params:
            return None

        # Two layouts:
        #   DIMENSIONAL_SIZE(shape_aspect_ref, name_string)
        #   DIMENSIONAL_LOCATION(name, desc, shape_aspect_ref1, shape_aspect_ref2)
        sa_ref = None
        sa_ref_2 = None
        semantic = ''

        if dtype in _LOCATION_DIMENSION_TYPES:
            if isinstance(params[0], str):
                semantic = params[0]
            if len(params) >= 3 and isinstance(params[2], Ref):
                sa_ref = params[2]
            if len(params) >= 4 and isinstance(params[3], Ref):
                sa_ref_2 = params[3]
        else:
            # DIMENSIONAL_SIZE_WITH_DATUM_FEATURE is itself a shape aspect:
            #   (name, desc, of_shape, product_definitional, applies_to, name)
            # so take the first entity reference that is not the product shape.
            for p in params:
                ref_entity = self._entity(p)
                if ref_entity is not None and not ref_entity.has_type('PRODUCT_DEFINITION_SHAPE'):
                    sa_ref = p
                    break
            for p in params:
                if isinstance(p, str) and p:
                    semantic = p
                    break

        empty = _Targets([], [], [], False)
        t1 = self._resolve_targets(sa_ref.entity_id) if sa_ref else empty
        t2 = self._resolve_targets(sa_ref_2.entity_id) if sa_ref_2 else empty

        info = DimensionInfo(
            entity_id=entity.entity_id,
            name=entity.type_name,
            dimension_type=dtype,
            semantic_name=semantic,
            face_ids=_dedupe(t1.faces + t2.faces),
            from_face_ids=t1.faces,
            to_face_ids=t2.faces,
            shape_aspect_id=sa_ref.entity_id if sa_ref else None,
            shape_aspect_id_2=sa_ref_2.entity_id if sa_ref_2 else None,
            edge_ids=_dedupe(t1.edges + t2.edges),
            other_geometry_ids=_dedupe(t1.others + t2.others),
            feature_count=self._feature_count(sa_ref.entity_id if sa_ref else None),
        )
        self._fill_dimension_values(info)
        self._fill_dimension_tolerance(info)
        return info

    def _fill_dimension_values(self, info: DimensionInfo):
        """Nominal / limit values from SHAPE_DIMENSION_REPRESENTATION items.

        Each item is a measure named 'nominal value', 'upper limit' or
        'lower limit', optionally qualified (TYPE_QUALIFIER('maximum'),
        VALUE_FORMAT_TYPE_QUALIFIER('NR2 1.2')).
        """
        rep = self.sf.entities.get(self._dim_reps.get(info.entity_id, -1))
        if rep is None or len(rep.params) < 2 or not isinstance(rep.params[1], list):
            return

        for item_ref in rep.params[1]:
            item = self._entity(item_ref)
            if item is None:
                continue
            value, unit = self._measure(item.entity_id)
            ri = item.get_part('REPRESENTATION_ITEM')
            role = ri.params[0].strip().lower() if ri and ri.params and isinstance(ri.params[0], str) else ''

            qri = item.get_part('QUALIFIED_REPRESENTATION_ITEM')
            for q_ref in (qri.params[0] if qri and qri.params and isinstance(qri.params[0], list) else []):
                self._apply_qualifier(info, self._entity(q_ref))

            if value is None:
                continue
            if unit and info.unit is None:
                info.unit = unit
            if role == 'upper limit':
                info.upper_limit = value
            elif role == 'lower limit':
                info.lower_limit = value
            elif info.nominal_value is None:
                info.nominal_value = value

    @staticmethod
    def _apply_qualifier(info: DimensionInfo, q: Optional[Entity]):
        if q is None or not q.params or not isinstance(q.params[0], str):
            return
        text = q.params[0].strip()
        if q.has_type('VALUE_FORMAT_TYPE_QUALIFIER') or text.upper().startswith('NR'):
            # 'NR2 1.2' → 2 decimals
            _, _, spec = text.partition(' ')
            if '.' in spec:
                try:
                    info.decimals = int(spec.split('.')[1])
                except ValueError:
                    pass
        elif q.has_type('TYPE_QUALIFIER'):
            info.qualifier = text.lower()

    def _fill_dimension_tolerance(self, info: DimensionInfo):
        """± tolerance or ISO fit from PLUS_MINUS_TOLERANCE(range, dimension)."""
        pm_id = self._plus_minus.get(info.entity_id)
        if pm_id is None:
            return
        info.plus_minus_id = pm_id
        rng = self._entity(self.sf[pm_id].params[0])
        if rng is None:
            return

        if rng.has_type('TOLERANCE_VALUE') and len(rng.params) >= 2:
            # TOLERANCE_VALUE(lower_bound, upper_bound)
            if isinstance(rng.params[0], Ref):
                info.lower_tolerance = self._measure(rng.params[0].entity_id)[0]
            if isinstance(rng.params[1], Ref):
                info.upper_tolerance = self._measure(rng.params[1].entity_id)[0]
        elif rng.has_type('LIMITS_AND_FITS'):
            # LIMITS_AND_FITS(form_variance, zone_variance, grade, source)
            texts = [p.strip() if isinstance(p, str) else '' for p in rng.params[:4]]
            texts += [''] * (4 - len(texts))
            form_variance, zone_variance, grade = texts[0], texts[1], texts[2]
            if zone_variance.lower() in ('hole', 'shaft'):
                info.fit_kind = zone_variance.lower()
                info.fit = (form_variance + grade) or None
            else:
                info.fit = (form_variance + zone_variance + grade) or None

    # ------------------------------------------------------------------
    # Extract datums
    # ------------------------------------------------------------------

    def _extract_datums(self) -> List[DatumInfo]:
        datums = []

        for eid in self.g.find_by_type('DATUM'):
            entity = self.sf[eid]
            # DATUM(name, desc, product_def_shape, product_definitional, label)
            label = ''
            if len(entity.params) >= 5 and isinstance(entity.params[4], str):
                label = entity.params[4]

            df_id = self._datum_to_feature.get(eid)
            targets = self._resolve_targets(df_id) if df_id else _Targets([], [], [], False)

            datums.append(DatumInfo(
                entity_id=eid,
                label=label,
                datum_feature_id=df_id,
                face_ids=targets.faces,
                edge_ids=targets.edges,
                other_geometry_ids=targets.others,
            ))

        return datums

    def _extract_datum_systems(self) -> List[DatumSystemInfo]:
        systems = []

        for eid in self.g.find_by_type('DATUM_SYSTEM'):
            entity = self.sf[eid]
            name = entity.params[0] if isinstance(entity.params[0], str) else ''

            refs = self._resolve_datum_system(eid)
            compartments = []
            if len(entity.params) >= 5 and isinstance(entity.params[4], list):
                compartments = [c.entity_id for c in entity.params[4] if isinstance(c, Ref)]

            systems.append(DatumSystemInfo(
                entity_id=eid,
                name=name,
                datum_labels=[r.label for r in refs],
                compartment_ids=compartments,
                datum_references=refs,
            ))

        return systems

    def _resolve_datum_system_labels(self, ds_id: int) -> List[str]:
        """Resolve a DATUM_SYSTEM entity to ordered datum labels ['A','B','C']."""
        return [r.label for r in self._resolve_datum_system(ds_id)]

    def _resolve_datum_system(self, ds_id: int) -> List[DatumReference]:
        """Resolve a DATUM_SYSTEM to its ordered compartments.

        DATUM_SYSTEM(name, desc, of_shape, product_definitional, (compartments))
        DATUM_REFERENCE_COMPARTMENT(name, desc, of_shape, pd, base, modifiers)
          base: DATUM, or COMMON_DATUM_LIST((DATUM_REFERENCE_ELEMENT, ...))
        """
        entity = self.sf.entities.get(ds_id)
        if not entity or not entity.has_type('DATUM_SYSTEM'):
            return []
        refs = []
        constituents = entity.params[4] if len(entity.params) >= 5 else []
        for comp_ref in (constituents if isinstance(constituents, list) else []):
            comp = self._entity(comp_ref)
            if comp is None or len(comp.params) < 5:
                continue
            labels, modifiers = self._resolve_datum_base(comp.params[4])
            modifiers += self._datum_modifiers(comp.params[5] if len(comp.params) > 5 else None)
            if labels:
                refs.append(DatumReference(label='-'.join(labels),
                                           modifiers=_dedupe_str(modifiers)))
        return refs

    def _resolve_datum_base(self, base: StepParam) -> Tuple[List[str], List[str]]:
        """Labels and element modifiers of a compartment/element base."""
        if isinstance(base, TypedValue):          # COMMON_DATUM_LIST((elements))
            base = base.value
        if isinstance(base, list):
            labels, modifiers = [], []
            for element_ref in base:
                element = self._entity(element_ref)
                if element is None or len(element.params) < 5:
                    continue
                sub_labels, sub_mods = self._resolve_datum_base(element.params[4])
                labels += sub_labels
                modifiers += sub_mods
                modifiers += self._datum_modifiers(element.params[5] if len(element.params) > 5 else None)
            return _dedupe_str(labels), modifiers
        datum = self._entity(base)
        if datum is not None and datum.has_type('DATUM') and len(datum.params) >= 5:
            if isinstance(datum.params[4], str) and datum.params[4]:
                return [datum.params[4]], []
        return [], []

    def _datum_modifiers(self, mods: StepParam) -> List[str]:
        """SIMPLE_DATUM_REFERENCE_MODIFIER(.X.) values and DATUM_REFERENCE_MODIFIER_WITH_VALUE."""
        out = []
        for m in (mods if isinstance(mods, list) else []):
            if isinstance(m, TypedValue) and isinstance(m.value, EnumValue):
                out.append(m.value.value)
            elif isinstance(m, EnumValue):
                out.append(m.value)
            elif isinstance(m, Ref):
                e = self._entity(m)
                if e is not None and e.params and isinstance(e.params[0], EnumValue):
                    value = self._measure(e.params[1].entity_id)[0] if len(e.params) > 1 and isinstance(e.params[1], Ref) else None
                    out.append(e.params[0].value if value is None else f"{e.params[0].value} {value:g}")
        return out


def _dedupe_str(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    return [i for i in items if not (i in seen or seen.add(i))]


# ======================================================================
# PMI Writer
# ======================================================================

class PMIWriter:
    """Write PMI annotations to a STEP file and attach them to faces.

    Creates the full entity chain:
        ADVANCED_FACE ← GISU ← SHAPE_ASPECT ← Tolerance/Dimension/Datum

    Usage:
        sf = step_engine.load("part.step")
        e = step_engine.edit(sf)
        w = PMIWriter(sf, e)

        w.add_flatness_tolerance(face_id=17, value=0.05)
        w.add_dimension(face_ids=[100, 200], nominal=50.0,
                        upper=0.1, lower=-0.1, name='diameter')
        w.add_datum(face_id=17, label='A')

        step_engine.save(sf, "part_with_pmi.step")
    """

    def __init__(self, step_file: StepFile, editor=None):
        from .editor import StepEditor
        self.sf = step_file
        self.editor = editor or StepEditor(step_file)
        self._brep_rep_id: Optional[int] = None
        self._pds_id: Optional[int] = None
        self._unit_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Resolve context entities
    # ------------------------------------------------------------------

    def _get_brep_rep_id(self) -> int:
        """Find the ADVANCED_BREP_SHAPE_REPRESENTATION entity."""
        if self._brep_rep_id:
            return self._brep_rep_id
        g = self.editor.graph
        ids = g.find_by_type('ADVANCED_BREP_SHAPE_REPRESENTATION')
        if ids:
            self._brep_rep_id = ids[0]
            return self._brep_rep_id
        raise ValueError("No ADVANCED_BREP_SHAPE_REPRESENTATION found in file")

    def _get_pds_id(self) -> int:
        """Find the PRODUCT_DEFINITION_SHAPE entity."""
        if self._pds_id:
            return self._pds_id
        g = self.editor.graph
        ids = g.find_by_type('PRODUCT_DEFINITION_SHAPE')
        if ids:
            self._pds_id = ids[0]
            return self._pds_id
        raise ValueError("No PRODUCT_DEFINITION_SHAPE found in file")

    def _get_length_unit_id(self) -> int:
        """Find (or create) a LENGTH_UNIT entity for measures."""
        if self._unit_id:
            return self._unit_id
        g = self.editor.graph
        # Look for existing SI length unit
        for eid in g.find_by_type('LENGTH_UNIT'):
            self._unit_id = eid
            return eid
        for eid in g.find_by_type('SI_UNIT'):
            e = self.sf[eid]
            if e.has_type('LENGTH_UNIT'):
                self._unit_id = eid
                return eid
        # Search all complex entities for one that has LENGTH_UNIT
        for eid, entity in self.sf.entities.items():
            if entity.is_complex and entity.has_type('LENGTH_UNIT'):
                self._unit_id = eid
                return eid
        raise ValueError("No LENGTH_UNIT found in file")

    # ------------------------------------------------------------------
    # Helper: create measure value entities
    # ------------------------------------------------------------------

    def _add_length_measure(self, value: float) -> int:
        """Create a LENGTH_MEASURE_WITH_UNIT + MEASURE_WITH_UNIT complex entity."""
        unit_id = self._get_length_unit_id()
        return self.editor.add_complex_entity([
            ('LENGTH_MEASURE_WITH_UNIT', []),
            ('MEASURE_REPRESENTATION_ITEM', []),
            ('MEASURE_WITH_UNIT', [
                TypedValue('LENGTH_MEASURE', value),
                Ref(unit_id),
            ]),
            ('REPRESENTATION_ITEM', ['']),
        ])

    def _add_shape_aspect(self, name: str = '', desc: str = 'GDT') -> int:
        """Create a SHAPE_ASPECT entity."""
        pds_id = self._get_pds_id()
        return self.editor.add_entity('SHAPE_ASPECT', [
            name, desc, Ref(pds_id), EnumValue('T'),
        ])

    def _add_gisu(self, shape_aspect_id: int, face_id: int,
                  desc: str = 'GDT') -> int:
        """Create a GEOMETRIC_ITEM_SPECIFIC_USAGE linking SA to face."""
        brep_id = self._get_brep_rep_id()
        return self.editor.add_entity('GEOMETRIC_ITEM_SPECIFIC_USAGE', [
            '', desc, Ref(shape_aspect_id), Ref(brep_id), Ref(face_id),
        ])

    def _add_composite_shape_aspect(self, child_sa_ids: List[int],
                                    name: str = '',
                                    desc: str = 'multiple elements') -> int:
        """Create a COMPOSITE_SHAPE_ASPECT with child SHAPE_ASPECTs."""
        pds_id = self._get_pds_id()
        csa_id = self.editor.add_entity('COMPOSITE_SHAPE_ASPECT', [
            name, desc, Ref(pds_id), EnumValue('T'),
        ])
        # Link children via SHAPE_ASPECT_RELATIONSHIP
        for child_id in child_sa_ids:
            self.editor.add_entity('SHAPE_ASPECT_RELATIONSHIP', [
                '', UNSET, Ref(csa_id), Ref(child_id),
            ])
        return csa_id

    # ------------------------------------------------------------------
    # Add tolerances
    # ------------------------------------------------------------------

    def add_tolerance(self, tolerance_type: str, face_ids: List[int],
                      value: float, name: str = '',
                      datum_labels: Optional[List[str]] = None,
                      modifiers: Optional[List[str]] = None,
                      datum_system_id: Optional[int] = None) -> int:
        """Add a geometric tolerance to one or more faces.

        Args:
            tolerance_type: e.g. 'FLATNESS', 'PERPENDICULARITY', 'POSITION', ...
            face_ids: ADVANCED_FACE entity IDs to annotate
            value: tolerance zone size (mm)
            name: annotation name
            datum_labels: ['A', 'B'] (creates datum system if needed)
            modifiers: ['MAXIMUM_MATERIAL_REQUIREMENT', ...]
            datum_system_id: existing DATUM_SYSTEM entity ID (overrides datum_labels)

        Returns:
            Entity ID of the created tolerance.
        """
        entity_type = tolerance_type.upper()
        if not entity_type.endswith('_TOLERANCE'):
            entity_type += '_TOLERANCE'

        # Create shape aspects and link to faces
        if len(face_ids) == 1:
            sa_id = self._add_shape_aspect(desc='GDT')
            self._add_gisu(sa_id, face_ids[0])
        else:
            child_ids = []
            for fid in face_ids:
                child_sa = self._add_shape_aspect(desc='GDT')
                self._add_gisu(child_sa, fid)
                child_ids.append(child_sa)
            sa_id = self._add_composite_shape_aspect(child_ids)

        # Create tolerance value
        val_id = self._add_length_measure(value)

        # Determine if we need a complex entity (datum refs or modifiers)
        needs_datum = bool(datum_labels or datum_system_id)
        needs_modifier = bool(modifiers)

        if needs_datum or needs_modifier:
            # Build complex entity
            parts = [
                ('GEOMETRIC_TOLERANCE', [name, '', Ref(val_id), Ref(sa_id)]),
            ]

            if needs_datum:
                if datum_system_id is None and datum_labels:
                    datum_system_id = self._find_or_create_datum_system(datum_labels)
                parts.append((
                    'GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE',
                    [[Ref(datum_system_id)]],
                ))

            if needs_modifier:
                parts.append((
                    'GEOMETRIC_TOLERANCE_WITH_MODIFIERS',
                    [[EnumValue(m) for m in modifiers]],
                ))

            parts.append((entity_type, []))
            return self.editor.add_complex_entity(parts)
        else:
            # Simple tolerance entity
            return self.editor.add_entity(entity_type, [
                name, '', Ref(val_id), Ref(sa_id),
            ])

    def add_flatness_tolerance(self, face_id: int, value: float,
                               name: str = '') -> int:
        """Shortcut: add a flatness tolerance to a single face."""
        return self.add_tolerance('FLATNESS', [face_id], value, name)

    def add_perpendicularity_tolerance(self, face_ids: List[int], value: float,
                                       datum_labels: List[str],
                                       name: str = '',
                                       modifiers: Optional[List[str]] = None) -> int:
        """Shortcut: add a perpendicularity tolerance."""
        return self.add_tolerance(
            'PERPENDICULARITY', face_ids, value, name,
            datum_labels=datum_labels, modifiers=modifiers,
        )

    def add_position_tolerance(self, face_ids: List[int], value: float,
                               datum_labels: List[str],
                               name: str = '',
                               modifiers: Optional[List[str]] = None) -> int:
        """Shortcut: add a position tolerance."""
        return self.add_tolerance(
            'POSITION', face_ids, value, name,
            datum_labels=datum_labels, modifiers=modifiers,
        )

    # ------------------------------------------------------------------
    # Add dimensions
    # ------------------------------------------------------------------

    def add_dimension(self, face_ids: List[int], nominal: float,
                      upper: Optional[float] = None,
                      lower: Optional[float] = None,
                      name: str = 'diameter',
                      dim_type: str = 'DIMENSIONAL_SIZE') -> int:
        """Add a dimensional annotation to faces.

        Args:
            face_ids: ADVANCED_FACE IDs
            nominal: nominal dimension value
            upper: upper tolerance (positive)
            lower: lower tolerance (negative)
            name: semantic name ('diameter', 'distance', etc.)
            dim_type: DIMENSIONAL_SIZE or DIMENSIONAL_LOCATION

        Returns:
            Entity ID of the dimension.
        """
        # Create shape aspects and link to faces
        if len(face_ids) == 1:
            sa_id = self._add_shape_aspect(desc='DIMENSION')
            self._add_gisu(sa_id, face_ids[0], desc='DIMENSION')
        else:
            child_ids = []
            for fid in face_ids:
                child_sa = self._add_shape_aspect(desc='DIMENSION')
                self._add_gisu(child_sa, fid, desc='DIMENSION')
                child_ids.append(child_sa)
            sa_id = self._add_composite_shape_aspect(child_ids)

        # Create the dimension entity
        dim_id = self.editor.add_entity(dim_type, [Ref(sa_id), name])

        # Create nominal value representation
        nominal_measure_id = self._add_length_measure(nominal)
        unit_id = self._get_length_unit_id()

        # Find or create representation context
        rep_ctx_id = self._find_rep_context()

        sdr_id = self.editor.add_entity('SHAPE_DIMENSION_REPRESENTATION', [
            '', [Ref(nominal_measure_id)], Ref(rep_ctx_id),
        ])
        self.editor.add_entity('DIMENSIONAL_CHARACTERISTIC_REPRESENTATION', [
            Ref(dim_id), Ref(sdr_id),
        ])

        # Add plus/minus tolerance if specified
        if upper is not None and lower is not None:
            upper_id = self._add_length_measure(upper)
            lower_id = self._add_length_measure(lower)
            tv_id = self.editor.add_entity('TOLERANCE_VALUE', [
                Ref(lower_id), Ref(upper_id),
            ])
            self.editor.add_entity('PLUS_MINUS_TOLERANCE', [
                Ref(tv_id), Ref(dim_id),
            ])

        return dim_id

    # ------------------------------------------------------------------
    # Add datums
    # ------------------------------------------------------------------

    def add_datum(self, face_ids: List[int], label: str,
                  name: str = '') -> Tuple[int, int]:
        """Add a datum definition attached to faces.

        Args:
            face_ids: ADVANCED_FACE IDs that define this datum
            label: datum label (e.g. 'A', 'B', 'C')
            name: annotation name

        Returns:
            (datum_id, datum_feature_id)
        """
        pds_id = self._get_pds_id()
        brep_id = self._get_brep_rep_id()

        # Create DATUM
        datum_id = self.editor.add_entity('DATUM', [
            '', UNSET, Ref(pds_id), EnumValue('F'), label,
        ])

        # Create DATUM_FEATURE
        df_name = name or f'Datum Feature {label}'
        df_id = self.editor.add_entity('DATUM_FEATURE', [
            df_name, UNSET, Ref(pds_id), EnumValue('T'),
        ])

        # Link DATUM_FEATURE to DATUM via SHAPE_ASPECT_RELATIONSHIP
        self.editor.add_entity('SHAPE_ASPECT_RELATIONSHIP', [
            '', UNSET, Ref(df_id), Ref(datum_id),
        ])

        # Link DATUM_FEATURE to geometry via IIRU
        face_refs = [Ref(fid) for fid in face_ids]
        self.editor.add_entity('ITEM_IDENTIFIED_REPRESENTATION_USAGE', [
            '', UNSET, Ref(df_id), Ref(brep_id),
            TypedValue('SET_REPRESENTATION_ITEM', face_refs),
        ])

        return datum_id, df_id

    def add_datum_system(self, datum_labels: List[str],
                         name: str = '') -> int:
        """Create a DATUM_SYSTEM from existing datum labels.

        The datums must already exist in the file (created via add_datum).
        Labels should be in order: [primary, secondary, tertiary].

        Returns:
            Entity ID of the DATUM_SYSTEM.
        """
        pds_id = self._get_pds_id()

        # Find existing DATUM entities by label
        g = self.editor.graph
        datum_map = {}  # label → entity_id
        for eid in g.find_by_type('DATUM'):
            entity = self.sf[eid]
            if len(entity.params) >= 5 and isinstance(entity.params[4], str):
                datum_map[entity.params[4]] = eid

        # Create compartments
        compartment_ids = []
        for label in datum_labels:
            if label not in datum_map:
                raise ValueError(f"Datum '{label}' not found in file. "
                                 f"Available: {list(datum_map.keys())}")
            comp_id = self.editor.add_entity('DATUM_REFERENCE_COMPARTMENT', [
                '', UNSET, Ref(pds_id), EnumValue('F'),
                Ref(datum_map[label]), UNSET,
            ])
            compartment_ids.append(comp_id)

        # Create DATUM_SYSTEM
        ds_name = name or f'Datum System [{"|".join(datum_labels)}]'
        ds_id = self.editor.add_entity('DATUM_SYSTEM', [
            ds_name, UNSET, Ref(pds_id), EnumValue('F'),
            [Ref(cid) for cid in compartment_ids],
        ])

        return ds_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_or_create_datum_system(self, labels: List[str]) -> int:
        """Find an existing datum system with these labels, or create one."""
        g = self.editor.graph
        for eid in g.find_by_type('DATUM_SYSTEM'):
            existing_labels = PMIExtractor(g)._resolve_datum_system_labels(eid)
            if existing_labels == labels:
                return eid
        return self.add_datum_system(labels)

    def _find_rep_context(self) -> int:
        """Find a REPRESENTATION_CONTEXT for dimension values."""
        g = self.editor.graph
        # Try to find existing one from any SHAPE_DIMENSION_REPRESENTATION
        for eid in g.find_by_type('SHAPE_DIMENSION_REPRESENTATION'):
            entity = self.sf[eid]
            if len(entity.params) >= 3 and isinstance(entity.params[2], Ref):
                return entity.params[2].entity_id
        # Try global rep context
        for eid in g.find_by_type('GLOBAL_UNIT_ASSIGNED_CONTEXT'):
            return eid
        # Fallback: any representation context
        for eid in g.find_by_type('REPRESENTATION_CONTEXT'):
            return eid
        raise ValueError("No REPRESENTATION_CONTEXT found in file")


# ======================================================================
# Convenience functions
# ======================================================================

def extract_pmi(step_file: StepFile,
                graph: Optional[ReferenceGraph] = None) -> PMIData:
    """Extract all PMI from a STEP file."""
    if graph is None:
        graph = ReferenceGraph(step_file)
    return PMIExtractor(graph).extract()


def tolerance_category(tolerance_type: str) -> str:
    """Get the GD&T category for a tolerance type.

    Returns: 'form', 'orientation', 'location', 'profile', or 'runout'.
    """
    key = tolerance_type.upper()
    if not key.endswith('_TOLERANCE'):
        key += '_TOLERANCE'
    return _TOLERANCE_CATEGORIES.get(key, 'unknown')


# ======================================================================
# PMI Vector Encoding (for GNN node features)
# ======================================================================

# Ordered tolerance types for multi-hot encoding (indices 2..15)
_TOLERANCE_TYPE_ORDER = [
    'FLATNESS', 'STRAIGHTNESS', 'ROUNDNESS', 'CYLINDRICITY',
    'PERPENDICULARITY', 'PARALLELISM', 'ANGULARITY',
    'POSITION', 'CONCENTRICITY', 'SYMMETRY', 'COAXIALITY',
    'SURFACE_PROFILE', 'LINE_PROFILE',
    'CIRCULAR_RUNOUT', 'TOTAL_RUNOUT',
]
_TOLERANCE_TYPE_TO_IDX = {t: i for i, t in enumerate(_TOLERANCE_TYPE_ORDER)}

# Dimension types for multi-hot encoding (indices 19..22)
_DIMENSION_TYPE_ORDER = [
    'DIMENSIONAL_SIZE', 'DIMENSIONAL_LOCATION',
    'ANGULAR_SIZE', 'ANGULAR_LOCATION',
]
_DIMENSION_TYPE_TO_IDX = {t: i for i, t in enumerate(_DIMENSION_TYPE_ORDER)}

PMI_VECTOR_DIMS = 30

PMI_VECTOR_LAYOUT = {
    0:  'has_pmi',
    1:  'has_tolerance',
    # 2..16: tolerance_type_multihot (15 dims)
    16: 'min_tolerance_value',
    17: 'num_tolerances',
    18: 'has_dimension',
    # 19..22: dimension_type_multihot (4 dims)
    23: 'nominal_value',
    24: 'has_datum',
    25: 'is_primary_datum',
    26: 'is_secondary_datum',
    27: 'is_tertiary_datum',
    28: 'datum_label_ordinal',
    29: 'num_annotations_total',
}


def encode_face_pmi(annotations: Dict,
                    datum_roles: Optional[Dict[str, int]] = None) -> List[float]:
    """Encode a face's PMI annotations as a fixed-size numeric vector.

    Args:
        annotations: output of PMIData.face_annotations(face_id)
            {'tolerances': [...], 'dimensions': [...], 'datums': [...]}
        datum_roles: optional {datum_label: role_index} where
            0=primary, 1=secondary, 2=tertiary. Built from datum_systems.

    Returns:
        List of 30 floats, ready for tensor conversion.
    """
    vec = [0.0] * PMI_VECTOR_DIMS

    tols = annotations.get('tolerances', [])
    dims = annotations.get('dimensions', [])
    dats = annotations.get('datums', [])

    total = len(tols) + len(dims) + len(dats)
    vec[0] = 1.0 if total > 0 else 0.0

    # Tolerances
    if tols:
        vec[1] = 1.0
        min_val = float('inf')
        for t in tols:
            ttype = t.tolerance_type
            idx = _TOLERANCE_TYPE_TO_IDX.get(ttype)
            if idx is not None:
                vec[2 + idx] = 1.0
            if t.value is not None:
                min_val = min(min_val, t.value)
        vec[16] = min_val if min_val != float('inf') else 0.0
        vec[17] = float(len(tols))

    # Dimensions
    if dims:
        vec[18] = 1.0
        first_nominal = None
        for d in dims:
            dtype = d.dimension_type
            idx = _DIMENSION_TYPE_TO_IDX.get(dtype)
            if idx is not None:
                vec[19 + idx] = 1.0
            if first_nominal is None and d.nominal_value is not None:
                first_nominal = d.nominal_value
        vec[23] = first_nominal if first_nominal is not None else 0.0

    # Datums
    if dats:
        vec[24] = 1.0
        d = dats[0]  # first datum on this face
        label = d.label.upper()
        if label and label.isalpha():
            vec[28] = float(ord(label) - ord('A') + 1)

        # Datum role in datum systems (primary/secondary/tertiary)
        if datum_roles:
            for dat in dats:
                role = datum_roles.get(dat.label)
                if role is not None and role <= 2:
                    vec[25 + role] = 1.0

    vec[29] = float(total)
    return vec


def build_datum_roles(pmi: PMIData) -> Dict[str, int]:
    """Build a map of datum_label → minimum role index across all datum systems.

    Role 0 = primary (first in system), 1 = secondary, 2 = tertiary.
    If a datum appears as primary in any system, it gets role 0.
    """
    roles: Dict[str, int] = {}
    for ds in pmi.datum_systems:
        for i, label in enumerate(ds.datum_labels):
            existing = roles.get(label)
            if existing is None or i < existing:
                roles[label] = i
    return roles
