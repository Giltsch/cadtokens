"""
STEP Engine - PMI (Product Manufacturing Information)

Extract, query, and write GD&T annotations (tolerances, dimensions, datums)
from/to STEP AP242 files. Resolves the full entity chain from PMI to geometry
(ADVANCED_FACE) using only graph traversal — no CAD kernel required.

Supports:
  - Geometric tolerances (flatness, perpendicularity, position, profile, etc.)
  - Dimensions (size, location, angular) with plus/minus tolerances
  - Datums and datum systems (labels, features, targets)
  - Full PMI-to-face resolution via SHAPE_ASPECT → GISU → ADVANCED_FACE
  - Writing new PMI and attaching it to existing faces
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .model import (
    StepFile, Entity, EntityPart, Ref, EnumValue, TypedValue,
    Unset, Derived, UNSET, DERIVED, StepParam,
)
from .graph import ReferenceGraph


# ======================================================================
# Data Structures
# ======================================================================

@dataclass
class ToleranceInfo:
    """A geometric tolerance (GD&T) annotation."""
    entity_id: int
    name: str
    tolerance_type: str             # FLATNESS, PERPENDICULARITY, POSITION, ...
    value: Optional[float] = None   # tolerance zone size
    face_ids: List[int] = field(default_factory=list)  # ADVANCED_FACE entity IDs
    datum_labels: List[str] = field(default_factory=list)  # ['A', 'B', 'C']
    datum_system_id: Optional[int] = None
    modifiers: List[str] = field(default_factory=list)  # MAXIMUM_MATERIAL_REQUIREMENT, etc.
    shape_aspect_id: Optional[int] = None
    value_entity_id: Optional[int] = None
    is_complex: bool = False        # True if multi-type entity


@dataclass
class DimensionInfo:
    """A dimensional annotation (size, location, angular).

    For DIMENSIONAL_SIZE: all faces in face_ids (single feature).
    For DIMENSIONAL_LOCATION: from_face_ids and to_face_ids are the two
    features whose distance is measured. face_ids = from + to combined.
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


@dataclass
class DatumInfo:
    """A datum reference."""
    entity_id: int                  # DATUM entity ID
    label: str                      # 'A', 'B', 'C', ...
    datum_feature_id: Optional[int] = None  # DATUM_FEATURE entity ID
    face_ids: List[int] = field(default_factory=list)


@dataclass
class DatumSystemInfo:
    """A datum reference frame (ordered set of datums)."""
    entity_id: int
    name: str
    datum_labels: List[str] = field(default_factory=list)  # ordered: [primary, secondary, ...]
    compartment_ids: List[int] = field(default_factory=list)


@dataclass
class PMIData:
    """All PMI extracted from a STEP file."""
    tolerances: List[ToleranceInfo] = field(default_factory=list)
    dimensions: List[DimensionInfo] = field(default_factory=list)
    datums: List[DatumInfo] = field(default_factory=list)
    datum_systems: List[DatumSystemInfo] = field(default_factory=list)

    @property
    def has_pmi(self) -> bool:
        return bool(self.tolerances or self.dimensions or self.datums)

    def face_annotations(self, face_id: int) -> Dict:
        """Get all PMI annotations for a specific face."""
        result = {
            'tolerances': [t for t in self.tolerances if face_id in t.face_ids],
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

        # Precompute lookup tables
        self._sa_to_faces: Optional[Dict[int, List[int]]] = None
        self._composite_children: Optional[Dict[int, List[int]]] = None
        self._datum_feature_to_datum: Optional[Dict[int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> PMIData:
        """Extract all PMI from the STEP file."""
        self._build_lookups()
        return PMIData(
            tolerances=self._extract_tolerances(),
            dimensions=self._extract_dimensions(),
            datums=self._extract_datums(),
            datum_systems=self._extract_datum_systems(),
        )

    # ------------------------------------------------------------------
    # Lookup table construction
    # ------------------------------------------------------------------

    def _build_lookups(self):
        """Build the linking tables: shape_aspect → faces, composite → children."""
        # SHAPE_ASPECT → ADVANCED_FACE via GEOMETRIC_ITEM_SPECIFIC_USAGE
        self._sa_to_faces = {}

        for eid in self.g.find_by_type('GEOMETRIC_ITEM_SPECIFIC_USAGE'):
            entity = self.sf[eid]
            # GISU(name, desc, definition=SHAPE_ASPECT, used_rep, identified_item=FACE)
            if len(entity.params) >= 5:
                sa_ref = entity.params[2]
                face_ref = entity.params[4]
                if isinstance(sa_ref, Ref) and isinstance(face_ref, Ref):
                    sa_id = sa_ref.entity_id
                    face_id = face_ref.entity_id
                    # Verify it's actually a face
                    if face_id in self.sf.entities:
                        face_entity = self.sf[face_id]
                        if face_entity.has_type('ADVANCED_FACE'):
                            self._sa_to_faces.setdefault(sa_id, []).append(face_id)

        # Also via ITEM_IDENTIFIED_REPRESENTATION_USAGE
        for eid in self.g.find_by_type('ITEM_IDENTIFIED_REPRESENTATION_USAGE'):
            entity = self.sf[eid]
            # IIRU(name, desc, definition=SHAPE_ASPECT, used_rep, identified_item)
            if len(entity.params) >= 5:
                sa_ref = entity.params[2]
                item = entity.params[4]
                if isinstance(sa_ref, Ref):
                    sa_id = sa_ref.entity_id
                    face_refs = self._extract_face_refs_from_item(item)
                    for fid in face_refs:
                        self._sa_to_faces.setdefault(sa_id, []).append(fid)

        # COMPOSITE_SHAPE_ASPECT → child SHAPE_ASPECTs via SHAPE_ASPECT_RELATIONSHIP
        self._composite_children = {}
        for eid in self.g.find_by_type('SHAPE_ASPECT_RELATIONSHIP'):
            entity = self.sf[eid]
            # SAR(name, desc, relating=parent, related=child)
            if len(entity.params) >= 4:
                parent = entity.params[2]
                child = entity.params[3]
                if isinstance(parent, Ref) and isinstance(child, Ref):
                    self._composite_children.setdefault(
                        parent.entity_id, []
                    ).append(child.entity_id)

        # DATUM_FEATURE → DATUM mapping via SHAPE_ASPECT_RELATIONSHIP
        self._datum_feature_to_datum = {}
        for eid in self.g.find_by_type('SHAPE_ASPECT_RELATIONSHIP'):
            entity = self.sf[eid]
            if len(entity.params) >= 4:
                p2 = entity.params[2]
                p3 = entity.params[3]
                if isinstance(p2, Ref) and isinstance(p3, Ref):
                    src = self.sf.entities.get(p2.entity_id)
                    tgt = self.sf.entities.get(p3.entity_id)
                    if src and tgt:
                        if src.has_type('DATUM_FEATURE') and tgt.has_type('DATUM'):
                            self._datum_feature_to_datum[p2.entity_id] = p3.entity_id

    def _extract_face_refs_from_item(self, item: StepParam) -> List[int]:
        """Extract ADVANCED_FACE refs from an identified_item parameter.

        Handles:
          - Direct Ref to a face
          - TypedValue like SET_REPRESENTATION_ITEM([#face1, #face2])
          - List of refs
        """
        face_ids = []
        if isinstance(item, Ref):
            if item.entity_id in self.sf.entities:
                e = self.sf[item.entity_id]
                if e.has_type('ADVANCED_FACE'):
                    face_ids.append(item.entity_id)
        elif isinstance(item, TypedValue):
            # SET_REPRESENTATION_ITEM([refs...])
            if isinstance(item.value, list):
                for v in item.value:
                    if isinstance(v, Ref) and v.entity_id in self.sf.entities:
                        e = self.sf[v.entity_id]
                        if e.has_type('ADVANCED_FACE'):
                            face_ids.append(v.entity_id)
            elif isinstance(item.value, Ref):
                face_ids.extend(self._extract_face_refs_from_item(item.value))
        elif isinstance(item, list):
            for v in item:
                face_ids.extend(self._extract_face_refs_from_item(v))
        return face_ids

    # ------------------------------------------------------------------
    # Resolve faces for a shape_aspect
    # ------------------------------------------------------------------

    def _resolve_faces(self, shape_aspect_id: int) -> List[int]:
        """Resolve a SHAPE_ASPECT (possibly composite) to face IDs."""
        # Direct lookup
        faces = list(self._sa_to_faces.get(shape_aspect_id, []))

        # If composite, resolve children recursively
        children = self._composite_children.get(shape_aspect_id, [])
        for child_id in children:
            faces.extend(self._resolve_faces(child_id))

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for f in faces:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    # ------------------------------------------------------------------
    # Extract tolerance value
    # ------------------------------------------------------------------

    def _extract_measure_value(self, entity_id: int) -> Optional[float]:
        """Extract a numeric value from a MEASURE_WITH_UNIT or similar entity."""
        if entity_id not in self.sf.entities:
            return None
        entity = self.sf[entity_id]

        # Try MEASURE_WITH_UNIT part (complex entities)
        mwu = entity.get_part('MEASURE_WITH_UNIT')
        if mwu and mwu.params:
            val = mwu.params[0]
            if isinstance(val, TypedValue):
                if isinstance(val.value, (int, float)):
                    return float(val.value)
            elif isinstance(val, (int, float)):
                return float(val)

        # Try LENGTH_MEASURE_WITH_UNIT (simple)
        lmwu = entity.get_part('LENGTH_MEASURE_WITH_UNIT')
        if lmwu and lmwu.params:
            val = lmwu.params[0]
            if isinstance(val, TypedValue):
                if isinstance(val.value, (int, float)):
                    return float(val.value)

        # Fallback: first TypedValue or numeric in any part
        for part in entity.parts:
            for p in part.params:
                if isinstance(p, TypedValue) and isinstance(p.value, (int, float)):
                    return float(p.value)
                if isinstance(p, (int, float)):
                    return float(p)

        return None

    # ------------------------------------------------------------------
    # Extract geometric tolerances
    # ------------------------------------------------------------------

    def _extract_tolerances(self) -> List[ToleranceInfo]:
        tolerances = []
        seen_ids = set()

        # 1. Simple standalone tolerances (FLATNESS_TOLERANCE, etc.)
        for ttype in _ALL_TOLERANCE_SUBTYPES:
            for eid in self.g.find_by_type(ttype):
                if eid in seen_ids:
                    continue
                entity = self.sf[eid]

                if entity.is_complex:
                    # Complex entity — extract from GEOMETRIC_TOLERANCE part
                    info = self._extract_complex_tolerance(entity)
                else:
                    # Simple tolerance: (name, desc, value_ref, shape_aspect_ref)
                    info = self._extract_simple_tolerance(entity, ttype)

                if info:
                    seen_ids.add(eid)
                    tolerances.append(info)

        # 2. Complex entities found via GEOMETRIC_TOLERANCE type
        for eid in self.g.find_by_type('GEOMETRIC_TOLERANCE'):
            if eid in seen_ids:
                continue
            entity = self.sf[eid]
            if entity.is_complex:
                info = self._extract_complex_tolerance(entity)
                if info:
                    seen_ids.add(eid)
                    tolerances.append(info)

        return tolerances

    def _extract_simple_tolerance(self, entity: Entity,
                                  ttype: str) -> Optional[ToleranceInfo]:
        """Extract info from a simple (non-complex) tolerance entity."""
        params = entity.params
        if len(params) < 4:
            return None

        name = params[0] if isinstance(params[0], str) else ''
        value_ref = params[2] if isinstance(params[2], Ref) else None
        sa_ref = params[3] if isinstance(params[3], Ref) else None

        value = self._extract_measure_value(value_ref.entity_id) if value_ref else None
        faces = self._resolve_faces(sa_ref.entity_id) if sa_ref else []
        short_type = ttype.replace('_TOLERANCE', '')

        return ToleranceInfo(
            entity_id=entity.entity_id,
            name=name,
            tolerance_type=short_type,
            value=value,
            face_ids=faces,
            shape_aspect_id=sa_ref.entity_id if sa_ref else None,
            value_entity_id=value_ref.entity_id if value_ref else None,
            is_complex=False,
        )

    def _extract_complex_tolerance(self, entity: Entity) -> Optional[ToleranceInfo]:
        """Extract info from a complex (multi-type) tolerance entity."""
        # Find the GEOMETRIC_TOLERANCE part (has the core params)
        gt_part = entity.get_part('GEOMETRIC_TOLERANCE')
        if not gt_part or len(gt_part.params) < 4:
            return None

        name = gt_part.params[0] if isinstance(gt_part.params[0], str) else ''
        value_ref = gt_part.params[2] if isinstance(gt_part.params[2], Ref) else None
        sa_ref = gt_part.params[3] if isinstance(gt_part.params[3], Ref) else None

        # Determine the specific tolerance subtype
        ttype = 'GEOMETRIC'
        for type_name in entity.type_names:
            if type_name in _ALL_TOLERANCE_SUBTYPES:
                ttype = type_name.replace('_TOLERANCE', '')
                break

        # Extract datum references
        datum_labels = []
        datum_system_id = None
        gtdr = entity.get_part('GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE')
        if gtdr and gtdr.params:
            # param[0] is list of datum_system refs
            ds_refs = gtdr.params[0] if isinstance(gtdr.params[0], list) else []
            for ds_ref in ds_refs:
                if isinstance(ds_ref, Ref):
                    datum_system_id = ds_ref.entity_id
                    datum_labels = self._resolve_datum_system_labels(ds_ref.entity_id)
                    break

        # Extract modifiers
        modifiers = []
        gtwm = entity.get_part('GEOMETRIC_TOLERANCE_WITH_MODIFIERS')
        if gtwm and gtwm.params:
            mod_list = gtwm.params[0] if isinstance(gtwm.params[0], list) else []
            for m in mod_list:
                if isinstance(m, EnumValue):
                    modifiers.append(m.value)

        value = self._extract_measure_value(value_ref.entity_id) if value_ref else None
        faces = self._resolve_faces(sa_ref.entity_id) if sa_ref else []

        return ToleranceInfo(
            entity_id=entity.entity_id,
            name=name,
            tolerance_type=ttype,
            value=value,
            face_ids=faces,
            datum_labels=datum_labels,
            datum_system_id=datum_system_id,
            modifiers=modifiers,
            shape_aspect_id=sa_ref.entity_id if sa_ref else None,
            value_entity_id=value_ref.entity_id if value_ref else None,
            is_complex=True,
        )

    # ------------------------------------------------------------------
    # Extract dimensions
    # ------------------------------------------------------------------

    def _extract_dimensions(self) -> List[DimensionInfo]:
        dimensions = []

        dim_types = [
            'DIMENSIONAL_SIZE', 'DIMENSIONAL_LOCATION',
            'ANGULAR_SIZE', 'ANGULAR_LOCATION',
            'DIRECTED_DIMENSIONAL_LOCATION',
            'DIMENSIONAL_SIZE_WITH_DATUM_FEATURE',
        ]

        for dtype in dim_types:
            for eid in self.g.find_by_type(dtype):
                entity = self.sf[eid]
                info = self._extract_dimension(entity, dtype)
                if info:
                    dimensions.append(info)

        return dimensions

    def _extract_dimension(self, entity: Entity,
                           dtype: str) -> Optional[DimensionInfo]:
        """Extract info from a dimensional entity."""
        params = entity.params
        if not params:
            return None

        # Two layouts:
        #   DIMENSIONAL_SIZE(shape_aspect_ref, name_string)
        #   DIMENSIONAL_LOCATION(name, desc, shape_aspect_ref1, shape_aspect_ref2)
        is_location = dtype in ('DIMENSIONAL_LOCATION', 'DIRECTED_DIMENSIONAL_LOCATION',
                                'ANGULAR_LOCATION')

        sa_ref = None
        sa_ref_2 = None
        semantic = ''

        if is_location:
            # DIMENSIONAL_LOCATION(name, desc, sa_ref1, sa_ref2)
            if len(params) >= 1 and isinstance(params[0], str):
                semantic = params[0]
            if len(params) >= 3 and isinstance(params[2], Ref):
                sa_ref = params[2]
            if len(params) >= 4 and isinstance(params[3], Ref):
                sa_ref_2 = params[3]
        else:
            # DIMENSIONAL_SIZE(sa_ref, name) / ANGULAR_SIZE(sa_ref, name)
            if isinstance(params[0], Ref):
                sa_ref = params[0]
            for p in params:
                if isinstance(p, str) and p:
                    semantic = p
                    break

        from_faces: List[int] = []
        to_faces: List[int] = []

        if sa_ref:
            from_faces = self._resolve_faces(sa_ref.entity_id)

        if sa_ref_2:
            to_faces = self._resolve_faces(sa_ref_2.entity_id)

        # face_ids = union of from + to, deduplicated
        seen: set = set()
        faces: List[int] = []
        for f in from_faces + to_faces:
            if f not in seen:
                seen.add(f)
                faces.append(f)

        # Find PLUS_MINUS_TOLERANCE referencing this dimension
        nominal = None
        upper = None
        lower = None
        pmt_id = None

        for pm_eid in self.g.find_by_type('PLUS_MINUS_TOLERANCE'):
            pm = self.sf[pm_eid]
            # PLUS_MINUS_TOLERANCE(tolerance_value_ref, dimension_ref)
            if len(pm.params) >= 2 and isinstance(pm.params[1], Ref):
                if pm.params[1].entity_id == entity.entity_id:
                    pmt_id = pm_eid
                    # Extract upper/lower from TOLERANCE_VALUE
                    if isinstance(pm.params[0], Ref):
                        tv = self.sf.entities.get(pm.params[0].entity_id)
                        if tv and tv.has_type('TOLERANCE_VALUE') and len(tv.params) >= 2:
                            if isinstance(tv.params[0], Ref):
                                lower = self._extract_measure_value(tv.params[0].entity_id)
                            if isinstance(tv.params[1], Ref):
                                upper = self._extract_measure_value(tv.params[1].entity_id)
                    break

        # Find nominal value via DIMENSIONAL_CHARACTERISTIC_REPRESENTATION
        for dcr_eid in self.g.find_by_type('DIMENSIONAL_CHARACTERISTIC_REPRESENTATION'):
            dcr = self.sf[dcr_eid]
            # DCR(dimension_ref, shape_dimension_representation_ref)
            if len(dcr.params) >= 2 and isinstance(dcr.params[0], Ref):
                if dcr.params[0].entity_id == entity.entity_id:
                    if isinstance(dcr.params[1], Ref):
                        sdr = self.sf.entities.get(dcr.params[1].entity_id)
                        if sdr and len(sdr.params) >= 2:
                            measures = sdr.params[1]
                            if isinstance(measures, list):
                                for m_ref in measures:
                                    if isinstance(m_ref, Ref):
                                        val = self._extract_measure_value(m_ref.entity_id)
                                        if val is not None:
                                            nominal = val
                                            break
                    break

        return DimensionInfo(
            entity_id=entity.entity_id,
            name=entity.type_name,
            dimension_type=dtype,
            semantic_name=semantic,
            nominal_value=nominal,
            upper_tolerance=upper,
            lower_tolerance=lower,
            face_ids=faces,
            from_face_ids=from_faces,
            to_face_ids=to_faces,
            shape_aspect_id=sa_ref.entity_id if sa_ref else None,
            shape_aspect_id_2=sa_ref_2.entity_id if sa_ref_2 else None,
            plus_minus_id=pmt_id,
        )

    # ------------------------------------------------------------------
    # Extract datums
    # ------------------------------------------------------------------

    def _extract_datums(self) -> List[DatumInfo]:
        datums = []

        for eid in self.g.find_by_type('DATUM'):
            entity = self.sf[eid]
            # DATUM(name, desc, product_def_shape, product_definitional, label)
            label = ''
            if len(entity.params) >= 5:
                label = entity.params[4] if isinstance(entity.params[4], str) else ''

            # Find the DATUM_FEATURE linked to this datum
            df_id = None
            faces = []
            for df_eid, datum_eid in self._datum_feature_to_datum.items():
                if datum_eid == eid:
                    df_id = df_eid
                    faces = self._resolve_faces(df_id)
                    break

            datums.append(DatumInfo(
                entity_id=eid,
                label=label,
                datum_feature_id=df_id,
                face_ids=faces,
            ))

        return datums

    def _extract_datum_systems(self) -> List[DatumSystemInfo]:
        systems = []

        for eid in self.g.find_by_type('DATUM_SYSTEM'):
            entity = self.sf[eid]
            name = entity.params[0] if isinstance(entity.params[0], str) else ''

            labels = self._resolve_datum_system_labels(eid)
            compartments = []
            if len(entity.params) >= 5 and isinstance(entity.params[4], list):
                for c_ref in entity.params[4]:
                    if isinstance(c_ref, Ref):
                        compartments.append(c_ref.entity_id)

            systems.append(DatumSystemInfo(
                entity_id=eid,
                name=name,
                datum_labels=labels,
                compartment_ids=compartments,
            ))

        return systems

    def _resolve_datum_system_labels(self, ds_id: int) -> List[str]:
        """Resolve a DATUM_SYSTEM entity to ordered datum labels ['A','B','C']."""
        labels = []
        entity = self.sf.entities.get(ds_id)
        if not entity or not entity.has_type('DATUM_SYSTEM'):
            return labels

        # param[4] = list of DATUM_REFERENCE_COMPARTMENT refs
        if len(entity.params) >= 5 and isinstance(entity.params[4], list):
            for comp_ref in entity.params[4]:
                if isinstance(comp_ref, Ref):
                    comp = self.sf.entities.get(comp_ref.entity_id)
                    if comp and len(comp.params) >= 5:
                        datum_ref = comp.params[4]
                        if isinstance(datum_ref, Ref):
                            datum = self.sf.entities.get(datum_ref.entity_id)
                            if datum and datum.has_type('DATUM'):
                                if len(datum.params) >= 5:
                                    label = datum.params[4]
                                    if isinstance(label, str):
                                        labels.append(label)
        return labels


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
