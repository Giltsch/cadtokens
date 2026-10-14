"""
STEP Engine - Editor

Provides edit operations on a StepFile: modify parameters, add/delete entities,
renumber, merge, transform geometry, extract subgraphs, and validate references.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

from .model import (
    StepFile, Entity, EntityPart, HeaderSection, StepParam,
    Ref, EnumValue, TypedValue, Unset, Derived,
)
from .graph import ReferenceGraph, _collect_refs


class StepEditor:
    """Edit operations on a StepFile.

    Maintains a lazy-built reference graph that auto-invalidates on changes.
    """

    def __init__(self, step_file: StepFile):
        self.sf = step_file
        self._graph: Optional[ReferenceGraph] = None

    @property
    def graph(self) -> ReferenceGraph:
        """Lazy-built reference graph (rebuilt after edits)."""
        if self._graph is None:
            self._graph = ReferenceGraph(self.sf)
        return self._graph

    def _invalidate(self):
        self._graph = None

    # ------------------------------------------------------------------
    # Parameter editing
    # ------------------------------------------------------------------

    def set_param(self, entity_id: int, param_index: int,
                  value: StepParam, part_index: int = 0):
        """Change a specific parameter of an entity."""
        entity = self.sf.entities[entity_id]
        entity.parts[part_index].params[param_index] = value
        self._invalidate()

    def get_param(self, entity_id: int, param_index: int,
                  part_index: int = 0) -> StepParam:
        """Read a specific parameter."""
        entity = self.sf.entities[entity_id]
        return entity.parts[part_index].params[param_index]

    # ------------------------------------------------------------------
    # Entity add/delete
    # ------------------------------------------------------------------

    def add_entity(self, type_name: str, params: List[StepParam],
                   entity_id: Optional[int] = None) -> int:
        """Add a new simple entity. Returns the assigned ID."""
        eid = entity_id if entity_id is not None else self.sf.next_id
        entity = Entity(
            entity_id=eid,
            parts=[EntityPart(type_name=type_name, params=params)]
        )
        self.sf.entities[eid] = entity
        self.sf._max_id = max(self.sf._max_id, eid)
        self._invalidate()
        return eid

    def add_complex_entity(self, parts: List[Tuple[str, List[StepParam]]],
                           entity_id: Optional[int] = None) -> int:
        """Add a new complex entity. parts = [(type_name, params), ...]."""
        eid = entity_id if entity_id is not None else self.sf.next_id
        entity = Entity(
            entity_id=eid,
            parts=[EntityPart(type_name=n, params=p) for n, p in parts]
        )
        self.sf.entities[eid] = entity
        self.sf._max_id = max(self.sf._max_id, eid)
        self._invalidate()
        return eid

    def delete_entity(self, entity_id: int, cascade: bool = False):
        """Delete an entity. If cascade=True, delete orphaned children too."""
        if entity_id not in self.sf.entities:
            return

        if cascade:
            # Find children that would become orphaned
            children = self.graph.references(entity_id)
            del self.sf.entities[entity_id]
            self._invalidate()
            for child_id in children:
                if child_id in self.sf.entities:
                    back = self.graph.referenced_by(child_id)
                    if not back:
                        self.delete_entity(child_id, cascade=True)
        else:
            del self.sf.entities[entity_id]
            self._invalidate()

    def delete_entities(self, entity_ids: Set[int]):
        """Delete multiple entities at once."""
        for eid in entity_ids:
            self.sf.entities.pop(eid, None)
        self._invalidate()

    # ------------------------------------------------------------------
    # Renumbering
    # ------------------------------------------------------------------

    def renumber(self, start: int = 1) -> Dict[int, int]:
        """Renumber all entities sequentially. Returns old->new ID map."""
        old_ids = sorted(self.sf.entities.keys())
        id_map = {old: start + i for i, old in enumerate(old_ids)}

        new_entities = {}
        for old_id, entity in self.sf.entities.items():
            new_id = id_map[old_id]
            new_parts = []
            for part in entity.parts:
                new_params = _remap_params(part.params, id_map)
                new_parts.append(EntityPart(
                    type_name=part.type_name, params=new_params
                ))
            new_entities[new_id] = Entity(entity_id=new_id, parts=new_parts)

        self.sf.entities = new_entities
        self.sf._max_id = start + len(old_ids) - 1 if old_ids else 0
        self._invalidate()
        return id_map

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, other: StepFile) -> Dict[int, int]:
        """Merge another StepFile into this one. Returns other's old->new ID map."""
        offset = self.sf._max_id
        id_map = {eid: eid + offset for eid in other.entities}

        for old_id, entity in other.entities.items():
            new_id = id_map[old_id]
            new_parts = []
            for part in entity.parts:
                new_params = _remap_params(part.params, id_map)
                new_parts.append(EntityPart(
                    type_name=part.type_name, params=new_params
                ))
            self.sf.entities[new_id] = Entity(
                entity_id=new_id, parts=new_parts
            )

        if id_map:
            self.sf._max_id = max(self.sf._max_id, max(id_map.values()))
        self._invalidate()
        return id_map

    # ------------------------------------------------------------------
    # Geometry transforms
    # ------------------------------------------------------------------

    def transform(self, matrix: List[List[float]]):
        """Apply a 4x4 affine transform to all geometry.

        Transforms CARTESIAN_POINTs (full affine) and DIRECTIONs (rotation only).
        """
        self.transform_points(matrix)
        self.transform_directions(matrix)

    def transform_points(self, matrix: List[List[float]]):
        """Apply 4x4 matrix to all CARTESIAN_POINT coordinates."""
        for eid in self.graph.find_by_type('CARTESIAN_POINT'):
            entity = self.sf.entities[eid]
            coords = entity.params[1] if len(entity.params) > 1 else None
            if isinstance(coords, list) and len(coords) >= 3:
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                nx = matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]*z + matrix[0][3]
                ny = matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]*z + matrix[1][3]
                nz = matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]*z + matrix[2][3]
                entity.parts[0].params[1] = [nx, ny, nz]

    def transform_directions(self, matrix: List[List[float]]):
        """Apply 3x3 rotation part of 4x4 matrix to all DIRECTION vectors."""
        for eid in self.graph.find_by_type('DIRECTION'):
            entity = self.sf.entities[eid]
            ratios = entity.params[1] if len(entity.params) > 1 else None
            if isinstance(ratios, list) and len(ratios) >= 3:
                dx, dy, dz = float(ratios[0]), float(ratios[1]), float(ratios[2])
                # Rotation only (upper-left 3x3)
                nx = matrix[0][0]*dx + matrix[0][1]*dy + matrix[0][2]*dz
                ny = matrix[1][0]*dx + matrix[1][1]*dy + matrix[1][2]*dz
                nz = matrix[2][0]*dx + matrix[2][1]*dy + matrix[2][2]*dz
                # Normalize
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 1e-15:
                    nx, ny, nz = nx/length, ny/length, nz/length
                entity.parts[0].params[1] = [nx, ny, nz]

    def translate(self, dx: float, dy: float, dz: float):
        """Translate all geometry by (dx, dy, dz)."""
        matrix = [
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1],
        ]
        self.transform_points(matrix)
        # Directions don't change under translation

    def scale(self, factor: float):
        """Uniform scale all geometry."""
        matrix = [
            [factor, 0, 0, 0],
            [0, factor, 0, 0],
            [0, 0, factor, 0],
            [0, 0, 0, 1],
        ]
        self.transform_points(matrix)
        # Directions don't change under uniform scale

    def scale_radii(self, factor: float):
        """Scale all radius parameters (CYLINDRICAL_SURFACE, CIRCLE, etc.)."""
        radius_types = [
            'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE', 'SPHERICAL_SURFACE',
            'TOROIDAL_SURFACE', 'CIRCLE',
        ]
        for type_name in radius_types:
            for eid in self.graph.find_by_type(type_name):
                entity = self.sf.entities[eid]
                params = entity.params
                # Radius is typically the last numeric parameter
                for i in range(len(params) - 1, -1, -1):
                    if isinstance(params[i], (int, float)):
                        entity.parts[0].params[i] = params[i] * factor
                        break

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def extract_subgraph(self, entity_ids: Set[int]) -> StepFile:
        """Extract entities and all their transitive refs into a new StepFile."""
        all_ids = set()
        for eid in entity_ids:
            all_ids |= self.graph.transitive_refs(eid)

        new_entities = {}
        for eid in sorted(all_ids):
            if eid in self.sf.entities:
                new_entities[eid] = self.sf.entities[eid]

        return StepFile(header=self.sf.header, entities=new_entities)

    def clone_subgraph(self, entity_ids: Set[int]) -> Tuple[StepFile, Dict[int, int]]:
        """Clone a subgraph with fresh IDs. Returns (new_file, old->new map)."""
        # Get all entities in the subgraph
        all_ids = set()
        for eid in entity_ids:
            all_ids |= self.graph.transitive_refs(eid)

        # Assign new IDs
        offset = self.sf._max_id
        id_map = {old: old + offset for old in sorted(all_ids)}

        new_entities = {}
        for old_id in sorted(all_ids):
            if old_id not in self.sf.entities:
                continue
            entity = self.sf.entities[old_id]
            new_id = id_map[old_id]
            new_parts = []
            for part in entity.parts:
                new_params = _remap_params(part.params, id_map)
                new_parts.append(EntityPart(
                    type_name=part.type_name, params=new_params
                ))
            new_entities[new_id] = Entity(entity_id=new_id, parts=new_parts)

        return StepFile(header=self.sf.header, entities=new_entities), id_map

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_refs(self) -> List[Tuple[int, int]]:
        """Find broken references. Returns [(entity_id, missing_ref_id), ...]."""
        broken = []
        for entity in self.sf:
            refs = set()
            for part in entity.parts:
                _collect_refs(part.params, refs)
            for ref_id in refs:
                if ref_id not in self.sf.entities:
                    broken.append((entity.entity_id, ref_id))
        return broken

    def remove_unreferenced(self, keep_types: Optional[Set[str]] = None) -> int:
        """Remove entities not referenced by anything (except specified types).

        Returns count of removed entities. Always keeps root structural entities.
        """
        keep = keep_types or {
            'APPLICATION_PROTOCOL_DEFINITION', 'APPLICATION_CONTEXT',
            'PRODUCT', 'PRODUCT_DEFINITION', 'SHAPE_DEFINITION_REPRESENTATION',
            'ADVANCED_BREP_SHAPE_REPRESENTATION', 'MANIFOLD_SOLID_BREP',
        }

        referenced = set()
        for entity in self.sf:
            refs = set()
            for part in entity.parts:
                _collect_refs(part.params, refs)
            referenced.update(refs)

        to_remove = set()
        for eid, entity in self.sf.entities.items():
            if eid not in referenced:
                # Check if it's a type we want to keep
                if not any(entity.has_type(t) for t in keep):
                    to_remove.add(eid)

        for eid in to_remove:
            del self.sf.entities[eid]

        if to_remove:
            self._invalidate()

        return len(to_remove)

    # ------------------------------------------------------------------
    # Search & replace
    # ------------------------------------------------------------------

    def find_entities(self, type_name: str,
                      param_filter: Optional[Dict[int, StepParam]] = None
                      ) -> List[int]:
        """Find entities by type and optional parameter filters.

        param_filter: {param_index: expected_value}
        """
        results = []
        for eid in self.graph.find_by_type(type_name):
            entity = self.sf.entities[eid]
            if param_filter:
                match = True
                for idx, expected in param_filter.items():
                    if idx >= len(entity.params):
                        match = False
                        break
                    if entity.params[idx] != expected:
                        match = False
                        break
                if not match:
                    continue
            results.append(eid)
        return results

    def replace_ref(self, old_ref: int, new_ref: int):
        """Replace all references to old_ref with new_ref across the file."""
        id_map = {old_ref: new_ref}
        for entity in self.sf:
            for part in entity.parts:
                part.params = _remap_params(part.params, id_map)
        self._invalidate()


# ======================================================================
# Helpers
# ======================================================================

def _remap_params(params: List[StepParam],
                  id_map: Dict[int, int]) -> List[StepParam]:
    """Recursively remap entity references using id_map."""
    result = []
    for p in params:
        if isinstance(p, Ref):
            result.append(Ref(id_map.get(p.entity_id, p.entity_id)))
        elif isinstance(p, list):
            result.append(_remap_params(p, id_map))
        elif isinstance(p, TypedValue):
            if isinstance(p.value, Ref):
                new_val = Ref(id_map.get(p.value.entity_id, p.value.entity_id))
                result.append(TypedValue(p.type_name, new_val))
            elif isinstance(p.value, list):
                result.append(TypedValue(p.type_name,
                                         _remap_params(p.value, id_map)))
            else:
                result.append(p)
        else:
            result.append(p)
    return result
