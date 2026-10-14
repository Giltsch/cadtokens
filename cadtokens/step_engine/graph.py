"""
STEP Engine - Reference Graph & B-Rep Extractor

Builds a bidirectional reference graph from a StepFile.
Provides traversal, querying, and B-Rep topology extraction
without any CAD kernel dependency.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .model import StepFile, Entity, Ref, TypedValue, StepParam


class ReferenceGraph:
    """Bidirectional entity reference graph.

    Provides O(1) lookup for forward refs, back refs, and type-based queries.
    Built once from a StepFile, then used for fast graph operations.
    """

    def __init__(self, step_file: StepFile):
        self.step_file = step_file
        self._refs: Dict[int, Set[int]] = {}       # entity -> referenced entities
        self._back_refs: Dict[int, Set[int]] = {}   # entity -> referencing entities
        self._type_index: Dict[str, List[int]] = {}  # type_name -> entity IDs
        self._build()

    def _build(self):
        for entity in self.step_file:
            eid = entity.entity_id
            self._refs[eid] = set()

            # Type index (all parts for complex entities)
            for part in entity.parts:
                if part.type_name not in self._type_index:
                    self._type_index[part.type_name] = []
                self._type_index[part.type_name].append(eid)

            # Collect references
            refs = set()
            for part in entity.parts:
                _collect_refs(part.params, refs)
            self._refs[eid] = refs

            for ref_id in refs:
                if ref_id not in self._back_refs:
                    self._back_refs[ref_id] = set()
                self._back_refs[ref_id].add(eid)

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def references(self, entity_id: int) -> Set[int]:
        """Entities directly referenced by entity_id."""
        return self._refs.get(entity_id, set())

    def referenced_by(self, entity_id: int) -> Set[int]:
        """Entities that directly reference entity_id."""
        return self._back_refs.get(entity_id, set())

    def find_by_type(self, type_name: str) -> List[int]:
        """Find all entity IDs with a given type (checks complex parts too)."""
        return list(self._type_index.get(type_name, []))

    def entity_count_by_type(self) -> Dict[str, int]:
        """Count entities per type."""
        return {k: len(v) for k, v in sorted(self._type_index.items())}

    # ------------------------------------------------------------------
    # Transitive traversal
    # ------------------------------------------------------------------

    def transitive_refs(self, entity_id: int) -> Set[int]:
        """All entities reachable forward from entity_id (full subgraph)."""
        visited = set()
        stack = [entity_id]
        while stack:
            eid = stack.pop()
            if eid in visited:
                continue
            visited.add(eid)
            stack.extend(self._refs.get(eid, set()))
        return visited

    def transitive_back_refs(self, entity_id: int) -> Set[int]:
        """All entities that eventually reference entity_id."""
        visited = set()
        stack = [entity_id]
        while stack:
            eid = stack.pop()
            if eid in visited:
                continue
            visited.add(eid)
            stack.extend(self._back_refs.get(eid, set()))
        return visited

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------

    def roots(self) -> List[int]:
        """Entities not referenced by any other entity (top-level)."""
        return [eid for eid in self.step_file.entities
                if not self._back_refs.get(eid)]

    def leaves(self) -> List[int]:
        """Entities that don't reference anything (terminal nodes)."""
        return [eid for eid in self.step_file.entities
                if not self._refs.get(eid)]

    def depth(self, entity_id: int) -> int:
        """Max reference depth from entity_id to a leaf."""
        visited = set()

        def _depth(eid):
            if eid in visited or eid not in self.step_file.entities:
                return 0
            visited.add(eid)
            children = self._refs.get(eid, set())
            if not children:
                return 0
            return 1 + max(_depth(c) for c in children)

        return _depth(entity_id)

    def connected_components(self) -> List[Set[int]]:
        """Find connected components (treating refs as undirected edges)."""
        visited = set()
        components = []

        for eid in self.step_file.entities:
            if eid in visited:
                continue
            component = set()
            stack = [eid]
            while stack:
                current = stack.pop()
                if current in visited or current not in self.step_file.entities:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(self._refs.get(current, set()))
                stack.extend(self._back_refs.get(current, set()))
            components.append(component)

        return components

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Quick statistics about the graph."""
        total_refs = sum(len(r) for r in self._refs.values())
        return {
            "entities": len(self.step_file.entities),
            "total_references": total_refs,
            "unique_types": len(self._type_index),
            "root_entities": len(self.roots()),
            "leaf_entities": len(self.leaves()),
            "components": len(self.connected_components()),
        }


class BRepExtractor:
    """Extract B-Rep topology from STEP entities without a CAD kernel.

    Walks the entity graph to extract faces, edges, vertices, surfaces,
    and adjacency — everything needed for GNN input or analysis.
    """

    def __init__(self, graph: ReferenceGraph):
        self.graph = graph
        self.sf = graph.step_file

    # ------------------------------------------------------------------
    # Topology extraction
    # ------------------------------------------------------------------

    def get_shells(self) -> List[int]:
        """Get all shell entity IDs (CLOSED_SHELL, OPEN_SHELL)."""
        return (self.graph.find_by_type('CLOSED_SHELL') +
                self.graph.find_by_type('OPEN_SHELL'))

    def get_faces(self) -> List[int]:
        """Get all ADVANCED_FACE entity IDs."""
        return self.graph.find_by_type('ADVANCED_FACE')

    def get_edge_curves(self) -> List[int]:
        """Get all EDGE_CURVE entity IDs."""
        return self.graph.find_by_type('EDGE_CURVE')

    def get_vertices(self) -> List[int]:
        """Get all VERTEX_POINT entity IDs."""
        return self.graph.find_by_type('VERTEX_POINT')

    # ------------------------------------------------------------------
    # Coordinate extraction
    # ------------------------------------------------------------------

    def get_points(self) -> Dict[int, Tuple[float, float, float]]:
        """Get all CARTESIAN_POINT coordinates as {entity_id: (x, y, z)}."""
        points = {}
        for eid in self.graph.find_by_type('CARTESIAN_POINT'):
            entity = self.sf[eid]
            coords = entity.params[1] if len(entity.params) > 1 else []
            if isinstance(coords, list) and len(coords) >= 2:
                x = float(coords[0]) if len(coords) > 0 else 0.0
                y = float(coords[1]) if len(coords) > 1 else 0.0
                z = float(coords[2]) if len(coords) > 2 else 0.0
                points[eid] = (x, y, z)
        return points

    def get_directions(self) -> Dict[int, Tuple[float, float, float]]:
        """Get all DIRECTION vectors as {entity_id: (dx, dy, dz)}."""
        dirs = {}
        for eid in self.graph.find_by_type('DIRECTION'):
            entity = self.sf[eid]
            ratios = entity.params[1] if len(entity.params) > 1 else []
            if isinstance(ratios, list) and len(ratios) >= 2:
                dx = float(ratios[0]) if len(ratios) > 0 else 0.0
                dy = float(ratios[1]) if len(ratios) > 1 else 0.0
                dz = float(ratios[2]) if len(ratios) > 2 else 0.0
                dirs[eid] = (dx, dy, dz)
        return dirs

    # ------------------------------------------------------------------
    # Face information
    # ------------------------------------------------------------------

    def get_face_surface(self, face_id: int) -> Optional[Entity]:
        """Get the surface entity of an ADVANCED_FACE.

        ADVANCED_FACE('name', (bounds...), #surface_ref, .T./.F.)
        """
        face = self.sf.get(face_id)
        if face is None:
            return None
        params = face.params
        # Surface ref is typically params[2]
        if len(params) >= 3 and isinstance(params[2], Ref):
            return self.sf.get(params[2].entity_id)
        return None

    def get_face_info(self) -> List[Dict]:
        """Extract info for all faces: id, surface type, placement."""
        result = []
        for fid in self.get_faces():
            surface = self.get_face_surface(fid)
            info = {
                "face_id": fid,
                "surface_type": surface.type_name if surface else "UNKNOWN",
                "surface_id": surface.entity_id if surface else None,
            }

            # Extract surface parameters (radius, etc.)
            if surface:
                for p in surface.params:
                    if isinstance(p, (int, float)):
                        info.setdefault("parameters", []).append(p)

            result.append(info)
        return result

    # ------------------------------------------------------------------
    # Adjacency
    # ------------------------------------------------------------------

    def get_face_adjacency(self) -> Dict[int, Set[int]]:
        """Compute face adjacency via shared EDGE_CURVEs.

        Two faces are adjacent if they share an edge (via ORIENTED_EDGE
        referencing the same EDGE_CURVE).
        """
        # Map each EDGE_CURVE to the faces that reference it
        edge_to_faces: Dict[int, Set[int]] = defaultdict(set)

        for face_id in self.get_faces():
            # Walk: ADVANCED_FACE -> FACE_BOUND -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE
            edge_curves = self._face_edge_curves(face_id)
            for ec_id in edge_curves:
                edge_to_faces[ec_id].add(face_id)

        # Build adjacency from shared edges
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for ec_id, faces in edge_to_faces.items():
            face_list = list(faces)
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    adjacency[face_list[i]].add(face_list[j])
                    adjacency[face_list[j]].add(face_list[i])

        return dict(adjacency)

    def _face_edge_curves(self, face_id: int) -> Set[int]:
        """Get all EDGE_CURVE IDs referenced by a face."""
        edge_curves = set()
        face = self.sf.get(face_id)
        if face is None:
            return edge_curves

        # params[1] is the list of FACE_BOUND refs
        bounds_list = face.params[1] if len(face.params) > 1 else []
        if not isinstance(bounds_list, list):
            bounds_list = [bounds_list]

        for bound_ref in bounds_list:
            if not isinstance(bound_ref, Ref):
                continue
            bound = self.sf.get(bound_ref.entity_id)
            if bound is None:
                continue

            # FACE_BOUND/FACE_OUTER_BOUND -> params[1] is EDGE_LOOP ref
            if len(bound.params) >= 2 and isinstance(bound.params[1], Ref):
                loop = self.sf.get(bound.params[1].entity_id)
            elif len(bound.params) >= 1 and isinstance(bound.params[0], Ref):
                loop = self.sf.get(bound.params[0].entity_id)
            else:
                continue

            if loop is None:
                continue

            # EDGE_LOOP -> params[1] is list of ORIENTED_EDGE refs
            orient_list = loop.params[1] if len(loop.params) > 1 else []
            if not isinstance(orient_list, list):
                orient_list = [orient_list]

            for oe_ref in orient_list:
                if not isinstance(oe_ref, Ref):
                    continue
                oe = self.sf.get(oe_ref.entity_id)
                if oe is None:
                    continue

                # ORIENTED_EDGE -> params[3] is EDGE_CURVE ref
                if len(oe.params) >= 4 and isinstance(oe.params[3], Ref):
                    edge_curves.add(oe.params[3].entity_id)

        return edge_curves

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def bounding_box(self) -> Optional[Dict[str, float]]:
        """Compute axis-aligned bounding box from all CARTESIAN_POINTs."""
        points = self.get_points()
        if not points:
            return None

        coords = list(points.values())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]

        return {
            "xmin": min(xs), "xmax": max(xs),
            "ymin": min(ys), "ymax": max(ys),
            "zmin": min(zs), "zmax": max(zs),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
            "depth": max(zs) - min(zs),
        }

    # ------------------------------------------------------------------
    # GNN-ready output
    # ------------------------------------------------------------------

    def to_graph_data(self, pmi=None) -> Dict:
        """Export B-Rep as a GNN-ready graph structure.

        Args:
            pmi: Optional PMIData instance. When provided, each node gets
                 structured PMI annotations and a fixed-size pmi_vector.

        Returns dict with:
            - node_ids: face entity IDs
            - node_features: per-face feature dict (with pmi if provided)
            - edge_index: adjacency in COO format [sources, targets]
            - edge_features: per-edge info (shared edge count)
            - pmi_vectors: (if pmi provided) [N_faces x 30] list of lists
            - pmi_encoding_dims: (if pmi provided) int, vector dimension
        """
        faces = self.get_faces()
        adjacency = self.get_face_adjacency()

        # Node features
        face_info = {f["face_id"]: f for f in self.get_face_info()}

        # Surface type mapping
        surface_types = sorted(set(
            face_info[fid].get("surface_type", "UNKNOWN") for fid in faces
        ))
        type_to_idx = {t: i for i, t in enumerate(surface_types)}

        node_features = []
        for fid in faces:
            info = face_info.get(fid, {})
            stype = info.get("surface_type", "UNKNOWN")
            # One-hot surface type
            onehot = [0] * len(surface_types)
            onehot[type_to_idx[stype]] = 1
            node_features.append({
                "face_id": fid,
                "surface_type": stype,
                "surface_type_onehot": onehot,
                "parameters": info.get("parameters", []),
            })

        # Edge index (COO format)
        face_idx = {fid: i for i, fid in enumerate(faces)}
        sources, targets = [], []
        for fid in faces:
            for neighbor in adjacency.get(fid, set()):
                if fid in face_idx and neighbor in face_idx:
                    sources.append(face_idx[fid])
                    targets.append(face_idx[neighbor])

        result = {
            "node_ids": faces,
            "node_features": node_features,
            "edge_index": [sources, targets],
            "num_nodes": len(faces),
            "num_edges": len(sources),
            "surface_types": surface_types,
        }

        # PMI enrichment
        if pmi is not None:
            from .pmi import (
                encode_face_pmi, build_datum_roles,
                tolerance_category, PMI_VECTOR_DIMS,
            )
            datum_roles = build_datum_roles(pmi)
            pmi_vectors = []

            for node in node_features:
                fid = node["face_id"]
                annotations = pmi.face_annotations(fid)

                # Structured PMI per node
                pmi_info = {"has_pmi": False, "tolerances": [], "dimensions": [], "datums": []}

                for t in annotations["tolerances"]:
                    pmi_info["tolerances"].append({
                        "type": t.tolerance_type,
                        "value": t.value,
                        "category": tolerance_category(t.tolerance_type),
                        "datum_labels": t.datum_labels,
                        "modifiers": t.modifiers,
                    })

                for d in annotations["dimensions"]:
                    pmi_info["dimensions"].append({
                        "type": d.dimension_type,
                        "semantic": d.semantic_name,
                        "nominal": d.nominal_value,
                        "upper": d.upper_tolerance,
                        "lower": d.lower_tolerance,
                    })

                for d in annotations["datums"]:
                    pmi_info["datums"].append({
                        "label": d.label,
                    })

                if pmi_info["tolerances"] or pmi_info["dimensions"] or pmi_info["datums"]:
                    pmi_info["has_pmi"] = True

                node["pmi"] = pmi_info

                # Fixed-size numeric vector
                vec = encode_face_pmi(annotations, datum_roles)
                node["pmi_vector"] = vec
                pmi_vectors.append(vec)

            result["pmi_vectors"] = pmi_vectors
            result["pmi_encoding_dims"] = PMI_VECTOR_DIMS
            result["pmi_summary"] = pmi.summary()

        return result


# ======================================================================
# Helpers
# ======================================================================

def _collect_refs(params: List[StepParam], refs: Set[int]):
    """Recursively collect all entity references from parameters."""
    for p in params:
        if isinstance(p, Ref):
            refs.add(p.entity_id)
        elif isinstance(p, list):
            _collect_refs(p, refs)
        elif isinstance(p, TypedValue):
            if isinstance(p.value, Ref):
                refs.add(p.value.entity_id)
            elif isinstance(p.value, list):
                _collect_refs(p.value, refs)
