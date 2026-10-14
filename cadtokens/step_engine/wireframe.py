"""
STEP Engine - Wireframe

Samples B-Rep edges (lines, circles, ellipses, B-splines, polylines) into
3D polylines without a CAD kernel, and finds anchor points on faces.
Used to draw part views for ballooned inspection reports.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .graph import BRepExtractor, ReferenceGraph
from .model import Entity, Ref, TypedValue

Point = Tuple[float, float, float]


# ======================================================================
# Small vector helpers
# ======================================================================

def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Point, b: Point) -> Point:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a: Point) -> Point:
    length = math.sqrt(_dot(a, a))
    return _mul(a, 1.0 / length) if length > 1e-15 else a


def _dist2(a: Point, b: Point) -> float:
    d = _sub(a, b)
    return _dot(d, d)


# ======================================================================
# Wireframe
# ======================================================================

class Wireframe:
    """Edge polylines and face anchor points of a STEP B-Rep."""

    def __init__(self, graph: ReferenceGraph, arc_step_deg: float = 10.0):
        self.g = graph
        self.sf = graph.step_file
        self.brep = BRepExtractor(graph)
        self.arc_step = math.radians(arc_step_deg)
        self._edges: Optional[Dict[int, List[Point]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edges(self) -> Dict[int, List[Point]]:
        """{EDGE_CURVE id: [points]} for every edge in the file."""
        if self._edges is None:
            self._edges = {}
            for eid in self.brep.get_edge_curves():
                try:
                    pts = self._sample_edge(self.sf[eid])
                except (TypeError, ValueError, IndexError, ZeroDivisionError):
                    pts = []
                if len(pts) >= 2:
                    self._edges[eid] = pts
        return self._edges

    def face_points(self, face_id: int) -> List[Point]:
        """Sampled boundary points of one face."""
        edges = self.edges()
        pts: List[Point] = []
        for ec in self.brep._face_edge_curves(face_id):
            pts.extend(edges.get(ec, []))
        return pts

    def anchor(self, face_ids: Sequence[int]) -> Optional[Point]:
        """A representative point for a feature: centroid of its first face's boundary."""
        for fid in face_ids:
            pts = self.face_points(fid)
            if pts:
                n = float(len(pts))
                return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n,
                        sum(p[2] for p in pts) / n)
        return None

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _ent(self, ref) -> Optional[Entity]:
        return self.sf.get(ref.entity_id) if isinstance(ref, Ref) else None

    def _point(self, ref) -> Optional[Point]:
        e = self._ent(ref)
        if e is None:
            return None
        if e.has_type('VERTEX_POINT'):
            return self._point(e.params[1])
        if e.has_type('CARTESIAN_POINT'):
            c = [float(v) for v in e.params[1]]
            return (c[0], c[1], c[2] if len(c) > 2 else 0.0)
        return None

    def _direction(self, ref, default: Point) -> Point:
        e = self._ent(ref)
        if e is None or not e.has_type('DIRECTION'):
            return default
        c = [float(v) for v in e.params[1]]
        return _norm((c[0], c[1], c[2] if len(c) > 2 else 0.0))

    def _placement(self, ref) -> Tuple[Point, Point, Point]:
        """(origin, x axis, y axis) of an AXIS2_PLACEMENT_3D."""
        e = self._ent(ref)
        origin = self._point(e.params[1]) if e is not None else (0.0, 0.0, 0.0)
        z = self._direction(e.params[2], (0.0, 0.0, 1.0)) if e is not None and len(e.params) > 2 else (0.0, 0.0, 1.0)
        x = self._direction(e.params[3], (1.0, 0.0, 0.0)) if e is not None and len(e.params) > 3 else (1.0, 0.0, 0.0)
        x = _norm(_sub(x, _mul(z, _dot(x, z))))            # orthogonalise
        if _dot(x, x) < 1e-12:
            x = _norm(_cross((0.0, 1.0, 0.0), z)) if abs(z[1]) < 0.9 else _norm(_cross((1.0, 0.0, 0.0), z))
        return origin or (0.0, 0.0, 0.0), x, _cross(z, x)

    # ------------------------------------------------------------------
    # Edge sampling
    # ------------------------------------------------------------------

    def _sample_edge(self, edge: Entity) -> List[Point]:
        # EDGE_CURVE(name, edge_start, edge_end, edge_geometry, same_sense)
        start = self._point(edge.params[1])
        end = self._point(edge.params[2])
        same_sense = not (len(edge.params) > 4 and getattr(edge.params[4], 'value', 'T') == 'F')
        curve = self._ent(edge.params[3])
        if start is None or end is None:
            return []
        if not same_sense:
            start, end = end, start
        pts = self._sample_curve(curve, start, end) if curve is not None else None
        if not pts:
            pts = [start, end]
        return pts

    def _sample_curve(self, curve: Entity, start: Point, end: Point) -> Optional[List[Point]]:
        # Wrappers that carry a 3D curve
        for wrapper in ('SURFACE_CURVE', 'SEAM_CURVE', 'INTERSECTION_CURVE', 'BOUNDED_SURFACE_CURVE'):
            part = curve.get_part(wrapper)
            if part is not None and len(part.params) > 1:
                inner = self._ent(part.params[1])
                return self._sample_curve(inner, start, end) if inner is not None else None

        if curve.has_type('LINE'):
            return [start, end]
        if curve.has_type('CIRCLE'):
            p = curve.get_part('CIRCLE')
            r = float(p.params[2])
            return self._sample_conic(self._placement(p.params[1]), r, r, start, end)
        if curve.has_type('ELLIPSE'):
            p = curve.get_part('ELLIPSE')
            return self._sample_conic(self._placement(p.params[1]),
                                      float(p.params[2]), float(p.params[3]), start, end)
        if curve.has_type('B_SPLINE_CURVE') or curve.has_type('B_SPLINE_CURVE_WITH_KNOTS'):
            pts = self._sample_bspline(curve)
            return _trim(pts, start, end) if pts else None
        if curve.has_type('POLYLINE'):
            pts = [self._point(r) for r in curve.params[1]]
            pts = [p for p in pts if p is not None]
            return _trim(pts, start, end) if len(pts) >= 2 else None
        if curve.has_type('TRIMMED_CURVE'):
            basis = self._ent(curve.params[1])
            return self._sample_curve(basis, start, end) if basis is not None else None
        return None

    def _sample_conic(self, placement, a: float, b: float,
                      start: Point, end: Point) -> List[Point]:
        origin, x, y = placement

        def angle(p: Point) -> float:
            d = _sub(p, origin)
            return math.atan2(_dot(d, y) / b, _dot(d, x) / a)

        t0, t1 = angle(start), angle(end)
        if _dist2(start, end) < 1e-12 * max(1.0, a * a):
            t1 = t0 + 2 * math.pi                      # closed edge: full conic
        while t1 <= t0:
            t1 += 2 * math.pi
        n = max(2, int(math.ceil((t1 - t0) / self.arc_step)))
        return [_add(origin, _add(_mul(x, a * math.cos(t)), _mul(y, b * math.sin(t))))
                for t in (t0 + (t1 - t0) * i / n for i in range(n + 1))]

    def _sample_bspline(self, curve: Entity) -> Optional[List[Point]]:
        bs = curve.get_part('B_SPLINE_CURVE')
        if bs is not None:
            degree, cp_refs = int(bs.params[0]), bs.params[1]
            knots_part = curve.get_part('B_SPLINE_CURVE_WITH_KNOTS')
            mults, knots = knots_part.params[0], knots_part.params[1]
        else:                                        # simple B_SPLINE_CURVE_WITH_KNOTS
            p = curve.parts[0].params
            degree, cp_refs, mults, knots = int(p[1]), p[2], p[6], p[7]
        cps = [self._point(r) for r in cp_refs]
        if any(c is None for c in cps):
            return None
        weights = None
        rational = curve.get_part('RATIONAL_B_SPLINE_CURVE')
        if rational is not None and rational.params:
            weights = [float(w) for w in rational.params[0]]

        knot_vector: List[float] = []
        for m, k in zip(mults, knots):
            knot_vector.extend([float(k)] * int(m))
        n = len(cps)
        if len(knot_vector) != n + degree + 1:
            return None
        u0, u1 = knot_vector[degree], knot_vector[n]
        samples = min(200, max(16, 8 * n))
        return [_de_boor(degree, knot_vector, cps, weights, u0 + (u1 - u0) * i / samples)
                for i in range(samples + 1)]


def _trim(pts: List[Point], start: Point, end: Point) -> List[Point]:
    """Cut a sampled curve to the stretch between the edge's vertices."""
    i0 = min(range(len(pts)), key=lambda i: _dist2(pts[i], start))
    i1 = min(range(len(pts)), key=lambda i: _dist2(pts[i], end))
    if i0 == i1:
        return pts                                    # closed curve
    seg = pts[i0:i1 + 1] if i0 < i1 else list(reversed(pts[i1:i0 + 1]))
    return [start] + seg[1:-1] + [end]


def _de_boor(p: int, t: List[float], cps: List[Point],
             weights: Optional[List[float]], u: float) -> Point:
    """Evaluate a (rational) B-spline at u."""
    n = len(cps)
    k = p
    while k < n - 1 and u >= t[k + 1]:
        k += 1
    w = weights or [1.0] * n
    d = [(cps[j][0] * w[j], cps[j][1] * w[j], cps[j][2] * w[j], w[j])
         for j in range(k - p, k + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = j + k - p
            denom = t[i + p - r + 1] - t[i]
            alpha = (u - t[i]) / denom if denom > 1e-15 else 0.0
            d[j] = tuple((1 - alpha) * d[j - 1][c] + alpha * d[j][c] for c in range(4))
    x, y, z, ww = d[p]
    return (x / ww, y / ww, z / ww) if abs(ww) > 1e-15 else (x, y, z)
