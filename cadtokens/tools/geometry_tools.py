#!/usr/bin/env python3
"""
Geometry Tools - Phase 0.2

Tool-Set für LLM/Agent um STEP-Dateien zu analysieren.
Diese Tools werden vom ReAct Agent aufgerufen.

Usage:
    from cadtokens.tools.geometry_tools import GeometryTools
    tools = GeometryTools()
    bbox = tools.get_bounding_box("part.step")
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import math

# PythonOCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
)
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape


class GeometryTools:
    """
    Tools für LLM/Agent um STEP-Dateien zu analysieren.
    
    Jede Methode ist so designed, dass sie vom LLM aufgerufen werden kann
    und ein JSON-serialisierbares Ergebnis zurückgibt.
    """
    
    def __init__(self):
        self._cache: Dict[str, TopoDS_Shape] = {}
        self._face_cache: Dict[str, List] = {}
        self._current_step_path: Optional[str] = None
    
    def load_step(self, step_path: str) -> bool:
        """Public Methode zum Laden einer STEP-Datei"""
        try:
            self._load_step(step_path)
            self._current_step_path = step_path
            return True
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            return False
    
    def _load_step(self, step_path: str) -> TopoDS_Shape:
        """Lädt STEP-Datei (mit Caching)"""
        if step_path in self._cache:
            return self._cache[step_path]
        
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != 1:
            raise ValueError(f"Konnte STEP nicht laden: {step_path}")
        
        reader.TransferRoots()
        shape = reader.OneShape()
        self._cache[step_path] = shape
        return shape
    
    def _get_faces(self, step_path: str) -> List:
        """Gibt alle Faces als Liste zurück (mit Caching)"""
        if step_path in self._face_cache:
            return self._face_cache[step_path]
        
        shape = self._load_step(step_path)
        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            faces.append(topods.Face(explorer.Current()))
            explorer.Next()
        
        self._face_cache[step_path] = faces
        return faces
    
    def _get_edges(self, step_path: str) -> List:
        """Gibt alle Edges als Liste zurück"""
        shape = self._load_step(step_path)
        edges = []
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edges.append(topods.Edge(explorer.Current()))
            explorer.Next()
        return edges
    
    def _get_surface_type(self, face) -> str:
        """Gibt Surface-Typ einer Face zurück"""
        adaptor = BRepAdaptor_Surface(face)
        stype = adaptor.GetType()
        type_map = {
            GeomAbs_Plane: "Plane",
            GeomAbs_Cylinder: "Cylinder",
            GeomAbs_Cone: "Cone",
            GeomAbs_Sphere: "Sphere",
            GeomAbs_Torus: "Torus",
            GeomAbs_BSplineSurface: "BSpline",
        }
        return type_map.get(stype, "Other")
    
    def _get_face_normal(self, face) -> Tuple[float, float, float]:
        """Berechnet die Normale einer Face am Zentrum"""
        try:
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            
            # Face Center finden
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            center = props.CentreOfMass()
            
            # Normale am Center
            adaptor = BRepAdaptor_Surface(face)
            u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
            
            pnt = gp_Pnt()
            d1u = gp_Vec()
            d1v = gp_Vec()
            adaptor.D1(u_mid, v_mid, pnt, d1u, d1v)
            
            normal = d1u.Crossed(d1v)
            if normal.Magnitude() > 1e-10:
                normal.Normalize()
                return (round(normal.X(), 4), round(normal.Y(), 4), round(normal.Z(), 4))
        except:
            pass
        return (0, 0, 0)
    
    def _get_face_centroid(self, face) -> Tuple[float, float, float]:
        """Berechnet den Schwerpunkt einer Face"""
        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            center = props.CentreOfMass()
            return (round(center.X(), 4), round(center.Y(), 4), round(center.Z(), 4))
        except:
            return (0, 0, 0)
    
    # =========================================================================
    # PUBLIC TOOLS (für LLM)
    # =========================================================================
    
    def get_bounding_box(self, step_path: str) -> Dict[str, float]:
        """
        Gibt die Bounding Box des Bauteils zurück.
        
        Args:
            step_path: Pfad zur STEP-Datei
            
        Returns:
            Dict mit width, height, depth, xmin, xmax, ymin, ymax, zmin, zmax
        """
        shape = self._load_step(step_path)
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        return {
            "width": round(xmax - xmin, 3),
            "height": round(ymax - ymin, 3),
            "depth": round(zmax - zmin, 3),
            "xmin": round(xmin, 3),
            "xmax": round(xmax, 3),
            "ymin": round(ymin, 3),
            "ymax": round(ymax, 3),
            "zmin": round(zmin, 3),
            "zmax": round(zmax, 3)
        }
    
    def get_volume_and_area(self, step_path: str) -> Dict[str, float]:
        """
        Gibt Volumen und Oberfläche des Bauteils zurück.
        
        Args:
            step_path: Pfad zur STEP-Datei
            
        Returns:
            Dict mit volume und surface_area
        """
        shape = self._load_step(step_path)
        
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume = props.Mass()
        
        brepgprop.SurfaceProperties(shape, props)
        surface_area = props.Mass()
        
        return {
            "volume": round(volume, 3),
            "surface_area": round(surface_area, 3)
        }
    
    def count_topology(self, step_path: str) -> Dict[str, int]:
        """
        Zählt topologische Elemente (Faces, Edges, Vertices).
        
        Args:
            step_path: Pfad zur STEP-Datei
            
        Returns:
            Dict mit faces, edges, vertices
        """
        shape = self._load_step(step_path)
        
        counts = {"faces": 0, "edges": 0, "vertices": 0}
        
        for topo_type, name in [(TopAbs_FACE, "faces"), 
                                 (TopAbs_EDGE, "edges"), 
                                 (TopAbs_VERTEX, "vertices")]:
            explorer = TopExp_Explorer(shape, topo_type)
            while explorer.More():
                counts[name] += 1
                explorer.Next()
        
        return counts
    
    def count_features(self, step_path: str, feature_type: Optional[str] = None) -> Dict[str, int]:
        """
        Zählt Features nach Surface-Typ.
        
        Args:
            step_path: Pfad zur STEP-Datei
            feature_type: Optional - nur diesen Typ zählen (Plane, Cylinder, etc.)
            
        Returns:
            Dict mit Anzahl pro Feature-Typ
        """
        faces = self._get_faces(step_path)
        
        counts = {}
        for face in faces:
            stype = self._get_surface_type(face)
            counts[stype] = counts.get(stype, 0) + 1
        
        if feature_type:
            return {feature_type: counts.get(feature_type, 0)}
        
        return counts
    
    def get_face_info(self, step_path: str, face_id: int) -> Dict[str, Any]:
        """
        Gibt detaillierte Informationen zu einer Face zurück.
        
        Args:
            step_path: Pfad zur STEP-Datei
            face_id: Index der Face (0-basiert)
            
        Returns:
            Dict mit type, area, centroid
        """
        faces = self._get_faces(step_path)
        
        if face_id < 0 or face_id >= len(faces):
            raise ValueError(f"Face ID {face_id} nicht gültig. Bauteil hat {len(faces)} Faces.")
        
        face = faces[face_id]
        
        # Typ
        stype = self._get_surface_type(face)
        
        # Fläche und Schwerpunkt
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        centroid = props.CentreOfMass()
        
        # Parametrische Info (Radius, etc.)
        adaptor = BRepAdaptor_Surface(face)
        parametric = self._get_parametric_info(adaptor, stype)
        
        # Normal
        normal = self._get_face_normal(face)
        
        result = {
            "face_id": face_id,
            "type": stype,
            "area": round(area, 3),
            "centroid": {
                "x": round(centroid.X(), 3),
                "y": round(centroid.Y(), 3),
                "z": round(centroid.Z(), 3)
            },
            "normal": normal
        }
        
        # Parametrische Daten hinzufügen
        if parametric.get("radius") is not None:
            result["radius"] = parametric["radius"]
            result["diameter"] = round(parametric["radius"] * 2, 6)
        if parametric.get("radius2") is not None:
            result["radius2"] = parametric["radius2"]
        if parametric.get("semi_angle") is not None:
            result["semi_angle"] = parametric["semi_angle"]
        
        return result
    
    def list_faces(self, step_path: str, max_faces: int = 50) -> Dict[str, Any]:
        """
        Listet Faces mit Grundinformationen (limitiert für Performance).
        
        Args:
            step_path: Pfad zur STEP-Datei
            max_faces: Maximale Anzahl Faces (default 50)
            
        Returns:
            Dict mit total_count, faces (limitiert), und truncated flag
        """
        faces = self._get_faces(step_path)
        total = len(faces)
        result = []
        
        for i, face in enumerate(faces[:max_faces]):
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            stype = self._get_surface_type(face)
            
            entry = {
                "id": i,
                "type": stype,
                "area": round(props.Mass(), 3)
            }
            
            # Radius/Durchmesser für zylindrische/sphärische Faces
            if stype in ("Cylinder", "Sphere", "Cone", "Torus"):
                adaptor = BRepAdaptor_Surface(face)
                parametric = self._get_parametric_info(adaptor, stype)
                if parametric.get("radius") is not None:
                    entry["radius"] = parametric["radius"]
                    entry["diameter"] = round(parametric["radius"] * 2, 6)
            
            result.append(entry)
        
        return {
            "total_faces": total,
            "shown": len(result),
            "truncated": total > max_faces,
            "faces": result
        }
    
    def get_face_adjacency(self, step_path: str) -> Dict[str, Any]:
        """
        Berechnet die Face-Adjazenz (für GNN).
        
        Zwei Faces sind adjazent wenn sie eine Edge teilen.
        
        Returns:
            Dict mit:
            - adjacency_list: {face_id: [neighbor_face_ids]}
            - edges: [{face1, face2, shared_edge_id}]
            - num_faces, num_edges
        """
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        from OCC.Core.TopExp import topexp
        
        shape = self._load_step(step_path)
        faces = self._get_faces(step_path)
        
        # Map: Edge -> Liste von Faces die diese Edge teilen
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
        
        # Face zu Index Mapping (über Position im Explorer)
        # Wir müssen die Faces identifizieren - verwenden ihre Position
        face_shapes = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_shapes.append(explorer.Current())
            explorer.Next()
        
        # Adjazenz berechnen
        adjacency_list: Dict[int, List[int]] = {i: [] for i in range(len(faces))}
        edge_list = []
        
        for edge_idx in range(1, edge_face_map.Size() + 1):
            edge = edge_face_map.FindKey(edge_idx)
            face_list = edge_face_map.FindFromIndex(edge_idx)
            
            # Faces die diese Edge teilen - mit korrektem Iterator
            connected_faces = []
            from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
            it = TopTools_ListIteratorOfListOfShape(face_list)
            while it.More():
                face_shape = it.Value()
                # Finde Index durch Vergleich
                for idx, fs in enumerate(face_shapes):
                    if fs.IsSame(face_shape):
                        connected_faces.append(idx)
                        break
                it.Next()
            
            # Adjazenz eintragen
            if len(connected_faces) == 2:
                f1, f2 = connected_faces
                if f2 not in adjacency_list[f1]:
                    adjacency_list[f1].append(f2)
                if f1 not in adjacency_list[f2]:
                    adjacency_list[f2].append(f1)
                
                edge_list.append({
                    "edge_id": edge_idx - 1,
                    "face1": f1,
                    "face2": f2
                })
        
        return {
            "num_faces": len(faces),
            "num_edges": len(edge_list),
            "adjacency_list": adjacency_list,
            "edges": edge_list
        }
    
    def get_gnn_data(self, step_path: str, pmi_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extrahiert alle Daten die für Graph Neural Networks benötigt werden.
        
        Reichhaltige Features für jeden Node (Face) und Edge (gemeinsame Kante).
        
        Args:
            step_path: Pfad zur STEP-Datei
            pmi_data: Optional - PMI Daten aus STEP Text Parser
        
        Returns:
            Dict mit:
            - nodes: Reichhaltige Face Features
            - edges: Reichhaltige Edge Features  
            - edge_index: PyTorch Geometric Format
            - adjacency_list: Quick lookup
            - pmi_graph: PMI Annotationen am Graph
        """
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import (
            GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
            GeomAbs_BSplineCurve, GeomAbs_BezierCurve
        )
        from OCC.Core.GCPnts import GCPnts_AbscissaPoint
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
        from OCC.Core.TopExp import topexp
        
        shape = self._load_step(step_path)
        faces = self._get_faces(step_path)
        
        # =====================================================================
        # NODE FEATURES (Faces)
        # =====================================================================
        nodes = []
        for i, face in enumerate(faces):
            # Basis Properties
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            
            # Surface Adaptor für detaillierte Infos
            adaptor = BRepAdaptor_Surface(face)
            surf_type = self._get_surface_type(face)
            
            # UV Parameter-Bereich
            u_min = adaptor.FirstUParameter()
            u_max = adaptor.LastUParameter()
            v_min = adaptor.FirstVParameter()
            v_max = adaptor.LastVParameter()
            
            # Curvature (Hauptkrümmungen an UV-Mitte)
            u_mid = (u_min + u_max) / 2
            v_mid = (v_min + v_max) / 2
            
            curvature_info = self._get_face_curvature(adaptor, u_mid, v_mid)
            
            # ─── UV-Grid Sampling (10x10 = 100 Punkte pro Face) ───
            # SOTA für GNN: Krümmungsfeld + Normalen über die Fläche abtasten
            # 8 Channels: [x, y, z, nx, ny, nz, gauss_k, mean_k]
            uv_result = self._sample_uv_grid(adaptor, u_min, u_max, v_min, v_max, 
                                              grid_size=10, face=face)
            uv_grid = uv_result["points"]
            trimming_mask = uv_result["trimming_mask"]
            
            # Parametrische Info (Radius für Zylinder/Kugel, etc.)
            parametric = self._get_parametric_info(adaptor, surf_type)
            
            # Face Bounding Box
            face_bbox = self._get_shape_bbox(face)
            
            # Anzahl Edges/Loops
            num_edges = 0
            num_loops = 0
            edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_exp.More():
                num_edges += 1
                edge_exp.Next()
            
            # Node Feature Dict
            node = {
                "id": i,
                # Geometrie-Typ
                "surface_type": surf_type,
                "surface_type_encoded": self._encode_surface_type(surf_type),
                # Metriken
                "area": round(props.Mass(), 6),
                "perimeter": self._get_face_perimeter(face),
                # Position
                "centroid": self._get_face_centroid(face),
                "normal": self._get_face_normal(face),
                # Curvature
                "curvature_min": curvature_info.get("min", 0),
                "curvature_max": curvature_info.get("max", 0),
                "curvature_gaussian": curvature_info.get("gaussian", 0),
                "curvature_mean": curvature_info.get("mean", 0),
                # Parametrisch
                **parametric,
                # Bounding Box
                "bbox_min": face_bbox["min"],
                "bbox_max": face_bbox["max"],
                "bbox_size": face_bbox["size"],
                # Topologie
                "num_edges": num_edges,
                "num_loops": num_loops,
                # UV-Grid (10x10 Punkte: xyz+normal+curvature = 8 pro Punkt = 800 floats)
                "uv_grid": uv_grid,
                # Trimming Mask (10x10 = 100 bools, ob Punkt innerhalb Face-Boundary)
                "trimming_mask": trimming_mask,
                # PMI (wird unten gefüllt)
                "pmi_annotations": []
            }
            nodes.append(node)
        
        # =====================================================================
        # EDGE FEATURES (Face-Face Kanten)
        # =====================================================================
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
        
        # Face zu Index Mapping
        face_shapes = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_shapes.append(explorer.Current())
            explorer.Next()
        
        edges = []
        adjacency_list: Dict[int, List[int]] = {i: [] for i in range(len(faces))}
        edge_index = [[], []]
        
        for edge_idx in range(1, edge_face_map.Size() + 1):
            edge_shape = edge_face_map.FindKey(edge_idx)
            face_list = edge_face_map.FindFromIndex(edge_idx)
            
            # Faces die diese Edge teilen
            connected_faces = []
            it = TopTools_ListIteratorOfListOfShape(face_list)
            while it.More():
                face_shape = it.Value()
                for idx, fs in enumerate(face_shapes):
                    if fs.IsSame(face_shape):
                        connected_faces.append(idx)
                        break
                it.Next()
            
            # Nur wenn 2 Faces (= echte Adjazenz)
            if len(connected_faces) == 2:
                f1, f2 = connected_faces
                
                # Edge Geometrie analysieren
                try:
                    edge_adaptor = BRepAdaptor_Curve(topods.Edge(edge_shape))
                    edge_type = self._get_edge_type(edge_adaptor)
                    edge_length = GCPnts_AbscissaPoint.Length(edge_adaptor)
                except:
                    edge_type = "Unknown"
                    edge_length = 0
                
                # Dihedral Angle (Winkel zwischen den Faces)
                dihedral = self._compute_dihedral_angle(
                    nodes[f1]["normal"], 
                    nodes[f2]["normal"]
                )
                
                # Convexity (konvex = positiv, konkav = negativ)
                convexity = self._compute_convexity(
                    faces[f1], faces[f2], edge_shape,
                    nodes[f1]["centroid"], nodes[f2]["centroid"]
                )
                
                edge_feature = {
                    "id": len(edges),
                    "source": f1,
                    "target": f2,
                    # Geometrie
                    "edge_type": edge_type,
                    "edge_type_encoded": self._encode_edge_type(edge_type),
                    "length": round(edge_length, 6),
                    # Winkel
                    "dihedral_angle": round(dihedral, 4),
                    "dihedral_angle_rad": round(math.radians(dihedral), 4),
                    # Konvexität
                    "convexity": convexity,  # "convex", "concave", "smooth"
                    "convexity_encoded": self._encode_convexity(convexity),
                }
                edges.append(edge_feature)
                
                # Adjazenz
                if f2 not in adjacency_list[f1]:
                    adjacency_list[f1].append(f2)
                if f1 not in adjacency_list[f2]:
                    adjacency_list[f2].append(f1)
                
                # Edge Index (bidirektional für GNN)
                edge_index[0].extend([f1, f2])
                edge_index[1].extend([f2, f1])
        
        # =====================================================================
        # PMI INTEGRATION (wenn vorhanden)
        # =====================================================================
        pmi_graph = {
            "has_pmi": False,
            "tolerances": [],
            "dimensions": [],
            "surface_finishes": [],
            "datums": []
        }
        
        if pmi_data:
            pmi_graph["has_pmi"] = True
            
            # Toleranzen
            for tol in pmi_data.get("tolerances", []):
                pmi_entry = {
                    "type": tol.get("type"),
                    "value": tol.get("value"),
                    "unit": tol.get("unit", "mm"),
                    "face_ids": tol.get("face_ids", []),
                    "from_face": tol.get("from_face"),
                    "to_face": tol.get("to_face")
                }
                pmi_graph["tolerances"].append(pmi_entry)
                
                # An Nodes annotieren
                for fid in tol.get("face_ids", []):
                    if 0 <= fid < len(nodes):
                        nodes[fid]["pmi_annotations"].append({
                            "type": "tolerance",
                            "subtype": tol.get("type"),
                            "value": tol.get("value")
                        })
            
            # Dimensions
            for dim in pmi_data.get("dimensions", []):
                pmi_entry = {
                    "type": dim.get("type"),
                    "value": dim.get("value"),
                    "unit": dim.get("unit", "mm"),
                    "face_ids": dim.get("face_ids", [])
                }
                pmi_graph["dimensions"].append(pmi_entry)
                
                for fid in dim.get("face_ids", []):
                    if 0 <= fid < len(nodes):
                        nodes[fid]["pmi_annotations"].append({
                            "type": "dimension",
                            "subtype": dim.get("type"),
                            "value": dim.get("value")
                        })
            
            # Surface Finishes
            for sf in pmi_data.get("surface_finishes", []):
                pmi_entry = {
                    "roughness": sf.get("roughness"),
                    "face_ids": sf.get("face_ids", [])
                }
                pmi_graph["surface_finishes"].append(pmi_entry)
                
                for fid in sf.get("face_ids", []):
                    if 0 <= fid < len(nodes):
                        nodes[fid]["pmi_annotations"].append({
                            "type": "surface_finish",
                            "roughness": sf.get("roughness")
                        })
            
            # Datums
            for datum in pmi_data.get("datums", []):
                pmi_graph["datums"].append({
                    "label": datum.get("label"),
                    "face_ids": datum.get("face_ids", [])
                })
                
                for fid in datum.get("face_ids", []):
                    if 0 <= fid < len(nodes):
                        nodes[fid]["pmi_annotations"].append({
                            "type": "datum",
                            "label": datum.get("label")
                        })
        
        # =====================================================================
        # RESULT
        # =====================================================================
        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "edge_index": edge_index,
            "adjacency_list": adjacency_list,
            "pmi_graph": pmi_graph,
            # Feature Dimensions für GNN
            "node_feature_dim": self._count_node_features(nodes[0] if nodes else {}),
            "edge_feature_dim": self._count_edge_features(edges[0] if edges else {})
        }
    
    def _get_face_curvature(self, adaptor: BRepAdaptor_Surface, u: float, v: float) -> Dict:
        """Berechnet Krümmungen an einem Punkt"""
        try:
            from OCC.Core.BRepLProp import BRepLProp_SLProps
            props = BRepLProp_SLProps(adaptor, u, v, 2, 1e-6)
            
            if props.IsCurvatureDefined():
                k1 = props.MinCurvature()
                k2 = props.MaxCurvature()
                return {
                    "min": round(k1, 6),
                    "max": round(k2, 6),
                    "gaussian": round(k1 * k2, 6),
                    "mean": round((k1 + k2) / 2, 6)
                }
        except:
            pass
        return {"min": 0, "max": 0, "gaussian": 0, "mean": 0}
    
    def _sample_uv_grid(self, adaptor: BRepAdaptor_Surface,
                        u_min: float, u_max: float,
                        v_min: float, v_max: float,
                        grid_size: int = 10,
                        face=None) -> Dict[str, List]:
        """
        Tastet ein gleichmäßiges UV-Grid auf der Face ab.
        
        Pro Punkt: [x, y, z, nx, ny, nz, gauss_k, mean_k] = 8 Features
        Zusätzlich: trimming_mask (0/1 pro Punkt, ob innerhalb der Face-Boundary)
        
        Kompatibel mit:
        - uv_encoder.py: [x,y,z, nx,ny,nz, gaussian_curv, mean_curv] = 8
        - train_real_cad.py: erwartet 8 Channels
        - ReGraph Format: points(3) + normals(3) + trimming_mask
        
        SOTA für GNN-basiertes CAD-Verständnis (BRepNet-Style).
        
        Returns:
            Dict mit:
            - "points": Liste von [x,y,z, nx,ny,nz, gauss_k, mean_k] (8 pro Punkt)
            - "trimming_mask": Liste von 0/1 (ob Punkt innerhalb Face-Boundary)
        """
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.gp import gp_Pnt, gp_Pnt2d
        
        points = []
        trimming_mask = []
        
        # Trimming-Classifier für Face-Boundary Check
        classifier = None
        if face is not None:
            try:
                from OCC.Core.BRepClass import BRepClass_FaceClassifier
                from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
                classifier = (BRepClass_FaceClassifier, face, TopAbs_IN, TopAbs_ON)
            except ImportError:
                pass
        
        # Clamp UV Bereich auf sinnvolle Grenzen (periodische Flächen können [-inf, inf] haben)
        u_range = u_max - u_min
        v_range = v_max - v_min
        max_range = 1000.0  # Sinnvolle Obergrenze
        if u_range > max_range or u_range <= 0:
            u_min, u_max = 0.0, 2 * 3.14159265
            u_range = u_max - u_min
        if v_range > max_range or v_range <= 0:
            v_min, v_max = 0.0, 2 * 3.14159265
            v_range = v_max - v_min
        
        u_step = u_range / (grid_size - 1) if grid_size > 1 else 0
        v_step = v_range / (grid_size - 1) if grid_size > 1 else 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                u = u_min + i * u_step
                v = v_min + j * v_step
                
                try:
                    # 3D Punkt auf der Fläche
                    pnt = adaptor.Value(u, v)
                    x = round(pnt.X(), 4)
                    y = round(pnt.Y(), 4)
                    z = round(pnt.Z(), 4)
                    
                    # Normal + Krümmungen an diesem Punkt
                    nx, ny, nz = 0.0, 0.0, 0.0
                    gauss_k, mean_k = 0.0, 0.0
                    try:
                        slprops = BRepLProp_SLProps(adaptor, u, v, 2, 1e-6)
                        # Normale
                        if slprops.IsNormalDefined():
                            normal = slprops.Normal()
                            nx = round(normal.X(), 4)
                            ny = round(normal.Y(), 4)
                            nz = round(normal.Z(), 4)
                        # Krümmungen (Gaussian + Mean, kompatibel mit uv_encoder.py)
                        if slprops.IsCurvatureDefined():
                            gauss_k = round(slprops.GaussianCurvature(), 6)
                            mean_k = round(slprops.MeanCurvature(), 6)
                            # Clamp extreme Werte
                            gauss_k = max(-100.0, min(100.0, gauss_k))
                            mean_k = max(-100.0, min(100.0, mean_k))
                    except:
                        pass
                    
                    points.append([x, y, z, nx, ny, nz, gauss_k, mean_k])
                    
                    # Trimming Mask: Punkt innerhalb der Face-Boundary?
                    if classifier is not None:
                        try:
                            BRepClass_FC, fc, IN, ON = classifier
                            fc_inst = BRepClass_FC(fc, gp_Pnt2d(u, v), 1e-6)
                            state = fc_inst.State()
                            trimming_mask.append(1 if state in (IN, ON) else 0)
                        except:
                            trimming_mask.append(1)  # Fallback: assume inside
                    else:
                        trimming_mask.append(1)  # Kein Face → alle "inside"
                except:
                    points.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    trimming_mask.append(0)
        
        return {"points": points, "trimming_mask": trimming_mask}
    
    def _get_parametric_info(self, adaptor: BRepAdaptor_Surface, surf_type: str) -> Dict:
        """Extrahiert parametrische Info je nach Surface Type"""
        info = {
            "radius": None,
            "radius2": None,  # für Torus
            "semi_angle": None,  # für Cone
        }
        
        try:
            if surf_type == "Cylinder":
                cyl = adaptor.Cylinder()
                info["radius"] = round(cyl.Radius(), 6)
            elif surf_type == "Sphere":
                sph = adaptor.Sphere()
                info["radius"] = round(sph.Radius(), 6)
            elif surf_type == "Cone":
                cone = adaptor.Cone()
                info["radius"] = round(cone.RefRadius(), 6)
                info["semi_angle"] = round(math.degrees(cone.SemiAngle()), 4)
            elif surf_type == "Torus":
                torus = adaptor.Torus()
                info["radius"] = round(torus.MajorRadius(), 6)
                info["radius2"] = round(torus.MinorRadius(), 6)
        except:
            pass
        
        return info
    
    def _get_face_perimeter(self, face) -> float:
        """Berechnet den Umfang einer Face"""
        try:
            perimeter = 0
            exp = TopExp_Explorer(face, TopAbs_EDGE)
            while exp.More():
                edge = topods.Edge(exp.Current())
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                from OCC.Core.GCPnts import GCPnts_AbscissaPoint
                adaptor = BRepAdaptor_Curve(edge)
                perimeter += GCPnts_AbscissaPoint.Length(adaptor)
                exp.Next()
            return round(perimeter, 6)
        except:
            return 0
    
    def _get_shape_bbox(self, shape) -> Dict:
        """Bounding Box einer Shape"""
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return {
            "min": [round(xmin, 4), round(ymin, 4), round(zmin, 4)],
            "max": [round(xmax, 4), round(ymax, 4), round(zmax, 4)],
            "size": [round(xmax-xmin, 4), round(ymax-ymin, 4), round(zmax-zmin, 4)]
        }
    
    def _get_edge_type(self, adaptor) -> str:
        """Bestimmt den Edge-Typ"""
        from OCC.Core.GeomAbs import (
            GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
            GeomAbs_BSplineCurve, GeomAbs_BezierCurve
        )
        t = adaptor.GetType()
        type_map = {
            GeomAbs_Line: "Line",
            GeomAbs_Circle: "Circle",
            GeomAbs_Ellipse: "Ellipse",
            GeomAbs_BSplineCurve: "BSpline",
            GeomAbs_BezierCurve: "Bezier"
        }
        return type_map.get(t, "Other")
    
    def _compute_dihedral_angle(self, normal1: List, normal2: List) -> float:
        """Berechnet Winkel zwischen zwei Normalen (in Grad)"""
        if not normal1 or not normal2:
            return 0
        
        # Dot product
        dot = sum(a*b for a, b in zip(normal1, normal2))
        dot = max(-1, min(1, dot))  # Clamp
        angle = math.degrees(math.acos(dot))
        return angle
    
    def _compute_convexity(self, face1, face2, edge, centroid1: List, centroid2: List) -> str:
        """Bestimmt ob die Kante konvex oder konkav ist"""
        try:
            # Vereinfachte Heuristik: Vergleiche Zentroid-Distanz mit Normalenrichtung
            n1 = self._get_face_normal_raw(face1)
            n2 = self._get_face_normal_raw(face2)
            
            if not n1 or not n2:
                return "smooth"
            
            # Vektor von centroid1 zu centroid2
            vec = [c2 - c1 for c1, c2 in zip(centroid1, centroid2)]
            
            # Dot mit Normale
            dot1 = sum(v*n for v, n in zip(vec, n1))
            
            if abs(dot1) < 0.01:
                return "smooth"
            elif dot1 > 0:
                return "convex"
            else:
                return "concave"
        except:
            return "smooth"
    
    def _get_face_normal_raw(self, face) -> Optional[List[float]]:
        """Raw Normal ohne Runden"""
        try:
            adaptor = BRepAdaptor_Surface(face)
            u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
            v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
            
            from OCC.Core.BRepLProp import BRepLProp_SLProps
            props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
            
            if props.IsNormalDefined():
                normal = props.Normal()
                return [normal.X(), normal.Y(), normal.Z()]
        except:
            pass
        return None
    
    def _encode_surface_type(self, surf_type: str) -> int:
        """One-hot index für Surface Type"""
        types = ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSpline", "Other"]
        return types.index(surf_type) if surf_type in types else len(types) - 1
    
    def _encode_edge_type(self, edge_type: str) -> int:
        """One-hot index für Edge Type"""
        types = ["Line", "Circle", "Ellipse", "BSpline", "Bezier", "Other"]
        return types.index(edge_type) if edge_type in types else len(types) - 1
    
    def _encode_convexity(self, convexity: str) -> int:
        """Encoded Convexity: -1=concave, 0=smooth, 1=convex"""
        return {"concave": -1, "smooth": 0, "convex": 1}.get(convexity, 0)
    
    def _count_node_features(self, node: Dict) -> int:
        """Zählt numerische Features für GNN"""
        # area, perimeter, centroid(3), normal(3), curvatures(4), radius, bbox(9), num_edges
        # = 24 base features
        # + uv_grid: 10x10 x 8 (xyz + normal + gauss_k + mean_k) = 800
        # + trimming_mask: 10x10 = 100
        return 924  # 24 base + 800 uv_grid + 100 trimming_mask
    
    def _count_edge_features(self, edge: Dict) -> int:
        """Zählt Edge Features"""
        # length, dihedral_angle, dihedral_rad, convexity_encoded, edge_type_encoded
        return 5
    
    def measure_distance(self, step_path: str, face1_id: int, face2_id: int) -> Dict[str, float]:
        """
        Misst den minimalen Abstand zwischen zwei Faces.
        
        Args:
            step_path: Pfad zur STEP-Datei
            face1_id: Index der ersten Face
            face2_id: Index der zweiten Face
            
        Returns:
            Dict mit distance und closest_points
        """
        faces = self._get_faces(step_path)
        
        if face1_id < 0 or face1_id >= len(faces):
            raise ValueError(f"Face ID {face1_id} nicht gültig.")
        if face2_id < 0 or face2_id >= len(faces):
            raise ValueError(f"Face ID {face2_id} nicht gültig.")
        
        face1 = faces[face1_id]
        face2 = faces[face2_id]
        
        dist_calc = BRepExtrema_DistShapeShape(face1, face2)
        
        if dist_calc.IsDone():
            distance = dist_calc.Value()
            p1 = dist_calc.PointOnShape1(1)
            p2 = dist_calc.PointOnShape2(1)
            
            return {
                "distance": round(distance, 3),
                "point1": {"x": round(p1.X(), 3), "y": round(p1.Y(), 3), "z": round(p1.Z(), 3)},
                "point2": {"x": round(p2.X(), 3), "y": round(p2.Y(), 3), "z": round(p2.Z(), 3)}
            }
        else:
            return {"error": "Abstandsberechnung fehlgeschlagen"}
    
    def check_parallel(self, step_path: str, face1_id: int, face2_id: int, 
                       tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Prüft ob zwei Faces parallel sind (nur für Planes).
        
        Args:
            step_path: Pfad zur STEP-Datei
            face1_id: Index der ersten Face
            face2_id: Index der zweiten Face
            tolerance: Winkel-Toleranz in Radiant
            
        Returns:
            Dict mit is_parallel, angle, face_types
        """
        faces = self._get_faces(step_path)
        
        face1 = faces[face1_id]
        face2 = faces[face2_id]
        
        adaptor1 = BRepAdaptor_Surface(face1)
        adaptor2 = BRepAdaptor_Surface(face2)
        
        type1 = self._get_surface_type(face1)
        type2 = self._get_surface_type(face2)
        
        # Nur für Planes sinnvoll
        if type1 != "Plane" or type2 != "Plane":
            return {
                "is_parallel": None,
                "reason": "Parallelitätsprüfung nur für Plane-Faces möglich",
                "face1_type": type1,
                "face2_type": type2
            }
        
        # Normalen holen
        plane1 = adaptor1.Plane()
        plane2 = adaptor2.Plane()
        
        normal1 = plane1.Axis().Direction()
        normal2 = plane2.Axis().Direction()
        
        # Winkel zwischen Normalen
        dot = abs(normal1.X() * normal2.X() + 
                  normal1.Y() * normal2.Y() + 
                  normal1.Z() * normal2.Z())
        
        # Clamp für acos
        dot = min(1.0, max(-1.0, dot))
        angle = math.acos(dot)
        
        is_parallel = angle < tolerance or abs(angle - math.pi) < tolerance
        
        return {
            "is_parallel": is_parallel,
            "angle_rad": round(angle, 4),
            "angle_deg": round(math.degrees(angle), 2),
            "face1_type": type1,
            "face2_type": type2
        }
    
    def get_summary(self, step_path: str) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung des Bauteils zurück.
        
        Args:
            step_path: Pfad zur STEP-Datei
            
        Returns:
            Umfassende Zusammenfassung
        """
        return {
            "file": Path(step_path).name,
            "topology": self.count_topology(step_path),
            "dimensions": self.get_bounding_box(step_path),
            "properties": self.get_volume_and_area(step_path),
            "feature_types": self.count_features(step_path)
        }
    
    def get_hole_diameters(self, step_path: str) -> Dict[str, Any]:
        """
        Findet alle zylindrischen Bohrungen/Löcher und gibt deren Durchmesser zurück.
        Gruppiert nach Durchmesser für einfache Schraubenbestimmung.
        
        Args:
            step_path: Pfad zur STEP-Datei
            
        Returns:
            Dict mit holes (Liste) und summary (gruppiert nach Durchmesser)
        """
        faces = self._get_faces(step_path)
        holes = []
        
        for i, face in enumerate(faces):
            stype = self._get_surface_type(face)
            if stype == "Cylinder":
                adaptor = BRepAdaptor_Surface(face)
                parametric = self._get_parametric_info(adaptor, stype)
                radius = parametric.get("radius", 0)
                if radius and radius > 0:
                    props = GProp_GProps()
                    brepgprop.SurfaceProperties(face, props)
                    
                    # Höhe des Zylinders über Bounding Box schätzen
                    face_bbox = self._get_shape_bbox(face)
                    height = max(face_bbox["size"]) if face_bbox["size"] else 0
                    
                    holes.append({
                        "face_id": i,
                        "radius": radius,
                        "diameter": round(radius * 2, 4),
                        "area": round(props.Mass(), 3),
                        "estimated_depth": round(height, 3),
                        "centroid": self._get_face_centroid(face)
                    })
        
        # Nach Durchmesser gruppieren
        diameter_groups: Dict[float, int] = {}
        for h in holes:
            d = round(h["diameter"], 2)
            diameter_groups[d] = diameter_groups.get(d, 0) + 1
        
        # Metrische Schrauben-Zuordnung (Näherung)
        metric_screws = {
            2.0: "M2", 2.2: "M2.5", 2.5: "M2.5", 3.0: "M3", 3.2: "M3",
            3.3: "M3.5", 3.5: "M4", 4.0: "M4", 4.2: "M4", 4.5: "M5",
            5.0: "M5", 5.3: "M6", 6.0: "M6", 6.4: "M6", 6.8: "M8",
            7.0: "M8", 8.0: "M8", 8.4: "M10", 10.0: "M10", 10.5: "M12",
            12.0: "M12", 13.0: "M14", 14.0: "M14", 15.0: "M16", 16.0: "M16",
            17.0: "M18", 18.0: "M20", 20.0: "M20", 22.0: "M24", 24.0: "M24"
        }
        
        suggestions = {}
        for d, count in diameter_groups.items():
            # Nächste passende Schraube finden
            closest = min(metric_screws.keys(), key=lambda x: abs(x - d))
            if abs(closest - d) < 1.5:  # Max 1.5mm Abweichung
                suggestions[f"{d}mm"] = {
                    "count": count,
                    "likely_screw": metric_screws[closest],
                    "note": "Kernloch" if d < closest else "Durchgangsloch"
                }
            else:
                suggestions[f"{d}mm"] = {"count": count, "likely_screw": "unbekannt"}
        
        return {
            "total_cylinders": len(holes),
            "holes": holes,
            "diameter_summary": diameter_groups,
            "screw_suggestions": suggestions
        }


# Tool-Definitionen für LLM (OpenAI Function Calling Format)
TOOL_DEFINITIONS = [
    {
        "name": "get_bounding_box",
        "description": "Gibt die Maße (Bounding Box) eines CAD-Bauteils zurück: Breite, Höhe, Tiefe.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                }
            },
            "required": ["step_path"]
        }
    },
    {
        "name": "get_volume_and_area",
        "description": "Gibt Volumen und Oberfläche eines CAD-Bauteils zurück.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                }
            },
            "required": ["step_path"]
        }
    },
    {
        "name": "count_features",
        "description": "Zählt geometrische Features (Plane, Cylinder, Cone, Sphere, Torus, BSpline) im Bauteil.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                },
                "feature_type": {
                    "type": "string",
                    "description": "Optional: Nur diesen Typ zählen (Plane, Cylinder, etc.)",
                    "enum": ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSpline"]
                }
            },
            "required": ["step_path"]
        }
    },
    {
        "name": "list_faces",
        "description": "Listet alle Faces des Bauteils mit ID, Typ und Fläche.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                }
            },
            "required": ["step_path"]
        }
    },
    {
        "name": "get_face_info",
        "description": "Gibt detaillierte Informationen zu einer bestimmten Face zurück.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                },
                "face_id": {
                    "type": "integer",
                    "description": "ID der Face (0-basiert)"
                }
            },
            "required": ["step_path", "face_id"]
        }
    },
    {
        "name": "measure_distance",
        "description": "Misst den minimalen Abstand zwischen zwei Faces.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                },
                "face1_id": {
                    "type": "integer",
                    "description": "ID der ersten Face"
                },
                "face2_id": {
                    "type": "integer",
                    "description": "ID der zweiten Face"
                }
            },
            "required": ["step_path", "face1_id", "face2_id"]
        }
    },
    {
        "name": "check_parallel",
        "description": "Prüft ob zwei planare Faces parallel zueinander sind.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                },
                "face1_id": {
                    "type": "integer",
                    "description": "ID der ersten Face"
                },
                "face2_id": {
                    "type": "integer",
                    "description": "ID der zweiten Face"
                }
            },
            "required": ["step_path", "face1_id", "face2_id"]
        }
    },
    {
        "name": "get_summary",
        "description": "Gibt eine komplette Zusammenfassung des CAD-Bauteils zurück.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                }
            },
            "required": ["step_path"]
        }
    },
    {
        "name": "get_hole_diameters",
        "description": "Findet alle zylindrischen Bohrungen/Löcher im Bauteil und gibt deren Durchmesser, Tiefe und passende metrische Schraubengröße (M3, M4, M5, etc.) zurück. Perfekt um Schraubengrößen zu bestimmen.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {
                    "type": "string",
                    "description": "Pfad zur STEP-Datei"
                }
            },
            "required": ["step_path"]
        }
    }
]


if __name__ == "__main__":
    # Test
    tools = GeometryTools()
    step_path = "data/s2.0.1/breps/step/100027_258e3965_0.stp"
    
    print("=== GeometryTools Test ===\n")
    
    print("1. Bounding Box:")
    print(json.dumps(tools.get_bounding_box(step_path), indent=2))
    
    print("\n2. Volume & Area:")
    print(json.dumps(tools.get_volume_and_area(step_path), indent=2))
    
    print("\n3. Feature Counts:")
    print(json.dumps(tools.count_features(step_path), indent=2))
    
    print("\n4. Face List:")
    faces = tools.list_faces(step_path)
    for f in faces[:5]:  # Nur erste 5
        print(f"   Face {f['id']}: {f['type']}, Area={f['area']}")
    
    print("\n5. Summary:")
    print(json.dumps(tools.get_summary(step_path), indent=2))
