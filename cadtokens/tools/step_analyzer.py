#!/usr/bin/env python3
"""
STEP Analyzer - Phase 0.1

Analysiert STEP-Dateien und extrahiert geometrische Informationen.
Dies ist die Basis für alle weiteren Tools.

Usage:
    python src/tools/step_analyzer.py <path_to_step>
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

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
from OCC.Core.TopoDS import topods


@dataclass
class FaceInfo:
    """Informationen über eine einzelne Face"""
    id: int
    surface_type: str
    area: float
    centroid: tuple  # (x, y, z)
    normal: Optional[tuple] = None  # (nx, ny, nz)


@dataclass
class AnalysisResult:
    """Ergebnis der STEP-Analyse"""
    file_path: str
    n_faces: int
    n_edges: int
    n_vertices: int
    bounding_box: Dict[str, float]  # xmin, ymin, zmin, xmax, ymax, zmax
    dimensions: Dict[str, float]    # width, height, depth
    volume: float
    surface_area: float
    face_types: Dict[str, int]      # Anzahl pro Typ
    faces: List[FaceInfo]


def get_surface_type_name(adaptor: BRepAdaptor_Surface) -> str:
    """Konvertiert GeomAbs Type zu lesbarem String"""
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


def analyze_step(step_path: str) -> AnalysisResult:
    """
    Analysiert eine STEP-Datei vollständig.
    
    Args:
        step_path: Pfad zur STEP-Datei
        
    Returns:
        AnalysisResult mit allen extrahierten Informationen
    """
    # STEP laden
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:  # IFSelect_RetDone
        raise ValueError(f"Konnte STEP nicht laden: {step_path}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    
    # Topologie zählen
    n_faces = 0
    n_edges = 0
    n_vertices = 0
    
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        n_faces += 1
        explorer.Next()
        
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        n_edges += 1
        explorer.Next()
        
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        n_vertices += 1
        explorer.Next()
    
    # Bounding Box
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    bounding_box = {
        "xmin": xmin, "ymin": ymin, "zmin": zmin,
        "xmax": xmax, "ymax": ymax, "zmax": zmax
    }
    
    dimensions = {
        "width": xmax - xmin,
        "height": ymax - ymin,
        "depth": zmax - zmin
    }
    
    # Volumen und Oberfläche
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()
    
    brepgprop.SurfaceProperties(shape, props)
    surface_area = props.Mass()
    
    # Face-Details extrahieren
    faces = []
    face_types = {}
    
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while explorer.More():
        face = topods.Face(explorer.Current())
        
        # Surface-Typ
        adaptor = BRepAdaptor_Surface(face)
        stype = get_surface_type_name(adaptor)
        face_types[stype] = face_types.get(stype, 0) + 1
        
        # Fläche
        face_props = GProp_GProps()
        brepgprop.SurfaceProperties(face, face_props)
        area = face_props.Mass()
        
        # Schwerpunkt
        centroid = face_props.CentreOfMass()
        
        faces.append(FaceInfo(
            id=face_id,
            surface_type=stype,
            area=area,
            centroid=(centroid.X(), centroid.Y(), centroid.Z())
        ))
        
        face_id += 1
        explorer.Next()
    
    return AnalysisResult(
        file_path=step_path,
        n_faces=n_faces,
        n_edges=n_edges,
        n_vertices=n_vertices,
        bounding_box=bounding_box,
        dimensions=dimensions,
        volume=volume,
        surface_area=surface_area,
        face_types=face_types,
        faces=faces
    )


def print_analysis(result: AnalysisResult):
    """Gibt Analyse-Ergebnis formatiert aus"""
    print(f"\n{'='*60}")
    print(f"STEP Analysis: {Path(result.file_path).name}")
    print(f"{'='*60}\n")
    
    print("📊 Topologie:")
    print(f"   Faces:    {result.n_faces}")
    print(f"   Edges:    {result.n_edges}")
    print(f"   Vertices: {result.n_vertices}")
    
    print(f"\n📏 Dimensionen:")
    print(f"   Width:  {result.dimensions['width']:.2f}")
    print(f"   Height: {result.dimensions['height']:.2f}")
    print(f"   Depth:  {result.dimensions['depth']:.2f}")
    
    print(f"\n📦 Eigenschaften:")
    print(f"   Volume:       {result.volume:.2f}")
    print(f"   Surface Area: {result.surface_area:.2f}")
    
    print(f"\n🔷 Face-Typen:")
    for ftype, count in sorted(result.face_types.items(), key=lambda x: -x[1]):
        print(f"   {ftype}: {count}")
    
    print(f"\n{'='*60}\n")


def to_json(result: AnalysisResult) -> str:
    """Konvertiert Ergebnis zu JSON"""
    return json.dumps({
        "file_path": result.file_path,
        "topology": {
            "faces": result.n_faces,
            "edges": result.n_edges,
            "vertices": result.n_vertices
        },
        "bounding_box": result.bounding_box,
        "dimensions": result.dimensions,
        "volume": result.volume,
        "surface_area": result.surface_area,
        "face_types": result.face_types,
        "faces": [
            {
                "id": f.id,
                "type": f.surface_type,
                "area": f.area,
                "centroid": f.centroid
            }
            for f in result.faces
        ]
    }, indent=2)


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    if len(sys.argv) < 2:
        # Default: erste STEP-Datei aus Dataset
        step_path = "data/s2.0.1/breps/step/00001_0dbd7c42.step"
        if not Path(step_path).exists():
            print("Usage: python step_analyzer.py <path_to_step>")
            sys.exit(1)
    else:
        step_path = sys.argv[1]
    
    try:
        result = analyze_step(step_path)
        print_analysis(result)
        
        # Optional: JSON ausgeben
        if "--json" in sys.argv:
            print(to_json(result))
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
        sys.exit(1)
