#!/usr/bin/env python3
"""
CAD Renderer - Rendert STEP-Dateien zu Bildern und Meshes

Für:
1. Vision-Input an LLM (qwen3-vl)
2. 3D Viewer im Frontend (STL/OBJ Export)
3. Face-Highlighting für interaktive Analyse
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import tempfile
import numpy as np

# PythonOCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation

# Optionale Rendering-Imports
try:
    from OCC.Display.SimpleGui import init_display
    HAS_DISPLAY = True
except:
    HAS_DISPLAY = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("⚠️  trimesh nicht installiert. Für besseres Rendering: pip install trimesh")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class CADRenderer:
    """Rendert STEP-Dateien zu verschiedenen Formaten"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "cad_renders"
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_step(self, step_path: str) -> TopoDS_Shape:
        """Lädt STEP-Datei"""
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != 1:
            raise ValueError(f"Konnte STEP nicht laden: {step_path}")
        reader.TransferRoots()
        return reader.OneShape()
    
    def step_to_stl(self, step_path: str, output_path: Optional[str] = None,
                    linear_deflection: float = 0.1, angular_deflection: float = 0.5) -> str:
        """
        Konvertiert STEP zu STL für 3D Viewer.
        
        Args:
            step_path: Pfad zur STEP-Datei
            output_path: Ausgabepfad (optional, sonst Cache)
            linear_deflection: Mesh-Qualität (kleiner = feiner)
            
        Returns:
            Pfad zur STL-Datei
        """
        shape = self.load_step(step_path)
        
        # Mesh generieren
        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()
        
        # Output Pfad
        if output_path is None:
            stem = Path(step_path).stem
            output_path = str(self.cache_dir / f"{stem}.stl")
        
        # STL schreiben
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)  # Binary für kleinere Dateien
        writer.Write(shape, output_path)
        
        return output_path
    
    def step_to_glb(self, step_path: str, output_path: Optional[str] = None,
                    highlight_faces: Optional[List[int]] = None,
                    highlight_color: Tuple[int, int, int, int] = (255, 100, 100, 255),
                    base_color: Tuple[int, int, int, int] = (180, 180, 190, 255)) -> str:
        """
        Konvertiert STEP zu GLB (für Gradio Model3D) mit optionalem Face-Highlighting.
        
        Args:
            step_path: Pfad zur STEP-Datei
            output_path: Ausgabepfad (optional)
            highlight_faces: Liste von Face-IDs die gehighlighted werden sollen
            highlight_color: RGBA Farbe für Highlights (default: rot)
            base_color: RGBA Farbe für normale Faces (default: grau)
        
        Returns:
            Pfad zur GLB-Datei
        """
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh nicht installiert: pip install trimesh")
        
        shape = self.load_step(step_path)
        
        # Mesh generieren
        linear_deflection = 0.1
        angular_deflection = 0.5
        mesh_gen = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh_gen.Perform()
        
        # Alle Faces extrahieren mit ihren Triangles
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0
        
        highlight_set = set(highlight_faces) if highlight_faces else set()
        
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        
        while explorer.More():
            face = topods.Face(explorer.Current())
            
            # Triangulation der Face holen
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            
            if triangulation is not None:
                # Vertices
                for i in range(1, triangulation.NbNodes() + 1):
                    pnt = triangulation.Node(i)
                    # Transformation anwenden
                    if not location.IsIdentity():
                        pnt = pnt.Transformed(location.Transformation())
                    all_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
                
                # Triangles
                for i in range(1, triangulation.NbTriangles() + 1):
                    tri = triangulation.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    # 0-indexed und mit offset
                    all_faces.append([
                        vertex_offset + n1 - 1,
                        vertex_offset + n2 - 1,
                        vertex_offset + n3 - 1
                    ])
                    
                    # Farbe für dieses Triangle
                    if face_idx in highlight_set:
                        all_colors.append(highlight_color)
                    else:
                        all_colors.append(base_color)
                
                vertex_offset += triangulation.NbNodes()
            
            face_idx += 1
            explorer.Next()
        
        if not all_vertices:
            raise ValueError("Keine Geometrie in der STEP-Datei gefunden")
        
        # Trimesh erstellen mit Vertex Colors
        vertices = np.array(all_vertices)
        faces = np.array(all_faces)
        face_colors = np.array(all_colors, dtype=np.uint8)
        
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_colors=face_colors
        )
        
        # Output Pfad
        if output_path is None:
            stem = Path(step_path).stem
            suffix = "_highlighted" if highlight_faces else ""
            output_path = str(self.cache_dir / f"{stem}{suffix}.glb")
        
        mesh.export(output_path)
        return output_path
    
    def step_to_obj(self, step_path: str, output_path: Optional[str] = None) -> str:
        """Konvertiert STEP zu OBJ"""
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh nicht installiert: pip install trimesh")
        
        stl_path = self.step_to_stl(step_path)
        mesh = trimesh.load(stl_path)
        
        if output_path is None:
            stem = Path(step_path).stem
            output_path = str(self.cache_dir / f"{stem}.obj")
        
        mesh.export(output_path)
        return output_path
    
    def render_to_image(self, step_path: str, output_path: Optional[str] = None,
                        size: Tuple[int, int] = (512, 512),
                        views: List[str] = ["front"]) -> List[str]:
        """
        Rendert STEP zu Bild(ern) mit matplotlib (macOS-kompatibel).
        
        Args:
            step_path: Pfad zur STEP-Datei
            output_path: Ausgabe-Basisname
            size: Bildgröße (width, height)
            views: Liste von Views ["front", "top", "side", "iso"]
            
        Returns:
            Liste von Bildpfaden
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh nicht installiert: pip install trimesh")
        
        # Zu STL konvertieren und laden
        stl_path = self.step_to_stl(step_path)
        mesh = trimesh.load(stl_path)
        
        # Mesh zentrieren und normalisieren
        mesh.vertices -= mesh.centroid
        scale = 1.0 / mesh.extents.max()
        mesh.vertices *= scale
        
        stem = Path(step_path).stem
        image_paths = []
        
        # View-Konfigurationen (elevation, azimuth)
        view_configs = {
            "front": (0, 0),
            "back": (0, 180),
            "top": (90, 0),
            "bottom": (-90, 0),
            "left": (0, 90),
            "right": (0, -90),
            "iso": (30, 45),
        }
        
        for view in views:
            try:
                fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
                ax = fig.add_subplot(111, projection='3d')
                
                # Mesh als Poly3DCollection
                verts = mesh.vertices[mesh.faces]
                poly = Poly3DCollection(verts, alpha=0.9)
                poly.set_facecolor('#6699CC')
                poly.set_edgecolor('#333333')
                poly.set_linewidth(0.1)
                ax.add_collection3d(poly)
                
                # Achsen-Limits
                lim = 0.6
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
                
                # View setzen
                elev, azim = view_configs.get(view, (30, 45))
                ax.view_init(elev=elev, azim=azim)
                
                # Achsen ausblenden
                ax.set_axis_off()
                ax.set_box_aspect([1, 1, 1])
                
                # Speichern
                if output_path:
                    img_path = f"{output_path}_{view}.png"
                else:
                    img_path = str(self.cache_dir / f"{stem}_{view}.png")
                
                plt.tight_layout()
                plt.savefig(img_path, dpi=100, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
                plt.close(fig)
                
                image_paths.append(img_path)
                
            except Exception as e:
                print(f"⚠️  Rendering fehlgeschlagen für {view}: {e}")
                continue
        
        return image_paths
    
    def get_thumbnail(self, step_path: str, size: Tuple[int, int] = (256, 256)) -> Optional[str]:
        """
        Erstellt ein einzelnes Thumbnail-Bild.
        
        Returns:
            Pfad zum Bild oder None wenn Rendering fehlschlägt
        """
        try:
            images = self.render_to_image(step_path, size=size, views=["iso"])
            return images[0] if images else None
        except Exception as e:
            print(f"⚠️  Thumbnail-Erstellung fehlgeschlagen: {e}")
            return None


# Singleton für einfachen Zugriff
_renderer = None

def get_renderer() -> CADRenderer:
    global _renderer
    if _renderer is None:
        _renderer = CADRenderer()
    return _renderer


if __name__ == "__main__":
    import sys
    
    renderer = CADRenderer()
    
    # Test
    step_path = sys.argv[1] if len(sys.argv) > 1 else "data/s2.0.1/breps/step/100027_258e3965_0.stp"
    
    print(f"📁 Verarbeite: {step_path}")
    
    # STL
    stl_path = renderer.step_to_stl(step_path)
    print(f"✅ STL: {stl_path}")
    
    # GLB (für Gradio)
    try:
        glb_path = renderer.step_to_glb(step_path)
        print(f"✅ GLB: {glb_path}")
    except Exception as e:
        print(f"⚠️  GLB: {e}")
    
    # Bild
    try:
        images = renderer.render_to_image(step_path, views=["iso"])
        for img in images:
            print(f"✅ Image: {img}")
    except Exception as e:
        print(f"⚠️  Image: {e}")
