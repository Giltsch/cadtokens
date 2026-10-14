"""
Utility: Convert OCC TopoDS_Shape → PyG Data in 906-dim format.

Uses the same feature layout as our pre-trained BRepGNN:
    [6 base] + [800 UV-grid (10x10 × 8ch)] + [100 trimming] = 906

Reusable for synthetic data generation and any OCC → GNN pipeline.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple, List


# Surface type mapping (matches brep_gnn.py)
FACE_TYPE_MAP = {
    0: 0,  # GeomAbs_Plane
    1: 1,  # GeomAbs_Cylinder
    2: 2,  # GeomAbs_Cone
    3: 3,  # GeomAbs_Sphere
    4: 4,  # GeomAbs_Torus
    7: 5,  # GeomAbs_BSplineSurface
}


def shape_to_pyg(shape, grid_size: int = 10, normalize: bool = True) -> Optional[Data]:
    """
    Convert an OCC TopoDS_Shape to PyG Data with 906-dim node features.

    Args:
        shape: OCC TopoDS_Shape (must have faces)
        grid_size: UV sampling resolution (default 10 → 10x10 grid)
        normalize: Normalize xyz coordinates to unit scale

    Returns:
        PyG Data with:
            x:          [N, 906] node features
            edge_index: [2, E] COO adjacency
            edge_attr:  [E, 4] edge features (zeros for now)
            face_types: [N] surface type indices
        or None if shape has no faces
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
        GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    # Collect faces
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(topods.Face(explorer.Current()))
        explorer.Next()

    if not faces:
        return None

    num_faces = len(faces)

    # Per-face feature extraction
    all_grids = np.zeros((num_faces, grid_size, grid_size, 8), dtype=np.float32)
    all_trimming = np.ones((num_faces, grid_size * grid_size), dtype=np.float32)
    face_types = np.zeros(num_faces, dtype=np.int64)
    base_features = np.zeros((num_faces, 6), dtype=np.float32)

    for fi, face in enumerate(faces):
        adaptor = BRepAdaptor_Surface(face)
        surface = BRep_Tool.Surface(face)

        # Surface type
        stype = adaptor.GetType()
        face_types[fi] = FACE_TYPE_MAP.get(stype, 5)

        # UV bounds
        u_min = adaptor.FirstUParameter()
        u_max = adaptor.LastUParameter()
        v_min = adaptor.FirstVParameter()
        v_max = adaptor.LastVParameter()

        # Clamp periodic/infinite parameters
        u_min = max(u_min, -1000)
        u_max = min(u_max, 1000)
        v_min = max(v_min, -1000)
        v_max = min(v_max, 1000)
        if u_max - u_min <= 0: u_min, u_max = 0.0, 6.283185
        if v_max - v_min <= 0: v_min, v_max = 0.0, 6.283185

        u_vals = np.linspace(u_min, u_max, grid_size)
        v_vals = np.linspace(v_min, v_max, grid_size)

        # Sample UV grid
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                try:
                    pnt = adaptor.Value(u, v)
                    all_grids[fi, i, j, 0] = pnt.X()
                    all_grids[fi, i, j, 1] = pnt.Y()
                    all_grids[fi, i, j, 2] = pnt.Z()

                    props = GeomLProp_SLProps(surface, u, v, 2, 1e-6)
                    if props.IsNormalDefined():
                        n = props.Normal()
                        all_grids[fi, i, j, 3] = n.X()
                        all_grids[fi, i, j, 4] = n.Y()
                        all_grids[fi, i, j, 5] = n.Z()
                    if props.IsCurvatureDefined():
                        all_grids[fi, i, j, 6] = np.clip(props.GaussianCurvature(), -100, 100)
                        all_grids[fi, i, j, 7] = np.clip(props.MeanCurvature(), -100, 100)
                except Exception:
                    pass

        # Trimming mask via face classifier
        try:
            from OCC.Core.BRepClass import BRepClass_FaceClassifier
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
            from OCC.Core.gp import gp_Pnt2d

            for i, u in enumerate(u_vals):
                for j, v in enumerate(v_vals):
                    clf = BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), 1e-6)
                    state = clf.State()
                    all_trimming[fi, i * grid_size + j] = 1.0 if state in (TopAbs_IN, TopAbs_ON) else 0.0
        except Exception:
            pass  # Keep default all-1 trimming

        # Base features: area, perimeter, centroid, radius
        try:
            props_g = GProp_GProps()
            brepgprop.SurfaceProperties(face, props_g)
            area = props_g.Mass()
            centroid = props_g.CentreOfMass()

            base_features[fi, 0] = face_types[fi]  # surface_type as float
            base_features[fi, 1] = area
            base_features[fi, 2] = 0.0  # perimeter (expensive, skip)
            base_features[fi, 3] = centroid.X()
            base_features[fi, 4] = centroid.Y()
            base_features[fi, 5] = centroid.Z()
        except Exception:
            base_features[fi, 0] = face_types[fi]

    # Normalize xyz if requested
    if normalize and num_faces > 0:
        pts = all_grids[:, :, :, :3].reshape(-1, 3)
        valid = np.any(pts != 0, axis=1)
        if valid.any():
            center = pts[valid].mean(axis=0)
            scale = np.abs(pts[valid] - center).max() + 1e-6
            all_grids[:, :, :, :3] = (all_grids[:, :, :, :3] - center) / scale
            base_features[:, 3:6] = (base_features[:, 3:6] - center) / scale

    # Build 906-dim features
    uv_flat = all_grids.reshape(num_faces, grid_size * grid_size * 8)  # [N, 800]
    trim_flat = all_trimming  # [N, 100]

    node_features = np.concatenate([base_features, uv_flat, trim_flat], axis=1)  # [N, 906]
    x = torch.tensor(node_features, dtype=torch.float32)
    ft = torch.tensor(face_types, dtype=torch.long)

    # Build adjacency from shared edges
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            edge_hash = exp.Current().__hash__()
            if edge_hash not in edge_to_faces:
                edge_to_faces[edge_hash] = []
            edge_to_faces[edge_hash].append(fi)
            exp.Next()

    src_list, tgt_list = [], []
    for edge_hash, face_indices in edge_to_faces.items():
        for a in face_indices:
            for b in face_indices:
                if a != b:
                    src_list.append(a)
                    tgt_list.append(b)

    if src_list:
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        # Deduplicate
        edge_index = torch.unique(edge_index, dim=1)
        edge_attr = torch.zeros(edge_index.shape[1], 4, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 4, dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        face_types=ft,
    )
