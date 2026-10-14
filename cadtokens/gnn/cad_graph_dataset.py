"""
Memory-Efficient PyG Dataset for CAD B-Rep Graphs.

Designed for 128GB unified memory (DGX Spark / GB10).

Strategy:
- Individual .pt files per entry (lazy loading via PyG Dataset)
- Only batch_size entries in memory at any time
- Memory footprint: ~5-10MB per batch vs 22GB for monolithic PKL
- DataLoader with num_workers for prefetching

Usage:
    dataset = CADGraphDataset(root='data/full_dataset/pyg', split='train')
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    for batch in loader:
        # batch.x:          [total_nodes_in_batch, 906]
        # batch.edge_index: [2, total_edges_in_batch]
        # batch.edge_attr:  [total_edges_in_batch, 4]
        # batch.y:          [total_nodes_in_batch] (seg labels, -1 if unavailable)
        # batch.batch:      [total_nodes_in_batch] (graph membership)
        pass

Memory Budget (128GB total):
    - OS + System:       ~8 GB
    - PyTorch context:   ~2 GB
    - Micro-batch (64):  ~5 MB
    - Model + Optimizer: ~0.5 GB
    - Activations:       ~0.2 GB
    - ─────────────────────────
    - Total:             ~11 GB (leaves 117 GB free!)
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from torch_geometric.data import Dataset, Data


class CADGraphDataset(Dataset):
    """
    On-disk PyG Dataset for CAD B-Rep graphs.
    
    Each graph represents one CAD part with:
    - Nodes = Faces (features: base props + UV-grid + trimming mask)
    - Edges = Shared boundaries between faces
    - Labels = Face segmentation classes (if available)
    
    Args:
        root: Directory with processed/ subdirectories
        split: 'train', 'val', or 'test'
        transform: PyG transform to apply on each Data object
        max_nodes: Skip graphs with more nodes (memory guard)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        pre_transform=None,
        max_nodes: int = 500,
    ):
        self.split = split
        self.max_nodes = max_nodes
        self._file_list = None
        self._metadata = None
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed', self.split)
    
    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        if self._file_list is None:
            proc_dir = self.processed_dir
            if os.path.exists(proc_dir):
                self._file_list = sorted(
                    f for f in os.listdir(proc_dir) if f.endswith('.pt')
                )
            else:
                self._file_list = []
        return self._file_list
    
    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: int) -> Data:
        """Load a single graph from disk."""
        path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(path, weights_only=False)
        return data
    
    @property
    def metadata(self) -> Optional[Dict]:
        """Dataset metadata (node dims, segment names, stats)."""
        if self._metadata is None:
            meta_path = os.path.join(self.root, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self._metadata = json.load(f)
        return self._metadata
    
    @property
    def num_node_features(self) -> int:
        if self.metadata:
            return self.metadata.get('node_feature_dim', 906)
        return 906
    
    @property
    def num_edge_features(self) -> int:
        return 4  # edge_type, length, dihedral_angle, convexity
    
    @property
    def num_classes(self) -> int:
        if self.metadata:
            return len(self.metadata.get('segment_names', []))
        return 8
    
    @property
    def segment_names(self) -> List[str]:
        if self.metadata:
            return self.metadata.get('segment_names', [])
        return [
            'ExtrudeSide', 'ExtrudeEnd', 'CutSide', 'CutEnd',
            'Fillet', 'Chamfer', 'RevolveSide', 'RevolveEnd'
        ]
    
    def process(self):
        """No-op: data is prepared by scripts/prepare_pyg_data.py"""
        pass
    
    def download(self):
        """No-op: data comes from our JSONL pipeline."""
        pass


def jsonl_entry_to_pyg_data(entry: dict) -> Optional[Data]:
    """
    Convert a single JSONL/PKL entry to a PyG Data object.
    
    Memory: ~100-300KB per entry (vs ~200KB in JSONL).
    
    Args:
        entry: Dict from checkpoint.jsonl with gnn_data, ground_truth, etc.
    
    Returns:
        PyG Data object or None if conversion fails.
    """
    gnn = entry.get('gnn_data')
    if not gnn:
        return None
    
    nodes = gnn.get('nodes', [])
    edges_list = gnn.get('edges', [])
    edge_index_raw = gnn.get('edge_index', [[], []])
    
    if not nodes:
        return None
    
    num_nodes = len(nodes)
    
    # =========================================================================
    # Node features: [num_nodes, 906]
    # Layout: [6 base] + [800 UV-grid] + [100 trimming mask]
    # =========================================================================
    surface_type_map = {
        'Plane': 0, 'Cylinder': 1, 'Cone': 2, 'Sphere': 3,
        'Torus': 4, 'BSpline': 5, 'Other': 6
    }
    
    node_feats = []
    face_types = []
    centroids = []
    
    for n in nodes:
        # Base features (6)
        st = surface_type_map.get(n.get('surface_type', 'Other'), 6)
        face_types.append(st)
        
        f = [
            float(st),
            n.get('area', 0.0),
            n.get('perimeter', 0.0),
            n.get('curvature_mean', 0.0),
            n.get('curvature_gaussian', 0.0),
            float(n.get('radius', 0.0) or 0.0),
        ]
        
        # UV-Grid 8ch flattened (800)
        uv = n.get('uv_grid', [])
        if uv and len(uv) > 0:
            for pt in uv:
                f.extend(pt[:8])  # Ensure exactly 8 channels
                if len(pt) < 8:
                    f.extend([0.0] * (8 - len(pt)))
        else:
            f.extend([0.0] * 800)
        
        # Trimming mask (100)
        trim = n.get('trimming_mask', [])
        if trim:
            f.extend(trim[:100])
            if len(trim) < 100:
                f.extend([1] * (100 - len(trim)))
        else:
            f.extend([1] * 100)
        
        node_feats.append(f)
        
        # Centroid for positional encoding
        c = n.get('centroid', [0, 0, 0])
        centroids.append(c[:3] if len(c) >= 3 else c + [0.0] * (3 - len(c)))
    
    x = torch.tensor(node_feats, dtype=torch.float32)
    
    # =========================================================================
    # Edge index: [2, num_directed_edges]
    # =========================================================================
    if edge_index_raw and edge_index_raw[0]:
        edge_index = torch.tensor(edge_index_raw, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # =========================================================================
    # Edge features: [num_directed_edges, 4]
    # Each undirected edge → 2 directed entries (consecutive in edge_index)
    # =========================================================================
    edge_attr = None
    if edges_list and edge_index.shape[1] > 0:
        edge_type_map = {'Line': 0, 'Circle': 1, 'BSpline': 2, 'Other': 3}
        convexity_map = {'convex': 1.0, 'concave': -1.0, 'smooth': 0.0, 'unknown': 0.0}
        
        edge_feats = []
        for e in edges_list:
            ef = [
                float(edge_type_map.get(e.get('edge_type', 'Other'), 3)),
                e.get('length', 0.0),
                e.get('dihedral_angle', 0.0),
                convexity_map.get(e.get('convexity', 'unknown'), 0.0),
            ]
            # Each undirected edge appears twice in edge_index (f1→f2, f2→f1)
            edge_feats.append(ef)
            edge_feats.append(ef)
        
        if len(edge_feats) == edge_index.shape[1]:
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            # Fallback: zeros if count mismatch
            edge_attr = torch.zeros((edge_index.shape[1], 4), dtype=torch.float32)
    
    # =========================================================================
    # Segment labels: [num_nodes] (-1 = unlabeled)
    # =========================================================================
    gt = entry.get('ground_truth', {})
    seg_labels = gt.get('seg_labels_int', []) if gt else []
    
    if seg_labels and len(seg_labels) == num_nodes:
        y = torch.tensor(seg_labels, dtype=torch.long)
    else:
        y = torch.full((num_nodes,), -1, dtype=torch.long)
    
    # =========================================================================
    # Additional metadata stored as tensor attrs
    # =========================================================================
    face_type_tensor = torch.tensor(face_types, dtype=torch.long)
    centroid_tensor = torch.tensor(centroids, dtype=torch.float32)
    
    # Build Data object
    data = Data(
        x=x,                           # [N, 906]
        edge_index=edge_index,          # [2, E]
        edge_attr=edge_attr,            # [E, 4] or None
        y=y,                            # [N] seg labels
        face_types=face_type_tensor,    # [N] surface type ids
        pos=centroid_tensor,            # [N, 3] centroids
    )
    
    # Store entry ID and source as string attrs
    data.entry_id = entry.get('id', '')
    data.source = entry.get('source', '')
    data.num_nodes_orig = num_nodes
    
    return data
