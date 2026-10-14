#!/usr/bin/env python3
"""
GNN-Enhanced Geometry Tools — Deep B-Rep analysis via pre-trained BRepGNN.

Provides rich semantic analysis beyond basic OCC measurements:
- Face classification with confidence scores
- Segment clustering (which faces form a feature group)
- Face similarity search via embedding distances
- Feature recognition (holes, pockets, bosses, fillets)
- Global shape fingerprint for part comparison

Uses the pre-trained MaskedPreTrainingModel with face_type_head and segment_head.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


from cadtokens.gnn.brep_gnn import BRepGNN, MaskedPreTrainingModel
from cadtokens.gnn.shape_to_graph import shape_to_pyg
from cadtokens.tools.geometry_tools import GeometryTools
from cadtokens.gnn.cad_graph_dataset import jsonl_entry_to_pyg_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Surface type labels (matches BRepGNN training)
FACE_TYPE_NAMES = {
    0: "Plane",
    1: "Cylinder",
    2: "Cone",
    3: "Sphere",
    4: "Torus",
    5: "BSpline",
    6: "Other",
}

# Semantic segment labels — 12-class scheme from semantic_seg pipeline
# Combines operation type + geometric role for richer understanding
SEGMENT_NAMES_12 = {
    0: "stock_wall",      # Main body wall (ExtrudeSide + NewBody)
    1: "stock_cap",       # Main body top/bottom (ExtrudeEnd + NewBody)
    2: "hole_wall",       # Hole cylindrical wall (CutSide + Cylinder)
    3: "hole_floor",      # Hole bottom (CutEnd near Cylinder)
    4: "pocket_wall",     # Pocket side wall (CutSide + Plane)
    5: "pocket_floor",    # Pocket bottom (CutEnd + Plane)
    6: "boss_wall",       # Added material wall (Extrude/Revolve + Join)
    7: "boss_cap",        # Added material cap (ExtrudeEnd + Join)
    8: "fillet",          # Fillet/round
    9: "chamfer",         # Chamfer/bevel
    10: "revolve_body",   # Revolve base body (Revolve + NewBody)
    11: "other",          # Unclassifiable
}

# Legacy 8-class operation-based labels (s2.0.1 ground truth)
SEGMENT_NAMES_8 = {
    0: "ExtrudeSide", 1: "ExtrudeEnd", 2: "CutSide", 3: "CutEnd",
    4: "Fillet", 5: "Chamfer", 6: "RevolveSide", 7: "RevolveEnd",
}

# Active label map — set at runtime based on loaded model
SEGMENT_NAMES = SEGMENT_NAMES_12


class GNNTools:
    """
    GNN-enhanced CAD analysis — uses pre-trained BRepGNN for deep understanding.
    
    Unlike basic OCC tools (which give raw geometry: area, type, radius),
    the GNN captures topological context: how faces relate, what functional
    role they play, and what manufacturing features they form.
    """
    
    def __init__(self, checkpoint_path: str = "checkpoints/gnn_pretrain/best_semantic.pt",
                 fallback_path: str = "checkpoints/gnn_pretrain/best.pt"):
        self.model: Optional[MaskedPreTrainingModel] = None
        self.encoder: Optional[BRepGNN] = None
        self.checkpoint_path = checkpoint_path
        self.fallback_path = fallback_path
        self._graph_cache: Dict[str, Any] = {}
        self._embedding_cache: Dict[str, Dict] = {}
        self._num_segments = 12  # default to semantic
        self._geo_tools = GeometryTools()  # for training-consistent feature extraction
        
    def load(self) -> bool:
        """Load the pre-trained GNN model with all heads."""
        try:
            # Build encoder
            self.encoder = BRepGNN(
                d_model=256, num_layers=6, heads=8,
                edge_dim=4, dropout=0.0,
            )
            
            # Determine which checkpoint to load
            import os
            ckpt_path = self.checkpoint_path
            if not os.path.exists(ckpt_path):
                print(f"  Semantic checkpoint not found, falling back to {self.fallback_path}")
                ckpt_path = self.fallback_path
                self._num_segments = 8
            
            # Load checkpoint to determine num_segments
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            
            # Detect num_segments from checkpoint
            last_seg_key = None
            for k in sorted(state_dict.keys()):
                if k.startswith("segment_head") and k.endswith(".weight"):
                    last_seg_key = k
            if last_seg_key:
                self._num_segments = state_dict[last_seg_key].shape[0]
            else:
                self._num_segments = 8  # fallback
            
            print(f"  Loading checkpoint: {ckpt_path} ({self._num_segments} segment classes)")
            
            # Build full pre-training model (encoder + heads)
            self.model = MaskedPreTrainingModel(
                encoder=self.encoder,
                node_feature_dim=906,
                num_face_types=7,
                num_segments=8,  # Original head size
                mask_ratio=0.0,
            )
            
            # If semantic (12-class), replace segment head before loading
            if self._num_segments != 8:
                import torch.nn as nn
                self.model.segment_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, self._num_segments),
                )
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval().to(DEVICE)
            
            # Set active SEGMENT_NAMES based on loaded model
            global SEGMENT_NAMES
            if self._num_segments == 12:
                SEGMENT_NAMES = SEGMENT_NAMES_12
            else:
                SEGMENT_NAMES = SEGMENT_NAMES_8
            
            print(f"  ✓ GNN loaded: {self._num_segments}-class segmentation")
            return True
        except Exception as e:
            import traceback
            print(f"GNN load error: {e}")
            traceback.print_exc()
            return False
    
    def _get_graph(self, shape_or_path) -> Optional[Any]:
        """Convert OCC shape/STEP path → PyG graph (cached).
        
        Uses the SAME feature extraction pipeline as training
        (GeometryTools.get_gnn_data → jsonl_entry_to_pyg_data) to ensure
        consistent node features, edge attributes, and topology.
        """
        # Accept either a shape object or a step path string
        if isinstance(shape_or_path, str):
            cache_key = shape_or_path
        else:
            cache_key = id(shape_or_path)
        
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        
        graph = None
        
        # Try training-consistent pipeline (requires step_path)
        step_path = shape_or_path if isinstance(shape_or_path, str) else None
        if step_path is None:
            # Check if GeometryTools has this shape cached with a path
            for path, cached_shape in self._geo_tools._cache.items():
                if cached_shape is shape_or_path:
                    step_path = path
                    break
        
        if step_path is not None:
            try:
                gnn_data = self._geo_tools.get_gnn_data(step_path)
                entry = {"gnn_data": gnn_data, "ground_truth": {}}
                graph = jsonl_entry_to_pyg_data(entry)
            except Exception as e:
                print(f"  Training pipeline failed: {e}, falling back to shape_to_pyg")
        
        # Fallback to shape_to_pyg (less accurate but works without step_path)
        if graph is None:
            shape = shape_or_path
            if isinstance(shape_or_path, str):
                from OCC.Core.STEPControl import STEPControl_Reader
                from OCC.Core.IFSelect import IFSelect_RetDone
                reader = STEPControl_Reader()
                if reader.ReadFile(shape_or_path) == IFSelect_RetDone:
                    reader.TransferRoots()
                    shape = reader.OneShape()
                else:
                    return None
            graph = shape_to_pyg(shape, grid_size=10, normalize=False)
        
        self._graph_cache[cache_key] = graph
        return self._graph_cache[cache_key]
    
    @torch.no_grad()
    def _compute_embeddings(self, shape_or_path) -> Optional[Dict]:
        """
        Run GNN on shape/step_path → per-face embeddings + classifications.
        Returns cached results.
        """
        cache_key = shape_or_path if isinstance(shape_or_path, str) else id(shape_or_path)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if self.model is None:
            return None
        
        graph = self._get_graph(shape_or_path)
        if graph is None:
            return None
        
        graph = graph.to(DEVICE)
        
        # Forward through encoder
        out = self.encoder(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr if hasattr(graph, "edge_attr") else None,
            batch=graph.batch if hasattr(graph, "batch") else None,
        )
        
        node_features = out["node_features"]       # [N, 256]
        global_features = out["global_features"]   # [1, 256]
        
        # Face type classification
        ft_logits = self.model.face_type_head(node_features)   # [N, 7]
        ft_probs = F.softmax(ft_logits, dim=-1)
        ft_preds = ft_probs.argmax(dim=-1)
        
        # Segment classification
        seg_logits = self.model.segment_head(node_features)    # [N, 8]
        seg_probs = F.softmax(seg_logits, dim=-1)
        seg_preds = seg_probs.argmax(dim=-1)
        
        result = {
            "node_embeddings": node_features.cpu(),
            "global_embedding": global_features.cpu().squeeze(0),
            "face_type_preds": ft_preds.cpu().numpy(),
            "face_type_probs": ft_probs.cpu().numpy(),
            "segment_preds": seg_preds.cpu().numpy(),
            "segment_probs": seg_probs.cpu().numpy(),
            "num_faces": node_features.shape[0],
        }
        
        self._embedding_cache[cache_key] = result
        return result
    
    # ─── Public Tool Methods ─────────────────────────────────────────────
    
    def classify_faces(self, shape) -> Dict[str, Any]:
        """
        Classify every face by surface type using GNN (with context-awareness).
        
        Unlike basic OCC surface type detection, the GNN considers
        topological context (neighbors, edge relationships) which can
        disambiguate borderline cases.
        
        Returns:
            {faces: [{id, predicted_type, confidence, ...}], summary: {...}}
        """
        emb = self._compute_embeddings(shape)
        if emb is None:
            return {"error": "GNN not loaded or graph extraction failed"}
        
        faces = []
        type_counts = {}
        
        for i in range(emb["num_faces"]):
            pred = int(emb["face_type_preds"][i])
            prob = float(emb["face_type_probs"][i, pred])
            type_name = FACE_TYPE_NAMES.get(pred, "Unknown")
            
            # Top-3 predictions
            top3_idx = np.argsort(emb["face_type_probs"][i])[::-1][:3]
            top3 = [
                {"type": FACE_TYPE_NAMES.get(int(idx), "?"), "confidence": float(emb["face_type_probs"][i, idx])}
                for idx in top3_idx
            ]
            
            faces.append({
                "face_id": i,
                "predicted_type": type_name,
                "confidence": round(prob, 3),
                "top3": top3,
            })
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "faces": faces,
            "summary": {
                "total_faces": emb["num_faces"],
                "type_distribution": type_counts,
                "avg_confidence": round(float(np.mean([f["confidence"] for f in faces])), 3),
            },
        }
    
    def detect_features(self, shape) -> Dict[str, Any]:
        """
        Detect manufacturing features by segment classification.
        
        Groups faces into functional features: holes, pockets, bosses,
        fillets, chamfers, etc. This goes beyond basic surface type detection
        by understanding the functional role of face groups.
        
        Returns:
            {features: [{type, face_ids, confidence, ...}], summary: {...}}
        """
        emb = self._compute_embeddings(shape)
        if emb is None:
            return {"error": "GNN not loaded or graph extraction failed"}
        
        # Group faces by predicted segment
        segment_groups: Dict[int, List[Dict]] = {}
        
        for i in range(emb["num_faces"]):
            seg = int(emb["segment_preds"][i])
            prob = float(emb["segment_probs"][i, seg])
            
            if seg not in segment_groups:
                segment_groups[seg] = []
            segment_groups[seg].append({
                "face_id": i,
                "confidence": prob,
                "face_type": FACE_TYPE_NAMES.get(int(emb["face_type_preds"][i]), "?"),
            })
        
        features = []
        for seg_id, face_list in sorted(segment_groups.items()):
            seg_name = SEGMENT_NAMES.get(seg_id, f"segment_{seg_id}")
            avg_conf = np.mean([f["confidence"] for f in face_list])
            face_types = {}
            for f in face_list:
                ft = f["face_type"]
                face_types[ft] = face_types.get(ft, 0) + 1
            
            features.append({
                "feature_type": seg_name,
                "face_ids": [f["face_id"] for f in face_list],
                "num_faces": len(face_list),
                "avg_confidence": round(float(avg_conf), 3),
                "face_type_composition": face_types,
            })
        
        # Summary
        feature_counts = {f["feature_type"]: f["num_faces"] for f in features}
        
        return {
            "features": features,
            "summary": {
                "total_faces": emb["num_faces"],
                "num_feature_groups": len(features),
                "feature_distribution": feature_counts,
            },
        }
    
    def find_similar_faces(self, shape, face_id: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Find faces most similar to a given face using GNN embeddings.
        
        Uses cosine similarity in the learned 256-dim embedding space —
        faces that play similar functional roles (even with different geometry)
        will have similar embeddings.
        
        Returns:
            {query_face: {...}, similar_faces: [{face_id, similarity, ...}]}
        """
        emb = self._compute_embeddings(shape)
        if emb is None:
            return {"error": "GNN not loaded"}
        
        if face_id < 0 or face_id >= emb["num_faces"]:
            return {"error": f"Face {face_id} out of range (0-{emb['num_faces']-1})"}
        
        embeddings = emb["node_embeddings"]  # [N, 256]
        query = embeddings[face_id].unsqueeze(0)  # [1, 256]
        
        # Cosine similarity
        sims = F.cosine_similarity(query, embeddings, dim=-1)  # [N]
        sims[face_id] = -1  # Exclude self
        
        top_k = min(top_k, emb["num_faces"] - 1)
        top_idx = torch.topk(sims, top_k).indices.numpy()
        
        query_info = {
            "face_id": face_id,
            "face_type": FACE_TYPE_NAMES.get(int(emb["face_type_preds"][face_id]), "?"),
            "segment": SEGMENT_NAMES.get(int(emb["segment_preds"][face_id]), "?"),
        }
        
        similar = []
        for idx in top_idx:
            idx = int(idx)
            similar.append({
                "face_id": idx,
                "similarity": round(float(sims[idx]), 3),
                "face_type": FACE_TYPE_NAMES.get(int(emb["face_type_preds"][idx]), "?"),
                "segment": SEGMENT_NAMES.get(int(emb["segment_preds"][idx]), "?"),
            })
        
        return {"query_face": query_info, "similar_faces": similar}
    
    def get_face_embedding_info(self, shape, face_id: int) -> Dict[str, Any]:
        """
        Get detailed GNN analysis for a specific face.
        
        Returns embedding-based analysis: predicted type with confidence,
        predicted feature segment, and the top alternative classifications.
        """
        emb = self._compute_embeddings(shape)
        if emb is None:
            return {"error": "GNN not loaded"}
        
        if face_id < 0 or face_id >= emb["num_faces"]:
            return {"error": f"Face {face_id} out of range"}
        
        # Face type distribution
        ft_probs = emb["face_type_probs"][face_id]
        ft_ranking = [
            {"type": FACE_TYPE_NAMES.get(i, "?"), "probability": round(float(ft_probs[i]), 3)}
            for i in np.argsort(ft_probs)[::-1]
            if ft_probs[i] > 0.01
        ]
        
        # Segment distribution
        seg_probs = emb["segment_probs"][face_id]
        seg_ranking = [
            {"segment": SEGMENT_NAMES.get(i, "?"), "probability": round(float(seg_probs[i]), 3)}
            for i in np.argsort(seg_probs)[::-1]
            if seg_probs[i] > 0.01
        ]
        
        return {
            "face_id": face_id,
            "predicted_type": FACE_TYPE_NAMES.get(int(emb["face_type_preds"][face_id]), "?"),
            "type_confidence": round(float(ft_probs.max()), 3),
            "type_ranking": ft_ranking,
            "predicted_segment": SEGMENT_NAMES.get(int(emb["segment_preds"][face_id]), "?"),
            "segment_confidence": round(float(seg_probs.max()), 3),
            "segment_ranking": seg_ranking,
        }
    
    def get_shape_fingerprint(self, shape) -> Dict[str, Any]:
        """
        Compute a global shape fingerprint using the GNN.
        
        The 256-dim global embedding captures the overall topology, complexity,
        and structure of the part. Useful for part similarity search and
        classification.
        
        Returns:
            {fingerprint: [256 floats], summary: {...}}
        """
        emb = self._compute_embeddings(shape)
        if emb is None:
            return {"error": "GNN not loaded"}
        
        global_emb = emb["global_embedding"].numpy()
        
        # Statistics about the embedding
        return {
            "fingerprint_dim": 256,
            "fingerprint_norm": round(float(np.linalg.norm(global_emb)), 3),
            "num_faces": emb["num_faces"],
            "face_type_distribution": dict(
                zip(
                    [FACE_TYPE_NAMES[i] for i in range(7)],
                    [int((emb["face_type_preds"] == i).sum()) for i in range(7)],
                )
            ),
            "feature_distribution": dict(
                zip(
                    [SEGMENT_NAMES.get(i, f"seg_{i}") for i in range(self._num_segments)],
                    [int((emb["segment_preds"] == i).sum()) for i in range(self._num_segments)],
                )
            ),
        }
    
    def get_smart_summary(self, shape) -> Dict[str, Any]:
        """
        Generate an intelligent summary combining GNN analysis with structure.
        
        This is the main entry point for LLM integration — provides a rich
        structured description that an LLM can reason about.
        """
        emb = self._compute_embeddings(shape)
        if emb is None:
            return {"error": "GNN not loaded"}
        
        nf = emb["num_faces"]
        
        # Face type distribution
        ft_dist = {}
        for i in range(nf):
            t = FACE_TYPE_NAMES.get(int(emb["face_type_preds"][i]), "?")
            ft_dist[t] = ft_dist.get(t, 0) + 1
        
        # Feature analysis
        seg_dist = {}
        seg_faces = {}
        for i in range(nf):
            s = SEGMENT_NAMES.get(int(emb["segment_preds"][i]), "?")
            seg_dist[s] = seg_dist.get(s, 0) + 1
            if s not in seg_faces:
                seg_faces[s] = []
            seg_faces[s].append(i)
        
        def face_info(fid):
            return {"face_id": fid, "face_type": FACE_TYPE_NAMES.get(int(emb["face_type_preds"][fid]), "?")}
        
        # Semantic 12-class feature detection (direct from labels)
        holes = [face_info(fid) for fid in seg_faces.get("hole_wall", []) + seg_faces.get("hole_floor", [])]
        pockets = [face_info(fid) for fid in seg_faces.get("pocket_wall", []) + seg_faces.get("pocket_floor", [])]
        bosses = [face_info(fid) for fid in seg_faces.get("boss_wall", []) + seg_faces.get("boss_cap", [])]
        fillets = [face_info(fid) for fid in seg_faces.get("fillet", [])]
        chamfers = [face_info(fid) for fid in seg_faces.get("chamfer", [])]
        revolve_parts = [face_info(fid) for fid in seg_faces.get("revolve_body", [])]
        stock = [face_info(fid) for fid in seg_faces.get("stock_wall", []) + seg_faces.get("stock_cap", [])]
        
        # Confidence stats
        ft_confs = [float(emb["face_type_probs"][i].max()) for i in range(nf)]
        seg_confs = [float(emb["segment_probs"][i].max()) for i in range(nf)]
        
        return {
            "num_faces": nf,
            "face_types": ft_dist,
            "feature_segments": seg_dist,
            "holes": holes,                    # hole_wall + hole_floor
            "pockets": pockets,                # pocket_wall + pocket_floor
            "bosses": bosses,                  # boss_wall + boss_cap (added material)
            "detected_fillets": fillets,
            "detected_chamfers": chamfers,
            "revolve_features": revolve_parts,
            "stock_body": stock,               # stock_wall + stock_cap (base body)
            "confidence": {
                "face_type_avg": round(float(np.mean(ft_confs)), 3),
                "face_type_min": round(float(np.min(ft_confs)), 3),
                "segment_avg": round(float(np.mean(seg_confs)), 3),
                "segment_min": round(float(np.min(seg_confs)), 3),
            },
        }
    
    def clear_cache(self):
        """Clear cached graphs and embeddings."""
        self._graph_cache.clear()
        self._embedding_cache.clear()


# ─── Tool definitions for LLM integration ────────────────────────────────────

GNN_TOOL_DEFINITIONS = [
    {
        "name": "gnn_classify_faces",
        "description": "Klassifiziert alle Faces des Bauteils mit dem GNN Neural Network. Erkennt Flächentypen (Plane, Cylinder, Cone, Sphere, Torus, BSpline) mit Konfidenz-Scores. Genauer als einfache OCC-Erkennung weil topologischer Kontext berücksichtigt wird.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {"type": "string", "description": "Pfad zur STEP-Datei"}
            },
            "required": ["step_path"],
        },
    },
    {
        "name": "gnn_detect_features",
        "description": "Erkennt Manufacturing Features im Bauteil: Bohrungen (through_hole, blind_hole), Taschen (pocket), Erhebungen (boss), Verrundungen (fillet), Fasen (chamfer). Gruppiert Faces nach ihrer funktionalen Rolle.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {"type": "string", "description": "Pfad zur STEP-Datei"}
            },
            "required": ["step_path"],
        },
    },
    {
        "name": "gnn_find_similar_faces",
        "description": "Findet Faces die einer bestimmten Face am ähnlichsten sind (über GNN Embedding-Distanz). Nützlich um wiederholte Features zu finden (z.B. alle Bohrungen gleicher Art).",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {"type": "string", "description": "Pfad zur STEP-Datei"},
                "face_id": {"type": "integer", "description": "ID der Referenz-Face"},
                "top_k": {"type": "integer", "description": "Anzahl ähnlicher Faces (default: 5)"},
            },
            "required": ["step_path", "face_id"],
        },
    },
    {
        "name": "gnn_face_info",
        "description": "Detaillierte GNN-Analyse einer einzelnen Face: Typ-Wahrscheinlichkeiten, Feature-Segment-Klassifikation, Ranking aller möglichen Typen.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {"type": "string", "description": "Pfad zur STEP-Datei"},
                "face_id": {"type": "integer", "description": "ID der Face"},
            },
            "required": ["step_path", "face_id"],
        },
    },
    {
        "name": "gnn_smart_summary",
        "description": "Intelligente GNN-Zusammenfassung des Bauteils: erkennt alle Features (Bohrungen, Taschen, Verrundungen, etc.), Face-Typ-Verteilung, und Konfidenz-Werte. Die beste Übersicht über die Bauteil-Struktur.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {"type": "string", "description": "Pfad zur STEP-Datei"}
            },
            "required": ["step_path"],
        },
    },
    {
        "name": "gnn_shape_fingerprint",
        "description": "Berechnet einen globalen Shape-Fingerprint (256-dim Embedding). Zeigt die Gesamtstruktur des Teils: Face-Typ-Verteilung, Feature-Verteilung, Komplexität.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_path": {"type": "string", "description": "Pfad zur STEP-Datei"}
            },
            "required": ["step_path"],
        },
    },
]
