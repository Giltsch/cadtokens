"""
cadtokens.gnn — GNN-based manufacturing feature recognition. ⚠️ Experimental

BRepGNN with 12-class semantic face segmentation:
  stock_wall, stock_cap, hole_wall, hole_floor, pocket_wall, pocket_floor,
  boss_wall, boss_cap, fillet, chamfer, revolve_body, other

Requires: torch, torch-geometric, pythonocc-core
"""

try:
    from cadtokens.gnn.brep_gnn import BRepGNN, MaskedPreTrainingModel
    from cadtokens.gnn.gnn_tools import GNNTools, GNN_TOOL_DEFINITIONS, SEGMENT_NAMES
    HAS_GNN = True
except ImportError:
    HAS_GNN = False

__all__ = ["BRepGNN", "MaskedPreTrainingModel", "GNNTools", "GNN_TOOL_DEFINITIONS", "SEGMENT_NAMES", "HAS_GNN"]
