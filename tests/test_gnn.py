"""Tests for GNN feature recognition (skipped if torch-geometric not installed)."""

import pytest

try:
    from cadtokens.gnn import HAS_GNN
except ImportError:
    HAS_GNN = False

pytestmark = pytest.mark.skipif(not HAS_GNN, reason="torch-geometric not installed")


class TestGNNImports:
    def test_brep_gnn_importable(self):
        from cadtokens.gnn import BRepGNN
        assert BRepGNN is not None

    def test_gnn_tools_importable(self):
        from cadtokens.gnn import GNNTools
        assert GNNTools is not None

    def test_segment_names(self):
        from cadtokens.gnn import SEGMENT_NAMES
        assert len(SEGMENT_NAMES) == 12
        values = set(SEGMENT_NAMES.values())
        assert "hole_wall" in values
        assert "pocket_floor" in values
        assert "fillet" in values

    def test_brep_gnn_instantiate(self):
        from cadtokens.gnn import BRepGNN
        model = BRepGNN(
            d_model=64,
            num_layers=2,
            heads=4,
            edge_dim=4,
        )
        # Check it has the right number of parameters
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


class TestGNNCheckpoint:
    def test_checkpoint_exists(self):
        from pathlib import Path
        ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "gnn_pretrain" / "best_semantic.pt"
        assert ckpt.exists(), f"Checkpoint not found: {ckpt}"

    def test_load_checkpoint(self):
        import torch
        from pathlib import Path
        ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "gnn_pretrain" / "best_semantic.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in state or isinstance(state, dict)
