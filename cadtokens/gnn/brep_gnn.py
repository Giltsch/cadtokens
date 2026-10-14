"""
B-Rep Graph Neural Network for CAD geometry understanding.

Architecture:
    - Structured input encoding (separate base/UV/trim processing)
    - GATv2 message passing with edge features
    - Global attention pooling
    - Task heads for pre-training and downstream

Pre-training objectives (Phase 2, Masterplan):
    1. Masked Node Feature Prediction (GraphMAE-style)
    2. Face Type Classification (auxiliary)
    3. Segment Label Prediction (supervised, where labels exist)

Memory-efficient design:
    - Uses PyG sparse COO edge_index (not dense adjacency matrices)
    - Supports gradient checkpointing
    - Variable-size graphs (no padding needed)
    - ~30M params at d_model=256 → ~120MB model memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch
from typing import Dict, Optional, Tuple
import math


class FeatureNormalizer(nn.Module):
    """
    Normalizes raw 906-dim features using precomputed dataset statistics.
    
    Handles the extreme value ranges in CAD data:
    - area: mean 19M, std 15B → log-scale + standardize
    - UV xyz: std ~3000 → standardize  
    - normals: already -1 to 1 → standardize
    - trimming: binary 0/1 → pass through
    """
    
    def __init__(self):
        super().__init__()
        # Will be populated by set_stats()
        self.register_buffer('base_mean', torch.zeros(6))
        self.register_buffer('base_std', torch.ones(6))
        self.register_buffer('uv_mean', torch.zeros(800))
        self.register_buffer('uv_std', torch.ones(800))
        # Indices of base features to log-scale (area=1, perimeter=2, radius=5)
        self.log_indices = [1, 2, 5]
    
    def set_stats(self, base_mean, base_std, uv_mean, uv_std):
        """Set normalization stats from metadata.json."""
        self.base_mean.copy_(torch.tensor(base_mean, dtype=torch.float32))
        self.base_std.copy_(torch.tensor(base_std, dtype=torch.float32).clamp(min=1e-6))
        self.uv_mean.copy_(torch.tensor(uv_mean, dtype=torch.float32))
        self.uv_std.copy_(torch.tensor(uv_std, dtype=torch.float32).clamp(min=1e-6))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize raw features in-place."""
        x_norm = x.clone()
        
        # Base features: log-scale large values, then standardize
        base = x_norm[:, :6]
        for idx in self.log_indices:
            base[:, idx] = torch.log1p(base[:, idx].abs()) * base[:, idx].sign()
        # Standardize base (after log transform, stats won't match exactly but that's ok)
        # We use a simple approach: just make values ~O(1)
        base[:, 0] = base[:, 0] / 7.0  # surface_type 0-6 → 0-1
        for idx in self.log_indices:
            base[:, idx] = base[:, idx] / 20.0  # log(19M) ≈ 17
        base[:, 3] = base[:, 3] / (self.base_std[3] + 1e-6)  # curvature_mean
        base[:, 4] = base[:, 4] / (self.base_std[4] + 1e-6)  # curvature_gaussian
        x_norm[:, :6] = base
        
        # UV features: standardize
        x_norm[:, 6:806] = (x_norm[:, 6:806] - self.uv_mean) / self.uv_std
        
        # Trimming mask: already 0/1, leave as-is
        
        return x_norm


class StructuredInputEncoder(nn.Module):
    """
    Encodes the 906-dim node features with structure-aware processing.
    
    Instead of a naive MLP on 906 dims, we process each sub-group separately:
    - Base features (6): surface_type, area, perimeter, curvatures, radius
    - UV-Grid (800): 10×10 grid × 8 channels → small CNN/MLP
    - Trimming mask (100): 10×10 binary mask → MLP
    
    This respects the data structure and reduces dimensionality efficiently.
    """
    
    def __init__(self, d_model: int = 256, grid_size: int = 10, uv_channels: int = 8):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.uv_channels = uv_channels
        
        # Base features encoder (6 → 64)
        self.base_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        
        # UV-Grid encoder (10×10×8 → 128)
        # Process as 2D feature map with small CNN
        self.uv_encoder = nn.Sequential(
            nn.Conv2d(uv_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # → [N, 64, 1, 1]
            nn.Flatten(),             # → [N, 64]
            nn.Linear(64, 128),
            nn.GELU(),
        )
        
        # Trimming mask encoder (100 → 64)
        self.trim_encoder = nn.Sequential(
            nn.Linear(100, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Fusion: 64 + 128 + 64 = 256 → d_model
        self.fusion = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [total_nodes, 906] NORMALIZED node features
        Returns:
            [total_nodes, d_model] encoded features
        """
        # Split structured sub-groups
        base = x[:, :6]                    # [N, 6]
        uv_flat = x[:, 6:806]              # [N, 800]
        trim = x[:, 806:906]               # [N, 100]
        
        # Encode base
        base_emb = self.base_encoder(base)  # [N, 64]
        
        # Encode UV-Grid: reshape to [N, 8, 10, 10] for CNN
        uv_grid = uv_flat.view(-1, self.grid_size, self.grid_size, self.uv_channels)
        uv_grid = uv_grid.permute(0, 3, 1, 2)  # [N, 8, 10, 10]
        uv_emb = self.uv_encoder(uv_grid)       # [N, 128]
        
        # Encode trimming mask
        trim_emb = self.trim_encoder(trim)  # [N, 64]
        
        # Fuse
        combined = torch.cat([base_emb, uv_emb, trim_emb], dim=-1)  # [N, 256]
        return self.fusion(combined)  # [N, d_model]


class BRepGNN(nn.Module):
    """
    Graph Neural Network for B-Rep CAD geometry.
    
    Uses GATv2 (Graph Attention Network v2) with:
    - Structured input encoding (base + UV + trim)
    - Edge features (type, length, dihedral, convexity)
    - Residual connections + LayerNorm
    - Global attention pooling
    
    Args:
        d_model: Hidden dimension (default 256)
        num_layers: Number of GATv2 layers (default 6)
        heads: Attention heads per layer (default 8)
        edge_dim: Edge feature dimension (default 4)
        dropout: Dropout rate (default 0.1)
        gradient_checkpointing: Trade compute for memory
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        heads: int = 8,
        edge_dim: int = 4,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        
        # Structured input encoder
        self.input_encoder = StructuredInputEncoder(d_model=d_model)
        
        # Edge feature projection
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.GELU(),
        )
        
        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model // heads,
                    heads=heads,
                    edge_dim=d_model,  # projected edge features
                    dropout=dropout,
                    concat=True,       # heads concatenated → d_model
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.ffns.append(nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            ))
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Global attention pooling
        self.pool_gate = nn.Sequential(
            nn.Linear(d_model, 1),
        )
        self.pool_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:          [total_nodes, 906] raw node features
            edge_index: [2, total_edges] COO format
            edge_attr:  [total_edges, 4] edge features (optional)
            batch:      [total_nodes] graph membership indices
        
        Returns:
            dict with:
                'node_features': [total_nodes, d_model]
                'global_features': [num_graphs, d_model]
        """
        # Encode structured inputs
        h = self.input_encoder(x)  # [N, d_model]
        
        # Encode edge features
        if edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)  # [E, d_model]
        else:
            edge_emb = None
        
        # GATv2 layers with residual + norm
        for i, (conv, norm, ffn) in enumerate(zip(self.convs, self.norms, self.ffns)):
            if self.gradient_checkpointing and self.training:
                h_new = torch.utils.checkpoint.checkpoint(
                    self._conv_block, conv, norm, ffn, h, edge_index, edge_emb,
                    use_reentrant=False
                )
            else:
                h_new = self._conv_block(conv, norm, ffn, h, edge_index, edge_emb)
            h = h_new
        
        h = self.final_norm(h)
        
        # Global attention pooling
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        
        gate = torch.sigmoid(self.pool_gate(h))  # [N, 1]
        h_gated = h * gate
        global_feat = global_add_pool(h_gated, batch)  # [num_graphs, d_model]
        global_feat = self.pool_proj(global_feat)
        
        return {
            'node_features': h,            # [total_nodes, d_model]
            'global_features': global_feat, # [num_graphs, d_model]
        }
    
    def get_node_embeddings(self, data) -> torch.Tensor:
        """
        Convenience method: Extract per-face node embeddings from a PyG Data object.
        
        Used by GeoProjection / CAD-Brain to get per-face tokens.
        
        Args:
            data: PyG Data with x, edge_index, edge_attr (single graph)
        Returns:
            [N_faces, d_model] per-face embeddings (after GATv2 message passing)
        """
        out = self.forward(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None,
            batch=data.batch if hasattr(data, 'batch') and data.batch is not None else None,
        )
        return out['node_features']
    
    def _conv_block(self, conv, norm, ffn, h, edge_index, edge_emb):
        """Single GATv2 block: attention → residual → norm → FFN → residual → norm."""
        # Message passing
        h_attn = conv(h, edge_index, edge_attr=edge_emb)
        h = norm(h + self.dropout(h_attn))
        
        # Feed-forward
        h = norm(h + ffn(h))
        
        return h


class MaskedPreTrainingModel(nn.Module):
    """
    Wrapper for self-supervised pre-training of BRepGNN.
    
    Pre-training tasks:
    1. Masked Feature Reconstruction: Predict masked node features
    2. Face Type Classification: Predict surface type (7 classes)
    3. Segment Prediction: Predict face segment label (8 classes, where available)
    
    Masking strategy:
    - Randomly mask 40% of nodes per graph
    - Replace masked node features with a learnable [MASK] token
    - Reconstruct original features only at masked positions
    """
    
    def __init__(
        self,
        encoder: BRepGNN,
        node_feature_dim: int = 906,
        num_face_types: int = 7,
        num_segments: int = 8,
        mask_ratio: float = 0.4,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.d_model = encoder.d_model
        self.node_feature_dim = node_feature_dim
        
        # Feature normalizer (populated with dataset stats before training)
        self.normalizer = FeatureNormalizer()
        
        # Learnable mask token (replaces masked NORMALIZED features)
        self.mask_token = nn.Parameter(torch.zeros(1, node_feature_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Reconstruction head: predict d_model-dim embedding (NOT raw features)
        # We use Scaled Cosine Error (SCE) in embedding space instead of raw MSE
        self.recon_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
        # Face type head: predict surface type (7 classes)
        self.face_type_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_face_types),
        )
        
        # Segment head: predict segment label (8 classes)
        self.segment_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_segments),
        )
    
    def create_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Create random node mask, ensuring at least 1 node remains unmasked per graph.
        
        Args:
            batch: [total_nodes] graph membership indices
        
        Returns:
            [total_nodes] bool tensor (True = masked)
        """
        device = batch.device
        num_nodes = batch.shape[0]
        num_graphs = batch.max().item() + 1
        
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        for g in range(num_graphs):
            node_idx = (batch == g).nonzero(as_tuple=True)[0]
            n = node_idx.shape[0]
            
            if n <= 1:
                continue  # Don't mask single-node graphs
            
            # Mask ceil(mask_ratio * n) nodes, but leave at least 1 unmasked
            num_mask = min(max(1, int(self.mask_ratio * n)), n - 1)
            perm = torch.randperm(n, device=device)[:num_mask]
            mask[node_idx[perm]] = True
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        face_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masking and loss computation.
        
        Returns:
            dict with losses and metrics
        """
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Normalize features
        x_norm = self.normalizer(x)
        
        # Create mask
        mask = self.create_mask(batch)
        
        # Get target embeddings: encode unmasked normalized features
        with torch.no_grad():
            target_out = self.encoder(x_norm, edge_index, edge_attr, batch)
            target_embeds = target_out['node_features'].detach()  # [N, d_model]
        
        # Apply mask: replace masked node features with mask token
        x_masked = x_norm.clone()
        x_masked[mask] = self.mask_token.expand(mask.sum(), -1)
        
        # Forward through encoder with masked input
        out = self.encoder(x_masked, edge_index, edge_attr, batch)
        node_features = out['node_features']  # [N, d_model]
        
        losses = {}
        metrics = {}
        
        # === Loss 1: Scaled Cosine Error (SCE) on masked embeddings ===
        # GraphMAE-style: predict embedding of masked nodes, not raw features
        # SCE = (1 - cos_sim)^gamma, more stable than MSE on raw features
        if mask.any():
            recon = self.recon_head(node_features[mask])     # [num_masked, d_model]
            target = target_embeds[mask]                      # [num_masked, d_model]
            
            # Scaled Cosine Error (gamma=1 → simple cosine distance)
            cos_sim = F.cosine_similarity(recon, target, dim=-1)  # [num_masked]
            recon_loss = (1.0 - cos_sim).mean()
            
            losses['recon'] = recon_loss
            metrics['recon_cos_sim'] = cos_sim.mean().item()
        
        # === Loss 2: Face Type Classification (all nodes) ===
        if face_types is not None:
            ft_logits = self.face_type_head(node_features)  # [N, 7]
            ft_loss = F.cross_entropy(ft_logits, face_types)
            losses['face_type'] = ft_loss
            
            ft_pred = ft_logits.argmax(dim=-1)
            ft_acc = (ft_pred == face_types).float().mean()
            metrics['face_type_acc'] = ft_acc.item()
        
        # === Loss 3: Segment Prediction (only labeled nodes) ===
        if y is not None:
            labeled = y >= 0
            if labeled.any():
                seg_logits = self.segment_head(node_features[labeled])
                seg_loss = F.cross_entropy(seg_logits, y[labeled])
                losses['segment'] = seg_loss
                
                seg_pred = seg_logits.argmax(dim=-1)
                seg_acc = (seg_pred == y[labeled]).float().mean()
                metrics['segment_acc'] = seg_acc.item()
        
        # === Total Loss ===
        total = losses.get('recon', 0.0)
        if 'face_type' in losses:
            total = total + 0.5 * losses['face_type']
        if 'segment' in losses:
            total = total + 1.0 * losses['segment']
        
        losses['total'] = total
        
        return {
            'losses': losses,
            'metrics': metrics,
            'node_features': out['node_features'],
            'global_features': out['global_features'],
        }
