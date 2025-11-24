"""
Hybrid Hyperbolic Network for Translation Quality Estimation.
Uses hyperbolic space for representation learning but cosine similarity for stability.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import geoopt
import logging

logger = logging.getLogger(__name__)

class HybridHyperbolicSiameseNet(nn.Module):
    """
    Hybrid approach: Use hyperbolic space for representation learning,
    but compute similarity using cosine distance for stability.
    """
    def __init__(self, model_name, curvature=1.0, dropout=0.2):
        super(HybridHyperbolicSiameseNet, self).__init__()
        
        # 1. Load Backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        # 2. Simple learnable curvature
        self.curvature = nn.Parameter(torch.tensor(curvature))
        
        # 3. Projection head for hyperbolic representation
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # 4. Learnable score parameters
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Temperature for similarity
        self.bias = nn.Parameter(torch.tensor(0.0))         # Bias for score transformation
        
    def get_curvature(self):
        return torch.clamp(self.curvature, 0.1, 2.0)
    
    def get_manifold(self):
        return geoopt.PoincareBall(c=self.get_curvature())
    
    def get_embedding(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward_siamese(self, source_ids, source_mask, target_ids, target_mask):
        # 1. Get embeddings
        u_euc = self.get_embedding(source_ids, source_mask)
        v_euc = self.get_embedding(target_ids, target_mask)
        
        # 2. Project embeddings
        u_proj = self.projection(u_euc)
        v_proj = self.projection(v_euc)
        
        # 3. Map to hyperbolic space for representation learning
        manifold = self.get_manifold()
        u_hyperbolic = manifold.expmap0(u_proj)
        v_hyperbolic = manifold.expmap0(v_proj)
        
        # 4. BUT: Use cosine similarity for final score (more stable)
        u_normalized = torch.nn.functional.normalize(u_hyperbolic, dim=1)
        v_normalized = torch.nn.functional.normalize(v_hyperbolic, dim=1)
        
        # Compute cosine similarity (higher = more similar)
        cosine_sim = torch.sum(u_normalized * v_normalized, dim=1)
        
        # Convert to [0,1] range with learnable parameters
        score = torch.sigmoid(self.temperature * cosine_sim + self.bias)
        
        # Debug logging
        if self.training and torch.rand(1).item() < 0.01:
            logger.info(f"Training stats - Cosine sim: {cosine_sim.mean().item():.4f}±{cosine_sim.std().item():.4f}")
            logger.info(f"Training stats - Score: {score.mean().item():.4f}±{score.std().item():.4f}")
            logger.info(f"Training stats - Temperature: {self.temperature.item():.4f}, Bias: {self.bias.item():.4f}")
        
        return score


class FixedHyperbolicSiameseNet(HybridHyperbolicSiameseNet):
    """
    Backwards compatible wrapper that maintains the original interface.
    """
    def __init__(self, model_name, curvature=1.0, dropout=0.1):
        super().__init__(model_name, curvature=curvature, dropout=dropout)
        # For compatibility with existing code
        self.scale_factor = 0.1