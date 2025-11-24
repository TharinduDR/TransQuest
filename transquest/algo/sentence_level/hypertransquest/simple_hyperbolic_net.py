import torch
import torch.nn as nn
from transformers import AutoModel
import geoopt
import logging

logger = logging.getLogger(__name__)

class SimpleHyperbolicSiameseNet(nn.Module):
    def __init__(self, model_name, curvature=1.0, dropout=0.1):
        super(SimpleHyperbolicSiameseNet, self).__init__()
        
        # 1. Load Backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        # 2. Curvature (Learnable, but clamped)
        # We use softplus to ensure c > 0
        self.c_param = nn.Parameter(torch.tensor(1.0))
        
        # 3. Projection Head
        # Projects 768-dim Euclidean -> 128-dim Euclidean (bottleneck helps QE)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(), # Tanh is often better for bounded embeddings
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 128) # Reduce dim for Hyperbolic stability
        )
        
       # 4. Scoring Parameters
        # FIX: Change 1.0 to 0.1 to prevent vanishing gradients at start
        self.alpha = nn.Parameter(torch.tensor(0.1)) 
        self.beta = nn.Parameter(torch.tensor(0.0))
        
    @property
    def curvature(self):
        # Ensure curvature stays positive and reasonable (0.1 to 5.0)
        return torch.clamp(torch.nn.functional.softplus(self.c_param), min=0.1, max=5.0)
    
    def get_manifold(self):
        return geoopt.PoincareBall(c=self.curvature)
    
    def get_embedding(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token embedding usually works best for Classification/Regression
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward_siamese(self, source_ids, source_mask, target_ids, target_mask):
        # 1. Get Euclidean CLS embeddings
        u_euc = self.get_embedding(source_ids, source_mask)
        v_euc = self.get_embedding(target_ids, target_mask)
        
        # 2. Project to lower dimension (Euclidean)
        u_proj = self.projection(u_euc)
        v_proj = self.projection(v_euc)
        
        # 3. Map to Hyperbolic Space (ExpMap)
        manifold = self.get_manifold()
        u_hyp = manifold.expmap0(u_proj)
        v_hyp = manifold.expmap0(v_proj)
        
        # 4. Calculate Hyperbolic Distance
        # Add small epsilon to avoid NaN gradients at distance 0
        dist = manifold.dist(u_hyp, v_hyp) + 1e-6
        
        # 5. Convert Distance to Score (0 to 1)
        # We want: High Distance = Low Score (0). Low Distance = High Score (1).
        # Formula: score = 2 * sigmoid(-alpha * dist) 
        # Or simpler: exp(-dist) which is naturally 0..1
        
        # Using the exponential decay wrapper with learnable scaling
        score = torch.exp(-self.alpha * dist)
        
        # Debug logging (occasional)
        if self.training and torch.rand(1).item() < 0.005:
            logger.info(f"Stats - Dist: {dist.mean():.3f}, Score: {score.mean():.3f}, c: {self.curvature.item():.3f}")
        
        return score