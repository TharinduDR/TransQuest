import torch
import torch.nn as nn
from transformers import AutoModel
import geoopt

class FiLMLayer(nn.Module):
    def __init__(self, num_features, num_langs=20):
        super().__init__()
        self.gamma = nn.Embedding(num_langs, num_features)
        self.beta = nn.Embedding(num_langs, num_features)
        nn.init.constant_(self.gamma.weight, 1.0)
        nn.init.constant_(self.beta.weight, 0.0)
    def forward(self, x, lang_ids):
        g = self.gamma(lang_ids).unsqueeze(1) if x.dim() == 3 else self.gamma(lang_ids)
        b = self.beta(lang_ids).unsqueeze(1) if x.dim() == 3 else self.beta(lang_ids)
        return x * g + b

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        h = self.layer2(self.act(self.layer1(x)))
        return self.norm(x + g * self.dropout(h))

class CrossAttentionStack(nn.Module):
    def __init__(self, hidden_size, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.norm_ffns = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src, src_key_padding_mask=None):
        for i, attn_layer in enumerate(self.layers):
            attn_out, _ = attn_layer(tgt, src, src, key_padding_mask=src_key_padding_mask)
            tgt = self.norms[i](tgt + self.dropout(attn_out))
            ff_out = self.ffns[i](tgt)
            tgt = self.norm_ffns[i](tgt + ff_out)
        return tgt

class HyperbolicInteractionNet(nn.Module):
    def __init__(self, model_name, curvature=1.0, dropout=0.1, num_langs=20, 
                 use_film=False, manifold_type="poincare", proj_type="mlp", learn_curvature=True,
                 scale_factor=0.3, use_interaction=True):
        super(HyperbolicInteractionNet, self).__init__()
        
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        self.use_film = use_film
        self.use_interaction = use_interaction
        self.manifold_type = manifold_type
        self.learn_curvature = learn_curvature
        
        if self.use_interaction:
            self.interaction = CrossAttentionStack(self.hidden_size, num_layers=2)
        if self.use_film: 
            self.film = FiLMLayer(self.hidden_size, num_langs=num_langs)
        
        if self.learn_curvature:
            # 1-D Tensor for robust gradients
            self.c_param = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_buffer('c_fixed', torch.tensor([curvature]))

        # We keep geoopt objects ONLY for utility functions (like mobius addition in complex cases)
        # but we won't use them for c-dependent math to avoid graph breaking.
        self.manifold_object = geoopt.PoincareBall(c=1.0)

        pooled_dim = self.hidden_size * 3
        self.hyp_dim = self.hidden_size // 2
        self.euc_dim = self.hidden_size - self.hyp_dim
        
        def make_proj(out_dim):
            if proj_type == "grn": return GatedResidualNetwork(pooled_dim, out_dim)
            elif proj_type == "linear": return nn.Linear(pooled_dim, out_dim)
            else: return nn.Sequential(nn.Linear(pooled_dim, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, out_dim))

        self.hyp_proj = make_proj(self.hyp_dim)
        self.euc_proj = make_proj(self.euc_dim)
        
        self.scale_factor = scale_factor
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    @property
    def c(self):
        if self.learn_curvature: 
            # Softplus ensures c > 0, offset 0.1 prevents instability at 0
            return torch.nn.functional.softplus(self.c_param) + 0.1
        return self.c_fixed

    # --- MANUAL GEOMETRY (PRESERVES GRAPH) ---
    def manual_expmap0(self, v):
        """Maps Euclidean vector v to Poincare Ball (origin 0) preserving gradient for c."""
        c = self.c # Get the tensor
        sqrt_c = torch.sqrt(c)
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        
        # Formula: tanh(sqrt(c) * norm) * (v / (sqrt(c) * norm))
        gamma = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return gamma * v

    def manual_dist(self, x, y):
        """Calculates Poincare distance preserving gradient for c."""
        c = self.c
        
        # Formula: dist = 2/sqrt(c) * artanh( sqrt(c) * ||-x + y||_mobius )
        # Simpler robust formula for distance squared:
        # d^2 = (2/c) * ( ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)) ) ... approx
        
        # Let's use the numerically stable formula:
        # d(u, v) = 1/sqrt(c) * acosh( 1 + 2c * ||u-v||^2 / ((1-c||u||^2)(1-c||v||^2)) )
        
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)
        euclidean_dist_sq = ((x - y) ** 2).sum(dim=-1, keepdim=True)
        
        # Denominators (lambda_x * lambda_y)
        denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
        denom = torch.clamp(denom, min=1e-7) # Stability
        
        # Argument for acosh
        arg = 1 + 2 * c * euclidean_dist_sq / denom
        
        # Distance
        sqrt_c = torch.sqrt(c)
        dist = (1 / sqrt_c) * torch.acosh(torch.clamp(arg, min=1.0 + 1e-7))
        
        return dist.squeeze(-1)

    def get_token_embeddings(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def multi_scale_pool(self, token_embeddings, attention_mask):
        cls_vec = token_embeddings[:, 0, :]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_vec = sum_embeddings / sum_mask
        token_embeddings[input_mask_expanded == 0] = -1e9
        max_vec = torch.max(token_embeddings, dim=1)[0]
        return torch.cat([cls_vec, mean_vec, max_vec], dim=1)

    def encode_hyperbolic(self, input_ids, attention_mask):
        seq = self.get_token_embeddings(input_ids, attention_mask)
        vec = self.multi_scale_pool(seq, attention_mask)
        # Use Manual Expmap
        return self.manual_expmap0(self.hyp_proj(vec) * self.scale_factor)

    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask, lang_ids):
        src_seq = self.get_token_embeddings(src_ids, src_mask)
        tgt_seq = self.get_token_embeddings(tgt_ids, tgt_mask)
        
        padding_mask = ~src_mask.bool()
        tgt_aligned = self.interaction(tgt_seq, src_seq, src_key_padding_mask=padding_mask) if self.use_interaction else tgt_seq
        
        if self.use_film: tgt_aligned = self.film(tgt_aligned, lang_ids)
        
        u_vec = self.multi_scale_pool(src_seq, src_mask)
        v_vec = self.multi_scale_pool(tgt_aligned, tgt_mask)
        
        # Use Manual Ops
        u_hyp = self.manual_expmap0(self.hyp_proj(u_vec) * self.scale_factor)
        v_hyp = self.manual_expmap0(self.hyp_proj(v_vec) * self.scale_factor)
        dist_hyp = self.manual_dist(u_hyp, v_hyp)
        
        u_euc = self.euc_proj(u_vec)
        v_euc = self.euc_proj(v_vec)
        dist_euc = torch.norm(u_euc - v_euc, p=2, dim=1)
        
        score = torch.exp(-self.alpha * (dist_hyp + dist_euc) + self.beta)
        return score, u_hyp, v_hyp

HyperbolicSiameseNet = HyperbolicInteractionNet