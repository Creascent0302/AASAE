import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicViewSampler(nn.Module):
    def __init__(self, num_views=8, gamma=10.0, eps=1e-6):
        super().__init__()
        self.num_views = num_views
        self.gamma = gamma
        self.eps = eps

    def forward(self, v_pad, v_len, grid_thws):
        B, _, D = v_pad.shape
        device = v_pad.device
        
        v_views = torch.zeros(B, self.num_views, D, device=device, dtype=v_pad.dtype)
        centers = torch.rand(B, self.num_views, 2, device=device) 
        
        for b in range(B):
            if grid_thws[b] is None:
                v_views[b] = v_pad[b, 0, :].unsqueeze(0).repeat(self.num_views, 1)
                continue
                
            H, W = grid_thws[b][1].item(), grid_thws[b][2].item()
            L = H * W
            
            y_coords = (torch.arange(H, device=device) + 0.5) / H
            x_coords = (torch.arange(W, device=device) + 0.5) / W
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1) 
            
            diff = centers[b].unsqueeze(1) - coords.unsqueeze(0) 
            dist_sq = (diff ** 2).sum(dim=-1) 
            
            m = torch.exp(-self.gamma * dist_sq) 
            
            valid_v_feat = v_pad[b, :L, :] 
            numerator = torch.mm(m, valid_v_feat) 
            denominator = m.sum(dim=1, keepdim=True) + self.eps 
            
            v_views[b] = numerator / denominator
            
        return v_views

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk=32):
        super().__init__()
        self.encoder = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.encoder, a=math.sqrt(5))
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.topk = topk

    def encode(self, x):
        weights = F.normalize(self.encoder, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=-1)
        cos_sim = F.linear(x_norm, weights)
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
        acts = 2.0 - torch.sqrt(2.0 - 2.0 * cos_sim)
        
        topk_vals, topk_indices = torch.topk(acts, k=self.topk, dim=-1)
        sparse_acts = torch.zeros_like(acts)
        sparse_acts.scatter_(-1, topk_indices, topk_vals)
        return sparse_acts

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decoder(latent)
        return recon, latent

class AsymmetricMultimodalSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk, num_views, gamma):
        super().__init__()
        self.sampler = DynamicViewSampler(num_views, gamma)
        self.vision_sae = SparseAutoencoder(input_dim, hidden_dim, topk)
        self.text_sae = SparseAutoencoder(input_dim, hidden_dim, topk)
        
    def forward(self, v_pad, v_len, grid_thws, t_pad, t_mask):
        t_sum = (t_pad * t_mask.unsqueeze(-1)).sum(dim=1)
        t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6) 
        
        v_views = self.sampler(v_pad, v_len, grid_thws) 
        
        recon_t, latent_t = self.text_sae(t_global)
        
        B, K, D = v_views.shape
        recon_v_flat, latent_v_flat = self.vision_sae(v_views.view(B * K, D))
        
        recon_v = recon_v_flat.view(B, K, D)
        latent_v = latent_v_flat.view(B, K, -1)
        
        return recon_v, v_views, recon_t, t_global, latent_v, latent_t