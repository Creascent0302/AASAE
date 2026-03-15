import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenAuxProj(nn.Module):
    """
    Token 级独立投影层。替代原论文中处理全局向量的 AuxiliaryAE。
    保持特征的细粒度空间结构，为后续的 FILIP 对齐损失做准备。
    """
    def __init__(self, dim):
        super().__init__()
        self.v_proj = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(dim, dim)

    def forward(self, v, t):
        return self.v_proj(v), self.t_proj(t)


class VL_SAE(nn.Module):
    """
    共享编码器，独立解码器。
    基于欧氏距离（转换为高效的余弦乘法实现）激活隐层概念。
    """
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0):
        super().__init__()
        # 概念字典权重
        self.encoder = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.encoder, a=math.sqrt(5))
        
        # 模态特异性解码器
        self.vision_decoder = nn.Linear(hidden_dim, input_dim)
        self.text_decoder = nn.Linear(hidden_dim, input_dim)

        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings):
        # 提取前 K 大的值和索引，并在全零张量上撒回 (scatter)
        topk_vals, topk_indices = torch.topk(embeddings, k=self.topk, dim=-1)
        sparse_embeddings = torch.zeros_like(embeddings)
        sparse_embeddings.scatter_(-1, topk_indices, topk_vals)
        return sparse_embeddings

    def encode(self, embeddings, mode='eval'):
        # L2 归一化
        weights = F.normalize(self.encoder, p=2, dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 性能巅峰：用极快且省显存的 F.linear 计算余弦相似度，替代 torch.cdist
        cos_sim = F.linear(embeddings, weights)
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0) # 防止精度溢出导致负数
        distances = torch.sqrt(2.0 - 2.0 * cos_sim)
        
        embeddings = 2.0 - distances
        return self.sparsify(embeddings)

    def forward(self, vision_embeddings=None, text_embeddings=None, mode='eval'):
        recon_vision_embeddings, recon_text_embeddings = None, None
        latent_v, latent_t = None, None
        
        if vision_embeddings is not None:
            latent_v = self.encode(vision_embeddings, mode=mode)
            recon_vision_embeddings = self.vision_decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode(text_embeddings, mode=mode)
            recon_text_embeddings = self.text_decoder(latent_t)
            
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t


class SAE_D(nn.Module):
    """
    独立编码器，独立解码器 (双流 SAE)。
    """
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0.1):
        super().__init__()
        self.v_encoder = nn.Linear(input_dim, hidden_dim)
        self.t_encoder = nn.Linear(input_dim, hidden_dim)
        self.activations = nn.ReLU()
        
        self.vision_decoder = nn.Linear(hidden_dim, input_dim)
        self.text_decoder = nn.Linear(hidden_dim, input_dim)

        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings):
        topk_vals, topk_indices = torch.topk(embeddings, k=self.topk, dim=-1)
        sparse_embeddings = torch.zeros_like(embeddings)
        sparse_embeddings.scatter_(-1, topk_indices, topk_vals)
        return sparse_embeddings

    def encode_v(self, embeddings):
        return self.sparsify(self.activations(self.v_encoder(embeddings)))
    
    def encode_t(self, embeddings):
        return self.sparsify(self.activations(self.t_encoder(embeddings)))

    def forward(self, vision_embeddings=None, text_embeddings=None):
        recon_vision_embeddings, recon_text_embeddings = None, None
        latent_v, latent_t = None, None
        
        if vision_embeddings is not None:
            latent_v = self.encode_v(vision_embeddings)
            recon_vision_embeddings = self.vision_decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode_t(text_embeddings)
            recon_text_embeddings = self.text_decoder(latent_t)
            
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t


class SAE_V(nn.Module):
    """
    共享编码器，共享解码器 (单流 SAE)。
    """
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activations = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings):
        topk_vals, topk_indices = torch.topk(embeddings, k=self.topk, dim=-1)
        sparse_embeddings = torch.zeros_like(embeddings)
        sparse_embeddings.scatter_(-1, topk_indices, topk_vals)
        return sparse_embeddings

    def encode(self, embeddings):
        return self.sparsify(self.activations(self.encoder(embeddings)))

    def forward(self, vision_embeddings=None, text_embeddings=None, mode='eval'):
        recon_vision_embeddings, recon_text_embeddings = None, None
        latent_v, latent_t = None, None
        
        if vision_embeddings is not None:
            latent_v = self.encode(vision_embeddings)
            recon_vision_embeddings = self.decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode(text_embeddings)
            recon_text_embeddings = self.decoder(latent_t)
            
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t