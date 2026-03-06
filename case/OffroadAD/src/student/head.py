import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingHead(nn.Module):
    """
    接收 [Backbone特征, 噪声图, 时间t]，预测 [速度场 v]
    """
    def __init__(self, backbone_dim, hidden_dim=128):
        super().__init__()
        
        # 1. 时间嵌入 (Time Embedding)
        # 将标量 t (0~1) 映射为向量
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. 特征融合层
        # 输入通道 = Backbone特征 + 1 (Noisy Map)
        self.conv_in = nn.Conv2d(backbone_dim + 1, hidden_dim, kernel_size=3, padding=1)
        
        # 3. 主干处理 (简单的 ResBlock 堆叠，保持轻量)
        self.body = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU()
        )
        
        # 4. 输出层 (预测速度场 Velocity)
        self.out_layer = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, features, noisy_map, t):
        """
        features: [B, C_feat, H/32, W/32]
        noisy_map: [B, 1, H, W]
        t: [B, 1]
        """
        # A. 处理时间嵌入
        t_emb = self.time_mlp(t) # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1) # [B, hidden_dim, 1, 1]
        
        # B. 对齐特征图尺寸
        # 将 Backbone 的特征上采样到与 noisy_map 一样大
        features_resized = F.interpolate(features, size=noisy_map.shape[2:], mode='bilinear', align_corners=False)
        
        # C. 拼接输入 (Early Fusion)
        x = torch.cat([features_resized, noisy_map], dim=1) # [B, C_feat+1, H, W]
        
        # D. 卷积处理
        x = self.conv_in(x)
        x = x + t_emb # 注入时间信息 (Broadcasting)
        x = self.body(x)
        
        # E. 输出速度
        velocity = self.out_layer(x)
        return velocity