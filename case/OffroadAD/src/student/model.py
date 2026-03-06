import torch.nn as nn
from student.backbone import VMambaAdapter
from student.head import FlowMatchingHead

class PhysicsFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 骨干
        self.backbone = VMambaAdapter()
        
        # 2. 头 (输入维度要匹配 backbone 输出)
        self.head = FlowMatchingHead(backbone_dim=self.backbone.out_channels)

    def forward(self, img, noisy_map, t):
        # 提取视觉特征
        feats = self.backbone(img)
        # 预测流
        velocity = self.head(feats, noisy_map, t)
        return velocity