import torch
import torch.nn as nn
import torchvision.models as models

class VMambaAdapter(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.use_vmamba = False
        
        try:
            # 尝试导入 VMamba (假设你已经安装了官方库)
            from vmamba import VSSM 
            self.model = VSSM(dims=96, depths=[2,2,9,2], ssms=['mamba']*4)
            self.use_vmamba = True
            self.out_channels = 768 # VMamba Small output
            print("Using VMamba Backbone.")
        except ImportError:
            # Fallback 到 ResNet50 (为了演示方便)
            print("VMamba not found. Falling back to ResNet50 for demonstration.")
            resnet = models.resnet50(pretrained=pretrained)
            self.model = nn.Sequential(*list(resnet.children())[:-2]) # 去掉全连接层
            self.use_vmamba = False
            self.out_channels = 2048

    def forward(self, x):
        return self.model(x)