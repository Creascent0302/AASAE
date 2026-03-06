import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from student.model import PhysicsFlowModel
from config import Config
import numpy as np

# --- 模拟数据集 (Mock Dataset) ---
class RellisMockDataset(Dataset):
    def __init__(self):
        self.len = 100
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # 返回: 图像, 物理真值图
        img = torch.randn(3, 512, 512) # 模拟 RGB
        gt_map = torch.rand(1, 512, 512) # 模拟 Teacher 生成的 0-1 Map
        return img, gt_map

# --- 训练核心 ---
def train():
    device = Config.DEVICE
    
    # 1. 初始化模型
    model = PhysicsFlowModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    
    # 2. 数据加载
    dataset = RellisMockDataset() # 替换为真实的 Dataset
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    print("Start Training Physics-Flow...")
    model.train()
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        for batch_idx, (images, gt_maps) in enumerate(dataloader):
            images = images.to(device)
            gt_maps = gt_maps.to(device)
            B = images.shape[0]
            
            # --- Flow Matching Training Logic (关键) ---
            
            # 1. 采样时间 t [0, 1]
            t = torch.rand(B, 1).to(device) # [B, 1]
            
            # 2. 采样初始高斯噪声 x0
            x0 = torch.randn_like(gt_maps).to(device)
            
            # 3. 构造直线轨迹上的中间状态 x_t
            # Formula: x_t = t * x_1 + (1 - t) * x_0
            # 这里 x_1 就是 gt_maps
            x_t = t.view(B, 1, 1, 1) * gt_maps + (1 - t.view(B, 1, 1, 1)) * x0
            
            # 4. 计算目标速度 v_target
            # Formula: v_target = x_1 - x_0
            target_velocity = gt_maps - x0
            
            # 5. 模型预测
            pred_velocity = model(images, x_t, t)
            
            # 6. 计算 Loss (MSE)
            loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
            
            # 7. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{Config.EPOCHS}] Batch {batch_idx}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    train()