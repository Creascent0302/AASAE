import torch
import cv2
import numpy as np
from student.model import PhysicsFlowModel
from config import Config

def inference_single_image(model, image_tensor):
    """
    单步生成推理 (One-Step Generation)
    """
    model.eval()
    device = Config.DEVICE
    B, _, H, W = image_tensor.shape
    
    with torch.no_grad():
        # 1. 准备噪声 x0 (Standard Gaussian)
        x0 = torch.randn(B, 1, H, W).to(device)
        
        # 2. 设置时间 t=0
        t = torch.zeros(B, 1).to(device)
        
        # 3. 预测速度场 v
        # 模型输入: 图片(提供Condition), 噪声(提供起点), 时间
        velocity = model(image_tensor, x0, t)
        
        # 4. 欧拉积分 (Euler Step)
        # x_1 = x_0 + v * dt (这里 dt=1.0)
        # 因为我们学习的是直线轨迹，一步到位
        x_final = x0 + velocity
        
        # 5. 映射回 0-1 (Sigmoid)
        # 因为 Cost Map 定义在 [0,1] 之间
        prediction = torch.sigmoid(x_final)
        
    return prediction

def main():
    # 加载模型
    model = PhysicsFlowModel().to(Config.DEVICE)
    # model.load_state_dict(torch.load("path/to/checkpoint.pth"))
    
    # 模拟输入
    dummy_img = torch.randn(1, 3, 512, 512).to(Config.DEVICE)
    
    print("Running Inference...")
    result = inference_single_image(model, dummy_img)
    
    # 后处理与可视化
    result_np = result.squeeze().cpu().numpy()
    result_vis = (result_np * 255).astype(np.uint8)
    
    # 保存结果
    cv2.imwrite("inference_result.jpg", result_vis)
    print("Done! Saved to inference_result.jpg")

if __name__ == "__main__":
    main()