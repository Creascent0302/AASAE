import torch

class Config:
    # 硬件设置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 图像参数
    IMG_SIZE = (512, 512)
    IN_CHANNELS = 3
    
    # 模型参数
    HIDDEN_DIM = 128  # Flow Head 的隐藏层维度
    
    # 路径配置
    # 请替换为你本地的模型路径
    QWEN_PATH = "../../../../pretrained_models/Qwen2.5-VL-7B-Instruct" 
    SAM_PATH = "../utils/sam2.1_b.pt"
    RELLIS_ROOT = "/path/to/rellis_3d"
    
    # 训练参数
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 50