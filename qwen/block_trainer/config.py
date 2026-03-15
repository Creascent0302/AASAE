import os

class Config:
    # --- Data & Paths (将在运行时被 argparse 覆盖) ---
    model_path = None
    image_folder = None
    dataset_file = None
    save_dir = None
    target_layer_name = None
    seed = 42
    
    # --- Extraction Settings ---
    chunk_size = 200  
    temp_chunk_prefix = "temp_chunk_"
    
    # --- SAE Architecture ---
    model_type = "SAE_V" 
    qwen_hidden_dim = 3584 
    sae_hidden_ratio = 8
    sae_hidden_dim = qwen_hidden_dim * sae_hidden_ratio
    topk = 64
    
    # --- Training Settings ---
    batch_size = 64 
    initial_lr = 1e-4
    weight_decay = 0.0

    # --- Asym_SAE 参数 ---
    enable_asym = False # 是否开启 Asym_SAE 增量训练
    num_views = 8       # K: 动态采样的视图数量
    gamma = 10.0        # 聚光灯的高斯衰减因子
    lambda_align = 0.5  # 最小匹配蕴含损失的权重
    
    @classmethod
    def setup_dirs(cls):
        if cls.save_dir:
            os.makedirs(cls.save_dir, exist_ok=True)