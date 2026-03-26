import os

class Config:
    # --- Data & Paths (将在运行时被 argparse 覆盖) ---
    model_path = "/home/liuzonghao/pretrained_models/Qwen2.5-VL-7B-Instruct"
    image_folder = "/home/liuzonghao/AASAE/VL-SAE/CC3M/cc3m_jpg"
    dataset_file = "/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_train.json"
    save_dir = "/home/liuzonghao/AASAE/qwen/block_trainer/checkpoints_sae"
    target_layer_name = "model.language_model.layers.20"
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
    batch_size = 32
    initial_lr = 1e-4
    weight_decay = 0.0

    # --- 训练方法控制 (filip 或 asym) ---
    train_method = 'asym' 
    num_views = 8       
    gamma = 10.0        
    lambda_align = 0.5  
      
    @classmethod
    def setup_dirs(cls):
        if cls.save_dir:
            os.makedirs(cls.save_dir, exist_ok=True)