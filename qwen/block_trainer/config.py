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
    
    @classmethod
    def setup_dirs(cls):
        if cls.save_dir:
            os.makedirs(cls.save_dir, exist_ok=True)