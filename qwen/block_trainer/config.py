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
    max_grad_norm = 5.0

    # --- SAE regularization & preprocessing (CaFE-style) ---
    input_unit_norm = True
    l1_coeff = 0.0
    filip_l1_coeff = 1e-4
    filip_token_topk = 128
    aux_penalty = 1.0 / 32.0
    top_k_aux = 256
    n_batches_to_dead = 20
    use_threshold_in_eval = False

    # --- Initialization helpers ---
    init_b_dec_batches = 8

    # --- 训练方法控制 (filip 或 asym) ---
    train_method = 'asym' 
    asym_use_views = True
    num_views = 8       
    gamma = 10.0        
    lambda_align = 0.5  
    asym_text_pool = "softmax_topk"  # mean | softmax_topk
    asym_text_temp = 0.8
    asym_text_topk = 32
    asym_union_temp = 0.5
    asym_aux_mode = "hybrid"  # asym | filip | hybrid
    asym_aux_alpha = 0.3
    asym_latent_align = 0.1
      
    @classmethod
    def setup_dirs(cls):
        if cls.save_dir:
            os.makedirs(cls.save_dir, exist_ok=True)