import json
import os
import torch
import argparse
from transformers import set_seed


try:    
    from config import Config
    from extractor import FeatureExtractor
    from trainer import SAETrainer, AuxProjTrainer
except ImportError:
    from block_trainer.config import Config
    from block_trainer.extractor import FeatureExtractor
    from block_trainer.trainer import SAETrainer, AuxProjTrainer


def get_args_parser():
    parser = argparse.ArgumentParser(description="Online SAE Training Pipeline")
    # 继承你之前的参数
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--target_layer_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--chunk_size", type=int, default=200, help="Number of images per extraction chunk")
    parser.add_argument("--sae_hidden_ratio", type=int, default=8)
    parser.add_argument("--topk", type=int, default=64)

    parser.add_argument("--train_method", type=str, default="filip", 
                    choices=['filip', 'asym', 'sym'],
                    help="Choose training paradigm: 'sym' (Global), 'filip' (Token-level) or 'asym' (Asymmetric Entailment)")    
    parser.add_argument("--num_views", type=int, default=8, help="K views for Asym mode")
    parser.add_argument("--gamma", type=float, default=10.0, help="Gamma for Gaussian mask")
    parser.add_argument("--lambda_align", type=float, default=0.05, help="Target weight for entailment penalty after warmup")
    parser.add_argument(
        "--filip_l1_coeff",
        type=float,
        default=Config.filip_l1_coeff,
        help="L1 sparsity coefficient for FILIP SAE.",
    )
    parser.add_argument(
        "--filip_token_topk",
        type=int,
        default=Config.filip_token_topk,
        help="Top-k tokens per sample for FILIP SAE training (0 disables).",
    )
    parser.add_argument(
        "--asym_use_views",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether ASYM uses view sampling (1) or token-level only (0).",
    )
    return parser

def load_dataset(file_path):
    with open(file_path, "r") as f:
        datasets = json.load(f)
    return datasets

def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    Config.model_path = args.model_path
    Config.image_folder = args.image_folder
    Config.dataset_file = args.dataset_file
    Config.save_dir = args.save_path
    Config.target_layer_name = args.target_layer_name
    Config.seed = args.seed
    Config.chunk_size = args.chunk_size
    Config.sae_hidden_ratio = args.sae_hidden_ratio
    Config.sae_hidden_dim = Config.qwen_hidden_dim * Config.sae_hidden_ratio
    Config.topk = args.topk
    Config.train_method = args.train_method
    Config.asym_use_views = bool(args.asym_use_views)
    Config.num_views = args.num_views
    Config.gamma = args.gamma
    Config.filip_l1_coeff = args.filip_l1_coeff
    Config.filip_token_topk = args.filip_token_topk
    target_lambda = args.lambda_align
    Config.lambda_align = 0.0

    set_seed(Config.seed)
    Config.setup_dirs()
    
    print(f"Loading dataset from {Config.dataset_file}...")
    dataset = load_dataset(Config.dataset_file)
    chunks = list(chunk_list(dataset, Config.chunk_size))
    total_chunks = len(chunks)
    print(f"Total dataset: {len(dataset)} items. Split into {total_chunks} chunks.")
    
    # 初始化引擎
    extractor = FeatureExtractor()

    print(f"\n{'*'*50}")
    print(f" 🚀 PHASE 1: Training Token Alignment ({Config.train_method})")
    print(f"{'*'*50}")
    val_chunk_path = extractor.extract_and_save_chunk(chunks[0], chunk_idx=0)
    if val_chunk_path is None:
            return

    auxprojector = AuxProjTrainer()
    for chunk_idx, current_chunk in enumerate(chunks[1:], start=1):
        print(f"\n--- [Phase 1] Processing Chunk {chunk_idx} / {total_chunks - 1} ---")
        
        chunk_path = extractor.extract_and_save_chunk(current_chunk, chunk_idx)
        if chunk_path is None: continue
            
        # 传入训练集和验证集
        auxprojector.train_on_chunk(chunk_path, val_chunk_path, chunk_idx)
        
        try:
            os.remove(chunk_path)
            print(f"[Orchestrator] Deleted temporary file: {chunk_path}")
        except OSError as e:
            print(f"[Orchestrator] Error deleting {chunk_path}: {e}")
            
        torch.cuda.empty_cache()

    print(f"\n{'*'*50}")
    print(f" 🚀 PHASE 2: Training SAE Dictionary ({Config.train_method})")
    print(f"{'*'*50}")
    trainer = SAETrainer()
    total_train_chunks = total_chunks - 1

    for chunk_idx, current_chunk in enumerate(chunks[1:], start=1):

        if Config.train_method == 'asym':
            progress = (chunk_idx - 1) / total_train_chunks if total_train_chunks > 1 else 1.0
            
            # 前 20% 时间，完全不给惩罚，让模型专心把重构 (EV) 学好
            if progress < 0.2:
                current_lambda = 0.0
            else:
                # 剩下的 80% 时间，让惩罚系数从 0 线性平滑上升到目标值 (如 0.05)
                current_lambda = target_lambda * ((progress - 0.2) / 0.8)
                
            Config.lambda_align = current_lambda
        else:
            current_lambda = 0.0

        print(f"\n{'='*50}")
        print(f"   Processing Chunk {chunk_idx} / {total_chunks - 1}")
        if Config.train_method == 'asym':
            print(f"   📈 Current Lambda Align: {current_lambda:.5f} (Target: {target_lambda})")
        print(f"{'='*50}")
        
        chunk_path = extractor.extract_and_save_chunk(current_chunk, chunk_idx)
        
        if chunk_path is None:
            continue
            
        trainer.train_on_chunk(chunk_path, val_chunk_path, chunk_idx)
        
        try:
            os.remove(chunk_path)
            print(f"[Orchestrator] Deleted temporary file: {chunk_path}")
        except OSError as e:
            print(f"[Orchestrator] Error deleting {chunk_path}: {e}")
            
        torch.cuda.empty_cache()

    try:
        os.remove(val_chunk_path)
        print(f"[Orchestrator] Deleted validation file: {val_chunk_path}")
    except OSError:
        pass

    print("\n🎉 Online Training Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()