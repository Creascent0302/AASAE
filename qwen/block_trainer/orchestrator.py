import json
import os
import torch
import argparse
from transformers import set_seed

# 导入其他模块
from config import Config
from extractor import FeatureExtractor
from trainer import SAETrainer

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

    parser.add_argument("--enable_asym", action="store_true", help="Enable training for Asymmetric SAE")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=10.0)

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
    Config.enable_asym = args.enable_asym
    Config.num_views = args.num_views
    Config.gamma = args.gamma
    
    set_seed(Config.seed)
    Config.setup_dirs()
    
    print(f"Loading dataset from {Config.dataset_file}...")
    dataset = load_dataset(Config.dataset_file)
    chunks = list(chunk_list(dataset, Config.chunk_size))
    total_chunks = len(chunks)
    print(f"Total dataset: {len(dataset)} items. Split into {total_chunks} chunks.")
    
    # 初始化引擎
    extractor = FeatureExtractor()
    trainer = SAETrainer()
    
    for chunk_idx, current_chunk in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"   Processing Chunk {chunk_idx + 1} / {total_chunks}")
        print(f"{'='*50}")
        
        chunk_path = extractor.extract_and_save_chunk(current_chunk, chunk_idx)
        
        if chunk_path is None:
            continue
            
        trainer.train_on_chunk(chunk_path, chunk_idx)
        
        try:
            os.remove(chunk_path)
            print(f"[Orchestrator] Deleted temporary file: {chunk_path}")
        except OSError as e:
            print(f"[Orchestrator] Error deleting {chunk_path}: {e}")
            
        torch.cuda.empty_cache()

    trainer.save_checkpoint(f"{Config.model_type}_final.pth")
    print("\n🎉 Online Training Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()