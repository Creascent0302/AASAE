import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
# 修正 Warning: 使用新的 amp API
from torch.amp import autocast, GradScaler
from sae_model import VL_SAE, SAE_D, SAE_V, AuxiliaryAE

def get_args_parser():
    parser = argparse.ArgumentParser('Train VL-SAE with Qwen Embeddings', add_help=False)
    parser.add_argument('--embeddings_path', required=True, type=str)
    parser.add_argument('--aux_ae_path', required=True, type=str)
    parser.add_argument('--save_path', default="./checkpoints", type=str)
    parser.add_argument('--save_prefix', default="qwen_sae", type=str)
    
    parser.add_argument('--model_type', default="VL_SAE", choices=["VL_SAE", "SAE_D", "SAE_V"], type=str)
    parser.add_argument('--topk', default=128, type=int)
    parser.add_argument('--hidden_ratio', default=8, type=int)
    
    parser.add_argument('--training_ratio', default=0.8, type=float)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--initial_lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # 新增参数：显式指定投影维度，默认 4096 (必须与 train_aux 时一致)
    parser.add_argument('--projection_dim', default=4096, type=int, help="Dimension of the Auxiliary AE projection space")

    return parser

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate(model, alignment_model, val_text_embeddings, val_image_embeddings, criterion, batch_size, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i in range(0, len(val_text_embeddings), batch_size):
            batch_img = val_image_embeddings[i:i+batch_size].to(device)
            batch_txt = val_text_embeddings[i:i+batch_size].to(device)
            
            # 使用新的 autocast API 修正 warning
            with autocast('cuda'):
                # 1. 对齐模型输出的是 [Batch, 4096]
                aligned_v, aligned_t, _, _ = alignment_model(batch_img, batch_txt)
                
                # 2. SAE 输入也必须是 [Batch, 4096]
                recon_v, recon_t, _, _ = model(aligned_v, aligned_t)
            
                # 目标是重建这个对齐后的特征
                loss = criterion(recon_v, aligned_v) + criterion(recon_t, aligned_t)
            
            total_loss += loss.item() * batch_img.size(0)
    
    avg_loss = total_loss / len(val_text_embeddings)
    model.train()
    return avg_loss

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings_data = torch.load(args.embeddings_path, map_location='cpu', weights_only=False)
    
    text_embeddings = torch.tensor(np.stack(embeddings_data['text_features']), dtype=torch.float32)
    image_embeddings = torch.tensor(np.stack(embeddings_data['image_features']), dtype=torch.float32)
    
    # ------------------ 关键修改区域 START ------------------
    
    # 原始 Qwen 维度 (例如 3584)
    raw_input_dim = text_embeddings.shape[1]
    print(f"Detected Raw Input Dimension (Qwen): {raw_input_dim}")
    
    # 对齐空间维度 (例如 4096 - 必须与 AuxAE 训练时一致)
    sae_input_dim = args.projection_dim
    print(f"SAE Input Dimension (Aligned Space): {sae_input_dim}")
    
    # SAE 的隐藏层大小通常基于 SAE 输入维度扩增
    sae_hidden_dim = sae_input_dim * args.hidden_ratio
    
    # A. 初始化对齐模型 (Auxiliary AE)
    # 输入: raw_input_dim (3584), 输出(内部投影): sae_input_dim (4096)
    print(f"Loading Auxiliary AE from {args.aux_ae_path}...")
    alignment_model = AuxiliaryAE(
        vision_dim=raw_input_dim, 
        text_dim=raw_input_dim, 
        projection_dim=sae_input_dim 
    ).to(device)
    
    aux_ckpt = torch.load(args.aux_ae_path, map_location='cpu')
    alignment_model.load_state_dict(aux_ckpt)
    alignment_model.eval()
    for param in alignment_model.parameters():
        param.requires_grad = False

    # B. 初始化 SAE 模型
    # 注意：SAE 的 input_dim 必须是 sae_input_dim (4096)，不能是 raw_input_dim
    if args.model_type == 'VL_SAE':
        autoencoder = VL_SAE(sae_input_dim, sae_hidden_dim, topk=args.topk, dropout=0).to(device)
    elif args.model_type == 'SAE_D':
        autoencoder = SAE_D(sae_input_dim, sae_hidden_dim, topk=args.topk, dropout=0).to(device)
    elif args.model_type == 'SAE_V':
        autoencoder = SAE_V(sae_input_dim, sae_hidden_dim, topk=args.topk, dropout=0).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # ------------------ 关键修改区域 END ------------------

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs)
    
    # 修正 Warning: 显式指定设备
    scaler = GradScaler('cuda')

    # Data Splitting
    total_samples = len(text_embeddings)
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * args.training_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_text = text_embeddings[train_indices]
    train_img = image_embeddings[train_indices]
    val_text = text_embeddings[val_indices]
    val_img = image_embeddings[val_indices]

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    print(f"Training started: {args.model_type}, TopK={args.topk}, HiddenRatio={args.hidden_ratio}")
    best_val_loss = float('inf')
    patience_counter = 0

    num_steps = (len(train_text) + args.batch_size - 1) // args.batch_size
    
    for epoch in range(args.num_epochs):
        if epoch < args.warmup_epochs:
            lr = args.initial_lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']

        autoencoder.train()
        epoch_loss = 0
        
        perm = torch.randperm(len(train_text))
        train_text = train_text[perm]
        train_img = train_img[perm]

        pbar = tqdm(range(0, len(train_text), args.batch_size), desc=f"Epoch {epoch+1}")
        
        for i in pbar:
            batch_img = train_img[i:i + args.batch_size].to(device)
            batch_txt = train_text[i:i + args.batch_size].to(device)
            
            optimizer.zero_grad()

            with autocast('cuda'):
                with torch.no_grad():
                    # Aux Model 输出 [Batch, 4096]
                    aligned_v, aligned_t, _, _ = alignment_model(batch_img, batch_txt)
                
                # SAE 输入 [Batch, 4096]
                recon_v, recon_t, _, _ = autoencoder(aligned_v, aligned_t)
                
                loss = criterion(recon_v, aligned_v) + criterion(recon_t, aligned_t)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / num_steps
        val_loss = validate(autoencoder, alignment_model, val_text, val_img, criterion, args.batch_size, device)
        
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], LR: {lr:.2e}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        save_name = f"{args.save_prefix}_topk{args.topk}_r{args.hidden_ratio}"
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), os.path.join(args.save_path, f'{save_name}_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                break
                
    torch.save(autoencoder.state_dict(), os.path.join(args.save_path, f'{save_name}_final.pth'))
    print("Training finished.")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)