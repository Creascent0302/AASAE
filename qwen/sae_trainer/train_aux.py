import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from sae_model import AuxiliaryAE
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Train Auxiliary AE with Qwen Embeddings', add_help=False)
    parser.add_argument('--embeddings_path', required=True, type=str, help="Path to the .pt file containing extracted features")
    parser.add_argument('--save_path', default="./checkpoints", type=str)
    parser.add_argument('--save_prefix', default="qwen", type=str, help="Prefix for saved model files")
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save_final', action='store_true')
    parser.add_argument('--training_ratio', default=0.8, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--warmup_steps', default=100, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--initial_lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--seed', default=42, type=int)
    return parser

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def contrastive_loss(vision_embed, text_embed, temperature=0.07):
    # Normalize features
    vision_embed = F.normalize(vision_embed, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)
    
    sim_matrix = torch.matmul(vision_embed, text_embed.transpose(0, 1)) / temperature    
    labels = torch.arange(vision_embed.shape[0], device=vision_embed.device)
    loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.transpose(0, 1), labels)
    return loss / 2.0

class AlignmentTrainer:
    def __init__(self, args, vision_dim, text_dim, device, config=None):
        self.device = device
        self.config = {
            'lr': args.initial_lr,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay,
            'temperature': args.temperature,
        }
        if config:
            self.config.update(config)
            
        self.model = AuxiliaryAE(vision_dim, text_dim).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['num_epochs']
        )
        self.scaler = GradScaler() # Use GradScaler for mixed precision safety
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_contrast_loss = 0
        total_recon_loss = 0
        
        for batch_idx, (vision_features, text_features) in enumerate(train_loader):
            vision_features = vision_features.to(self.device, non_blocking=True)
            text_features = text_features.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with autocast():
                vision_embed, text_embed, vision_recon, text_recon = self.model(vision_features, text_features)
                
                contrast_l = contrastive_loss(vision_embed, text_embed, self.config['temperature'])
                vision_recon_l = F.mse_loss(vision_recon, vision_features)
                text_recon_l = F.mse_loss(text_recon, text_features)
                recon_l = vision_recon_l + text_recon_l
                loss = 1.0 * contrast_l + recon_l
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_contrast_loss += contrast_l.item()
            total_recon_loss += recon_l.item()
            
        return {
            'total_loss': total_loss / len(train_loader),
            'contrast_loss': total_contrast_loss / len(train_loader),
            'recon_loss': total_recon_loss / len(train_loader)
        }
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_contrast_loss = 0
        total_recon_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (val_vision_features, val_text_features) in enumerate(val_loader):
                val_vision_features = val_vision_features.to(self.device, non_blocking=True)
                val_text_features = val_text_features.to(self.device, non_blocking=True)

                with autocast():
                    vision_embed, text_embed, vision_recon, text_recon = self.model(
                        val_vision_features, val_text_features
                    )
                    
                    contrast_l = contrastive_loss(vision_embed, text_embed, self.config['temperature'])
                    vision_recon_l = F.mse_loss(vision_recon, val_vision_features)
                    text_recon_l = F.mse_loss(text_recon, val_text_features)
                    recon_l = vision_recon_l + text_recon_l
                    loss = 1.0 * contrast_l + recon_l
                    
                    # Calculate accuracy based on cosine similarity
                    vision_norm = F.normalize(vision_embed, dim=-1)
                    text_norm = F.normalize(text_embed, dim=-1)
                    similarity = torch.matmul(vision_norm, text_norm.transpose(0, 1))
                    predictions = similarity.argmax(dim=-1)
                    labels = torch.arange(len(predictions), device=predictions.device)
                
                batch_size = len(predictions)
                total_loss += loss.item() * batch_size
                total_contrast_loss += contrast_l.item() * batch_size
                total_recon_loss += recon_l.item() * batch_size
                total_correct += (predictions == labels).sum().item()
                total_samples += batch_size
        
        return {
            'avg_loss': total_loss / total_samples,
            'contrast_loss': total_contrast_loss / total_samples,
            'recon_loss': total_recon_loss / total_samples,
            'accuracy': total_correct / total_samples
        }

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings_data = torch.load(args.embeddings_path, map_location='cpu', weights_only=False)
    
    # 转换为 float32 以保证训练稳定性和兼容性
    text_embeddings = torch.tensor(np.stack(embeddings_data['text_features']), dtype=torch.float32)
    image_embeddings = torch.tensor(np.stack(embeddings_data['image_features']), dtype=torch.float32)
    
    # 获取维度
    input_dim_text = text_embeddings.shape[1]
    input_dim_vision = image_embeddings.shape[1]
    
    print(f"Data shapes - Text: {text_embeddings.shape}, Vision: {image_embeddings.shape}")
    assert input_dim_text == input_dim_vision, "Text and Vision dimensions should match for this architecture (or modification needed)"
    
    config = {
        'lr': args.initial_lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'temperature': args.temperature,
    }

    trainer = AlignmentTrainer(args, vision_dim=input_dim_vision, text_dim=input_dim_text, device=device, config=config)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # Data splitting
    total_samples = len(text_embeddings)
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * args.training_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = TensorDataset(image_embeddings[train_indices], text_embeddings[train_indices])
    val_dataset = TensorDataset(image_embeddings[val_indices], text_embeddings[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        trainer.scheduler.step()
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f} (Contrast: {train_metrics['contrast_loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f})")
        print(f"  Val   - Loss: {val_metrics['avg_loss']:.4f} (Acc: {val_metrics['accuracy']*100:.2f}%)")
        
        # Save best model
        if val_metrics['avg_loss'] < best_val_loss:
            best_val_loss = val_metrics['avg_loss']
            save_file = os.path.join(args.save_path, f'{args.save_prefix}_aux_best.pt')
            torch.save(trainer.model.state_dict(), save_file)
            print(f"  --> Saved best model to {save_file}")
        
    if args.save_final:
        save_file = os.path.join(args.save_path, f'{args.save_prefix}_aux_last.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, save_file)
        print(f"Saved final model to {save_file}")