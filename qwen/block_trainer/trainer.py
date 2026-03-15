import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from config import Config
from sae_model import SAE_V, SAE_D, VL_SAE, TokenAuxProj

class PairDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]['vision'].float(), self.data[i]['text'].float()

def collate_fn(batch):
    v_list = [b[0] for b in batch]
    t_list = [b[1] for b in batch]
    v_len = torch.tensor([len(v) for v in v_list])
    t_len = torch.tensor([len(t) for t in t_list])
    
    v_pad = pad_sequence(v_list, batch_first=True)
    t_pad = pad_sequence(t_list, batch_first=True)
    
    v_mask = torch.arange(v_pad.size(1))[None, :] < v_len[:, None]
    t_mask = torch.arange(t_pad.size(1))[None, :] < t_len[:, None]
    return v_pad, t_pad, v_mask, t_mask

def batch_filip_loss(v_proj, t_proj, v_mask, t_mask, temp=0.07):
    """Token 级细粒度对比损失"""
    B, Lv, D = v_proj.shape
    _, Lt, _ = t_proj.shape

    v_norm = F.normalize(v_proj, dim=-1)
    t_norm = F.normalize(t_proj, dim=-1)

    sim = torch.einsum('b i d, c j d -> b c i j', v_norm, t_norm) / temp

    sim_v = sim.clone()
    sim_v.masked_fill_(~t_mask.view(1, B, 1, Lt), -1e4) 
    max_sim_v = sim_v.max(dim=3)[0] 
    align_v = (max_sim_v * v_mask.view(B, 1, Lv)).sum(dim=2) / v_mask.sum(dim=1).view(B, 1)

    sim_t = sim.clone()
    sim_t.masked_fill_(~v_mask.view(B, 1, Lv, 1), -1e4) 
    max_sim_t = sim_t.max(dim=2)[0] 
    align_t = (max_sim_t * t_mask.view(1, B, Lt)).sum(dim=2) / t_mask.sum(dim=1).view(1, B)

    align_score = (align_v + align_t) / 2.0 
    labels = torch.arange(B, device=v_proj.device)
    loss = F.cross_entropy(align_score, labels) + F.cross_entropy(align_score.t(), labels)
    return loss / 2.0

class SAETrainer:
    def __init__(self):
        print("[Trainer] Initializing Asynchronous Multi-GPU T-VL-SAE Training...")
        
        self.device_map = {
            "SAE_V": torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"),
            "SAE_D": torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0"),
            "VL_SAE": torch.device("cuda:3" if torch.cuda.device_count() > 3 else "cuda:0")
        }
        
        # 将模型和投影层直接实例化在它们专属的 GPU 上
        self.models = {
            "SAE_V": SAE_V(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_V"]),
            "SAE_D": SAE_D(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_D"]),
            "VL_SAE": VL_SAE(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["VL_SAE"])
        }
        
        self.aux_projs = {
            name: TokenAuxProj(Config.qwen_hidden_dim).to(self.device_map[name]) for name in self.models.keys()
        }
        
        self.criterion = nn.MSELoss()
        
        # 优化器状态也会自动被绑定到对应模型所在的 GPU 上
        self.optimizers = {
            name: optim.Adam(
                list(self.models[name].parameters()) + list(self.aux_projs[name].parameters()), 
                lr=Config.initial_lr, weight_decay=Config.weight_decay
            ) for name in self.models.keys()
        }
        
        self.scalers = {name: GradScaler('cuda') for name in self.models.keys()}

    def train_on_chunk(self, chunk_path, chunk_idx):
        print(f"\n[Trainer] Loading dual-modal data from {chunk_path}...")
        
        data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        dataset = PairDataset(data)
        
        loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
        
        for model in self.models.values(): model.train()
        for aux in self.aux_projs.values(): aux.train()
            
        total_losses = {"SAE_V": 0.0, "SAE_D": 0.0, "VL_SAE": 0.0}
        
        # 单层循环。从 DataLoader 获取 CPU 数据，然后分发给各卡
        pbar = tqdm(loader, desc=f"Multi-GPU Training (Chunk {chunk_idx})")
        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu in pbar:
            
            # 依次在三个 GPU 上发起计算指令（底层完全异步并发执行）
            for name in ["SAE_V", "SAE_D", "VL_SAE"]:
                target_device = self.device_map[name]
                
                # 数据被发送到当前模型专属的 GPU
                v_pad = v_pad_cpu.to(target_device, non_blocking=True)
                t_pad = t_pad_cpu.to(target_device, non_blocking=True)
                v_mask = v_mask_cpu.to(target_device, non_blocking=True)
                t_mask = t_mask_cpu.to(target_device, non_blocking=True)
                
                self.optimizers[name].zero_grad()
                
                with autocast('cuda'):
                    v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                    align_loss = batch_filip_loss(v_proj, t_proj, v_mask, t_mask)
                    
                    v_proj_flat = v_proj[v_mask]
                    t_proj_flat = t_proj[t_mask]
                    
                    recon_v, recon_t, _, _ = self.models[name](vision_embeddings=v_proj_flat, text_embeddings=t_proj_flat)
                    
                    # 严谨的数学隔离：切断 SAE 重建误差向对齐投影层的回传
                    recon_loss = self.criterion(recon_v, v_proj_flat.detach()) + self.criterion(recon_t, t_proj_flat.detach())
                    
                    loss = recon_loss + 0.1 * align_loss
                    
                # 梯度的缩放和反向传播都在其专属卡上独立完成
                self.scalers[name].scale(loss).backward()
                self.scalers[name].step(self.optimizers[name])
                self.scalers[name].update()
                
                total_losses[name] += loss.item()
                
                if name == "VL_SAE":
                    display_loss = loss.item()
                    
                # 清理张量引用，让这块 GPU 的缓存可以被复用
                del recon_v, recon_t, loss, v_proj, t_proj, v_pad, t_pad
            
            pbar.set_postfix({'VL_SAE_Loss': f"{display_loss:.4f}"})

        num_batches = len(loader)
        print(f"[Trainer] Chunk {chunk_idx} Finished | "
              f"V: {total_losses['SAE_V']/num_batches:.4f}, "
              f"D: {total_losses['SAE_D']/num_batches:.4f}, "
              f"VL: {total_losses['VL_SAE']/num_batches:.4f}")

    def save_checkpoint(self, suffix):
        os.makedirs(Config.save_dir, exist_ok=True)
        for name, model in self.models.items():
            filename = suffix.replace(Config.model_type, name) if hasattr(Config, 'model_type') else f"{name}_{suffix}"
            path = os.path.join(Config.save_dir, filename)
            # 保存时统一映射回 cpu 防止加载设备冲突
            torch.save(model.state_dict(), path)
            print(f"[Trainer] Checkpoint saved: {path}")