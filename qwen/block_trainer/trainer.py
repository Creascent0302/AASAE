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
import math
class PairDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): 
        # 安全获取 grid_thw，如果在没有该字段的老数据集上运行，会返回 None
        grid = self.data[i].get('grid_thw', None)
        return self.data[i]['vision'].float(), self.data[i]['text'].float(), grid

def collate_fn(batch):
    v_list = [b[0] for b in batch]
    t_list = [b[1] for b in batch]
    grid_thws = [b[2] for b in batch]

    v_len = torch.tensor([len(v) for v in v_list])
    t_len = torch.tensor([len(t) for t in t_list])
    
    v_pad = pad_sequence(v_list, batch_first=True)
    t_pad = pad_sequence(t_list, batch_first=True)
    
    v_mask = torch.arange(v_pad.size(1))[None, :] < v_len[:, None]
    t_mask = torch.arange(t_pad.size(1))[None, :] < t_len[:, None]
    return v_pad, t_pad, v_mask, t_mask, grid_thws, v_len

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
    
class DynamicViewSampler(nn.Module):
    """鲁棒的动态高斯聚光灯采样器 (支持任何变长序列的自适应 2D 映射)"""
    def __init__(self, num_views, gamma):
        super().__init__()
        self.num_views = num_views
        self.gamma = gamma # 高斯衰减因子，越大越聚焦于中心点

    def forward(self, v_pad, v_len, grid_thws):
        B, _, D = v_pad.shape
        device = v_pad.device
        v_views = torch.zeros(B, self.num_views, D, device=device, dtype=v_pad.dtype)
        centers = torch.rand(B, self.num_views, 2, device=device) 
        
        for b in range(B):
            if grid_thws[b] is None:
                v_views[b] = v_pad[b, 0, :].unsqueeze(0).repeat(self.num_views, 1)
                continue
            
            # 1. 获取实际序列长度与原始网格比例
            Lv = v_len[b].item()
            H, W = grid_thws[b][1].item(), grid_thws[b][2].item()
            
            # 2. 动态计算等效的 2D 尺寸 (完美兼容所有的空间池化与特殊 Token)
            W_eff = max(1, int(round(math.sqrt(Lv * (W / H)))))
            H_eff = max(1, math.ceil(Lv / W_eff))
            
            # 3. 将 1D 序列映射回 2D 归一化坐标系 [0, 1]
            indices = torch.arange(Lv, device=device)
            x_coords = (indices % W_eff).float() / W_eff
            y_coords = (indices // W_eff).float() / H_eff
            coords = torch.stack([x_coords, y_coords], dim=-1) # [Lv, 2]
            
            # 4. 计算欧氏距离平方
            diff = centers[b].unsqueeze(1) - coords.unsqueeze(0) # [K, Lv, 2]
            dist_sq = (diff ** 2).sum(dim=-1) # [K, Lv]
            
            # 5. 高斯掩码与加权池化
            m = torch.exp(-self.gamma * dist_sq) # [K, Lv]
            valid_v_feat = v_pad[b, :Lv, :] # [Lv, D]
            
            numerator = torch.mm(m, valid_v_feat) # [K, D]
            denominator = m.sum(dim=1, keepdim=True) + 1e-6
            v_views[b] = numerator / denominator
            
        return v_views
    
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

        self.samplers = {}
        if Config.train_method == 'asym':
            self.samplers = {
                name: DynamicViewSampler(Config.num_views, Config.gamma).to(self.device_map[name])
                for name in self.models.keys()
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

    def calc_entailment_loss(self, latent_t, latent_v):
        """最小匹配蕴含损失"""
        diff = latent_t.unsqueeze(1) - latent_v 
        entailment_penalty = F.relu(diff).sum(dim=-1) 
        min_penalty, _ = entailment_penalty.min(dim=1) 
        return min_penalty.mean()

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
        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, grid_thws, v_len_cpu in pbar:
            
            # 依次在三个 GPU 上发起计算指令（底层完全异步并发执行）
            for name in ["SAE_V", "SAE_D", "VL_SAE"]:
                target_device = self.device_map[name]
                
                # 数据被发送到当前模型专属的 GPU
                v_pad = v_pad_cpu.to(target_device, non_blocking=True)
                t_pad = t_pad_cpu.to(target_device, non_blocking=True)
                v_mask = v_mask_cpu.to(target_device, non_blocking=True)
                t_mask = t_mask_cpu.to(target_device, non_blocking=True)
                v_len = v_len_cpu.to(target_device, non_blocking=True)

                self.optimizers[name].zero_grad()
                
                with autocast('cuda'):
                    v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                    if not Config.train_method == 'asym':
                        align_loss = batch_filip_loss(v_proj, t_proj, v_mask, t_mask)
                        
                        v_proj_flat = v_proj[v_mask]
                        t_proj_flat = t_proj[t_mask]
                        
                        recon_v, recon_t, _, _ = self.models[name](vision_embeddings=v_proj_flat, text_embeddings=t_proj_flat)
                        
                        # 严谨的数学隔离：切断 SAE 重建误差向对齐投影层的回传
                        recon_loss = self.criterion(recon_v, v_proj_flat.detach()) + self.criterion(recon_t, t_proj_flat.detach())
                        
                        loss = recon_loss + 0.1 * align_loss

                    else:
                        # 【分支 A：非对称动态视图蕴含】
                        t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
                        t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                        v_views = self.samplers[name](v_proj, v_len, grid_thws) # [B, K, D]
                        
                        # 把这三种架构当做黑盒直接输入
                        recon_v, recon_t, latent_v, latent_t = self.models[name](vision_embeddings=v_views, text_embeddings=t_global)
                        
                        loss_rv = self.criterion(recon_v, v_views.detach())
                        loss_rt = self.criterion(recon_t, t_global.detach())
                        loss_align = self.calc_entailment_loss(latent_t, latent_v)
                        loss = loss_rv + loss_rt + Config.lambda_align * loss_align
                    
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
        # 在文件名中加入训练方法，防止对比模型被互相覆盖
        method_str = f"{Config.train_method}_"
        for name, model in self.models.items():
            path = os.path.join(Config.save_dir, f"{name}_{method_str}{suffix}")
            torch.save(model.state_dict(), path)
            print(f"[Trainer] Checkpoint saved: {path}")