import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
try:
    from config import Config
    from sae_model import SAE_V, SAE_D, VL_SAE, TokenAuxProj
except ImportError:
    from block_trainer.config import Config
    from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE, TokenAuxProj
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
    """内存优化版：Token 级细粒度对比损失"""
    B, Lv, D = v_proj.shape
    _, Lt, _ = t_proj.shape

    v_norm = F.normalize(v_proj, dim=-1)
    t_norm = F.normalize(t_proj, dim=-1)

    align_score = torch.zeros(B, B, device=v_proj.device)

    # 将 4D 巨型张量拆解，遍历当前 Batch 中的每一张图像 b，计算它与所有文本 c 的相似度
    for b in range(B):
        # sim_b: [Lv, B, Lt]
        sim_b = torch.einsum('i d, c j d -> i c j', v_norm[b], t_norm) / temp
        
        # 1. 计算图像 b 到 所有文本 的对齐分数 (align_v)
        sim_v = sim_b.clone()
        sim_v.masked_fill_(~t_mask.unsqueeze(0), -1e4) # mask_shape: [1, B, Lt]
        max_sim_v = sim_v.max(dim=2)[0] # [Lv, B]
        
        valid_v_len = v_mask[b].sum()
        if valid_v_len > 0:
            align_v_b = (max_sim_v * v_mask[b].unsqueeze(1)).sum(dim=0) / valid_v_len
        else:
            align_v_b = torch.zeros(B, device=v_proj.device)

        # 2. 计算所有文本 到 图像 b 的对齐分数 (align_t)
        sim_t = sim_b.clone()
        sim_t.masked_fill_(~v_mask[b].view(Lv, 1, 1), -1e4)
        max_sim_t = sim_t.max(dim=0)[0] # [B, Lt]
        align_t_b = (max_sim_t * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp(min=1)

        # 综合双向得分
        align_score[b, :] = (align_v_b + align_t_b) / 2.0

    labels = torch.arange(B, device=v_proj.device)
    loss = F.cross_entropy(align_score, labels) + F.cross_entropy(align_score.t(), labels)
    return loss / 2.0
    
def global_contrastive_loss(v_global, t_global, temp=0.07):
    v_norm = F.normalize(v_global, dim=-1)
    t_norm = F.normalize(t_global, dim=-1)
    
    sim_matrix = torch.matmul(v_norm, t_norm.transpose(0, 1)) / temp
    labels = torch.arange(v_global.shape[0], device=v_global.device)
    loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.transpose(0, 1), labels)
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
        v_views = torch.zeros(B, self.num_views, D, device=device, dtype=v_pad.dtype) # 预留空间
        centers = torch.rand(B, self.num_views, 2, device=device) # 随机采点
        
        for b in range(B):
            if grid_thws[b] is None:
                v_views[b] = v_pad[b, 0, :].unsqueeze(0).repeat(self.num_views, 1)
                continue
            
            # 1. 获取实际序列长度与原始网格比例
            Lv = v_len[b].item()
            H, W = grid_thws[b][1].item(), grid_thws[b][2].item()
            
            # 2. 动态计算等效的 2D 尺寸 (完美兼容所有的空间池化与特殊 Token)
            # $W / H$ 是图像真实的物理宽高比（Aspect Ratio）。既然总面积是 $L_v$，且比例是 $W/H$，根据面积公式 $\text{Width} \times \text{Height} = \text{Area}$，我们可以逆推出等效宽度 $W_{eff} = \sqrt{L_v \times (W/H)}$。
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

class AuxProjTrainer:
    def __init__(self):
        print(f"\n[Phase 1: AuxProj] Initializing SHARED Token Alignment Training ({Config.train_method.upper()})...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.shared_aux_proj = TokenAuxProj(Config.qwen_hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.shared_aux_proj.parameters(), lr=Config.initial_lr, weight_decay=Config.weight_decay)
        self.scaler = GradScaler('cuda')
        self.best_val_loss = float('inf')

    def _compute_loss(self, v_proj, t_proj, v_mask, t_mask):
        if Config.train_method == 'sym':
            # 原始方法：先取平均，再算全局对比损失
            v_sum = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1)
            v_global = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
            
            t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
            t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
            
            return global_contrastive_loss(v_global, t_global)
        else:
            # FILIP 和 ASYM 方法：保留序列结构，算 Token 级损失
            return batch_filip_loss(v_proj, t_proj, v_mask, t_mask)

    def train_on_chunk(self, train_chunk_path, val_chunk_path, chunk_idx):
        print(f"\n[Phase 1] Training SHARED AuxProj on Chunk {chunk_idx} | Validating on fixed Val_Chunk...")
        train_data = torch.load(train_chunk_path, map_location='cpu', weights_only=False)
        val_data = torch.load(val_chunk_path, map_location='cpu', weights_only=False)
        
        train_loader = DataLoader(PairDataset(train_data), batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(PairDataset(val_data), batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # 1. 训练过程
        self.shared_aux_proj.train()
        pbar = tqdm(train_loader, desc=f"Phase 1: Training (Chunk {chunk_idx})")
        
        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _ in pbar:
            try:
                v_pad = v_pad_cpu.to(self.device, non_blocking=True)
                t_pad = t_pad_cpu.to(self.device, non_blocking=True)
                v_mask = v_mask_cpu.to(self.device, non_blocking=True)
                t_mask = t_mask_cpu.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                with autocast('cuda'):
                    v_proj, t_proj = self.shared_aux_proj(v_pad, t_pad)
                    loss = self._compute_loss(v_proj, t_proj, v_mask, t_mask)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                pbar.set_postfix({'Align_Loss': f"{loss.item():.4f}"})
                del loss, v_proj, t_proj, v_pad, t_pad
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                continue

        # 2. 验证过程
        self.shared_aux_proj.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _ in tqdm(val_loader, desc="Phase 1: Validating", leave=False):
                try:
                    v_pad = v_pad_cpu.to(self.device, non_blocking=True)
                    t_pad = t_pad_cpu.to(self.device, non_blocking=True)
                    v_mask = v_mask_cpu.to(self.device, non_blocking=True)
                    t_mask = t_mask_cpu.to(self.device, non_blocking=True)
                    
                    with autocast('cuda'):
                        v_proj, t_proj = self.shared_aux_proj(v_pad, t_pad)
                        loss = self._compute_loss(v_proj, t_proj, v_mask, t_mask)
                    val_loss_total += loss.item()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue

        # 3. 保存最佳权重
        os.makedirs(Config.save_dir, exist_ok=True)
        avg_val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"--- Phase 1 Validation Report (Chunk {chunk_idx}) ---")
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            save_path = os.path.join(Config.save_dir, f"shared_best_aux_proj_{Config.train_method}.pth")
            torch.save(self.shared_aux_proj.state_dict(), save_path)
            print(f"  [🌟 Update] AuxProj Loss dropped to {avg_val_loss:.4f} -> Saved best model!")
        else:
            print(f"  [📉 Info] Loss: {avg_val_loss:.4f} (Best: {self.best_val_loss:.4f})")

class SAETrainer:
    def __init__(self):
        print(f"\n[Phase 2: SAE] Initializing SAE Dictionary Training ({Config.train_method.upper()})...")
        self.device_map = {
            "SAE_V": torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"),
            "SAE_D": torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0"),
            "VL_SAE": torch.device("cuda:3" if torch.cuda.device_count() > 3 else "cuda:0")
        }
        
        sae_cfg = {
            "input_unit_norm": Config.input_unit_norm,
            "l1_coeff": Config.l1_coeff,
            "aux_penalty": Config.aux_penalty,
            "top_k_aux": Config.top_k_aux,
            "n_batches_to_dead": Config.n_batches_to_dead,
            "use_threshold_in_eval": Config.use_threshold_in_eval,
        }

        self.models = {
            "SAE_V": SAE_V(
                Config.qwen_hidden_dim,
                Config.sae_hidden_dim,
                Config.topk,
                cfg=sae_cfg,
            ).to(self.device_map["SAE_V"]),
            "SAE_D": SAE_D(
                Config.qwen_hidden_dim,
                Config.sae_hidden_dim,
                Config.topk,
                cfg=sae_cfg,
            ).to(self.device_map["SAE_D"]),
            "VL_SAE": VL_SAE(
                Config.qwen_hidden_dim,
                Config.sae_hidden_dim,
                Config.topk,
                cfg=sae_cfg,
            ).to(self.device_map["VL_SAE"]),
        }
        
        self.aux_projs = {}
        load_path = os.path.join(Config.save_dir, f"shared_best_aux_proj_{Config.train_method}.pth")
        
        if os.path.exists(load_path):
            shared_state_dict = torch.load(load_path, map_location='cpu', weights_only=True)
            for name, device in self.device_map.items():
                aux = TokenAuxProj(Config.qwen_hidden_dim).to(device)
                aux.load_state_dict(shared_state_dict)
                aux.eval()
                for param in aux.parameters():
                    param.requires_grad = False 
                self.aux_projs[name] = aux
        else:
            print(f"  [❌ ERROR] Shared AuxProj weights not found! YOU MUST RUN PHASE 1 FIRST.")
            import sys; sys.exit(1)

        self.samplers = {}
        if Config.train_method == 'asym':
            self.samplers = {
                name: DynamicViewSampler(Config.num_views, Config.gamma).to(self.device_map[name])
                for name in self.models.keys()
            }

        self.optimizers = {
            name: optim.Adam(self.models[name].parameters(), lr=Config.initial_lr, weight_decay=Config.weight_decay) 
            for name in self.models.keys()
        }
        self.scalers = {name: GradScaler('cuda') for name in self.models.keys()}
        self.best_val_loss = {name: float('inf') for name in self.models.keys()}
        self._b_dec_initialized = False

    @torch.no_grad()
    def _init_b_dec_from_val(self, val_chunk_path):
        if self._b_dec_initialized:
            return

        val_data = torch.load(val_chunk_path, map_location='cpu', weights_only=False)
        val_loader = DataLoader(
            PairDataset(val_data),
            batch_size=Config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        device = self.device_map["SAE_V"]
        aux_proj = self.aux_projs["SAE_V"].to(device)
        sampler = self.samplers.get("SAE_V") if Config.train_method == 'asym' else None

        sum_v = None
        sum_t = None
        count_v = 0
        count_t = 0

        for batch_idx, (v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, grid_thws, v_len_cpu) in enumerate(val_loader):
            if batch_idx >= Config.init_b_dec_batches:
                break

            v_pad = v_pad_cpu.to(device, non_blocking=True)
            t_pad = t_pad_cpu.to(device, non_blocking=True)
            v_mask = v_mask_cpu.to(device, non_blocking=True)
            t_mask = t_mask_cpu.to(device, non_blocking=True)
            v_len = v_len_cpu.to(device, non_blocking=True)

            v_proj, t_proj = aux_proj(v_pad, t_pad)

            if Config.train_method == 'sym':
                v_sum = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1)
                v_inputs = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)

                t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
                t_inputs = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
            elif Config.train_method == 'filip':
                v_inputs = v_proj[v_mask]
                t_inputs = t_proj[t_mask]
            else:
                t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
                t_inputs = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                v_views = sampler(v_proj, v_len, grid_thws)
                v_inputs = v_views.reshape(-1, v_views.shape[-1])

            if v_inputs.numel() > 0:
                sum_v = v_inputs.sum(dim=0) if sum_v is None else sum_v + v_inputs.sum(dim=0)
                count_v += v_inputs.shape[0]
            if t_inputs.numel() > 0:
                sum_t = t_inputs.sum(dim=0) if sum_t is None else sum_t + t_inputs.sum(dim=0)
                count_t += t_inputs.shape[0]

        mean_v = (sum_v / max(count_v, 1)).detach().cpu() if sum_v is not None else None
        mean_t = (sum_t / max(count_t, 1)).detach().cpu() if sum_t is not None else None

        for model in self.models.values():
            model.set_b_dec_from_mean(mean_v, mean_t)

        self._b_dec_initialized = True

    def calc_entailment_loss(self, latent_t, latent_v):
        # 文本存在的概念在视觉的多视图并集中存在即可，不要求完全匹配（即允许视觉概念冗余，但不允许文本概念缺失）
        t_detached = latent_t.detach()

        v_union = latent_v.max(dim=1)[0]
        # 相对尺度归一化，防止绝对幅度差异导致惩罚爆炸
        # t_norm = t_detached / (t_detached.max(dim=-1, keepdim=True)[0] + 1e-8)
        # v_norm = v_union / (v_union.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        diff = t_detached - v_union 
        entailment_penalty = F.relu(diff).sum(dim=-1) 
        
        return entailment_penalty.mean()

    def _compute_sae_loss(self, name, v_proj, t_proj, v_mask, t_mask, v_len, grid_thws):
        if Config.train_method == 'sym':
            # === SYM 方法 ===
            v_sum = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1)
            v_global = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
            t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
            t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
            
            recon_v, recon_t, _, _, loss_v, loss_t = self.models[name](
                vision_embeddings=v_global,
                text_embeddings=t_global,
                return_loss=True,
            )
            loss_rv = loss_v["loss"] if loss_v is not None else 0.0
            loss_rt = loss_t["loss"] if loss_t is not None else 0.0
            return loss_rv + loss_rt
            
        elif Config.train_method == 'filip':
            # === FILIP 方法 ===
            # v_proj_flat = v_proj[v_mask]
            # t_proj_flat = t_proj[t_mask]
            
            # recon_v, recon_t, _, _ = self.models[name](vision_embeddings=v_proj_flat, text_embeddings=t_proj_flat)
            # return self.criterion(recon_v, v_proj_flat) + self.criterion(recon_t, t_proj_flat)
            v_proj_flat = v_proj[v_mask]
            t_proj_flat = t_proj[t_mask]
            
            recon_v, recon_t, latent_v, latent_t, loss_v, loss_t = self.models[name](
                vision_embeddings=v_proj_flat,
                text_embeddings=t_proj_flat,
                return_loss=True,
            )

            loss_rv = loss_v["loss"] if loss_v is not None else 0.0
            loss_rt = loss_t["loss"] if loss_t is not None else 0.0
            
            # 【核心修正】：为 FILIP 加入严谨的 Token 级特征并集惩罚
            penalty = 0.0
            v_idx, t_idx = 0, 0
            for b in range(v_proj.size(0)): # 遍历 Batch 中的每个样本
                lv = v_mask[b].sum().item()
                lt = t_mask[b].sum().item()
                
                lv_latents = latent_v[v_idx : v_idx+lv]
                lt_latents = latent_t[t_idx : t_idx+lt]
                
                # 图像所有 Token 的概念并集
                v_union = lv_latents.max(dim=0)[0] 
                
                # 文本的每一个 Token 的概念，都应该在图像的并集中找到
                diff = lt_latents.detach() - v_union.unsqueeze(0)
                
                # 求单个样本内所有文本 Token 惩罚的平均值
                penalty += F.relu(diff).sum(dim=-1).mean() 
                
                v_idx += lv
                t_idx += lt
                
            loss_align = penalty / v_proj.size(0) # 平均到 Batch
            return loss_rv + loss_rt + Config.lambda_align * loss_align
            
        elif Config.train_method == 'asym':
            # === ASYM 方法 ===
            t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
            t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
            v_views = self.samplers[name](v_proj, v_len, grid_thws)
            
            recon_v, recon_t, latent_v, latent_t, loss_v, loss_t = self.models[name](
                vision_embeddings=v_views,
                text_embeddings=t_global,
                return_loss=True,
            )

            loss_rv = loss_v["loss"] if loss_v is not None else 0.0
            loss_rt = loss_t["loss"] if loss_t is not None else 0.0
            loss_align = self.calc_entailment_loss(latent_t, latent_v)
            
            return loss_rv + loss_rt + Config.lambda_align * loss_align

    def train_on_chunk(self, train_chunk_path, val_chunk_path, chunk_idx):
        print(f"\n[Phase 2] SAE Training on Chunk {chunk_idx}...")
        self._init_b_dec_from_val(val_chunk_path)
        train_data = torch.load(train_chunk_path, map_location='cpu', weights_only=False)
        val_data = torch.load(val_chunk_path, map_location='cpu', weights_only=False)
        
        train_loader = DataLoader(PairDataset(train_data), batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(PairDataset(val_data), batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # 1. 训练阶段
        for model in self.models.values(): model.train()
        pbar = tqdm(train_loader, desc=f"Phase 2: Training (Chunk {chunk_idx})")
        
        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, grid_thws, v_len_cpu in pbar:
            for name in self.models.keys():
                target_device = self.device_map[name]
                try:
                    v_pad = v_pad_cpu.to(target_device, non_blocking=True)
                    t_pad = t_pad_cpu.to(target_device, non_blocking=True)
                    v_mask = v_mask_cpu.to(target_device, non_blocking=True)
                    t_mask = t_mask_cpu.to(target_device, non_blocking=True)
                    v_len = v_len_cpu.to(target_device, non_blocking=True)

                    self.optimizers[name].zero_grad()
                    with autocast('cuda'):
                        with torch.no_grad():
                            v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                        loss = self._compute_sae_loss(name, v_proj, t_proj, v_mask, t_mask, v_len, grid_thws)

                    self.scalers[name].scale(loss).backward()
                    self.scalers[name].unscale_(self.optimizers[name])
                    torch.nn.utils.clip_grad_norm_(self.models[name].parameters(), Config.max_grad_norm)
                    self.models[name].make_decoder_weights_and_grad_unit_norm()
                    self.scalers[name].step(self.optimizers[name])
                    self.scalers[name].update()
                    
                    if name == "VL_SAE": pbar.set_postfix({'SAE_Loss': f"{loss.item():.4f}"})
                    del loss, v_proj, t_proj, v_pad, t_pad
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.optimizers[name].zero_grad()
                    continue

        # 2. 验证阶段
        for model in self.models.values(): model.eval()
        val_losses = {name: 0.0 for name in self.models.keys()}
        
        with torch.no_grad():
            for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, grid_thws, v_len_cpu in tqdm(val_loader, desc="Phase 2: Validating", leave=False):
                for name, target_device in self.device_map.items():
                    try:
                        v_pad = v_pad_cpu.to(target_device, non_blocking=True)
                        t_pad = t_pad_cpu.to(target_device, non_blocking=True)
                        v_mask = v_mask_cpu.to(target_device, non_blocking=True)
                        t_mask = t_mask_cpu.to(target_device, non_blocking=True)
                        v_len = v_len_cpu.to(target_device, non_blocking=True)

                        with autocast('cuda'):
                            v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                            loss = self._compute_sae_loss(name, v_proj, t_proj, v_mask, t_mask, v_len, grid_thws)
                            val_losses[name] += loss.item()
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        continue

        # 3. 独立保存最佳权重
        os.makedirs(Config.save_dir, exist_ok=True)
        print(f"--- Phase 2 Validation Report (Chunk {chunk_idx}) ---")
        method_str = f"{Config.train_method}_"
        for name in self.models.keys():
            avg_val_loss = val_losses[name] / len(val_loader) if len(val_loader) > 0 else 0
            if avg_val_loss < self.best_val_loss[name]:
                self.best_val_loss[name] = avg_val_loss
                save_path = os.path.join(Config.save_dir, f"{name}_{method_str}new_best_sae.pth")
                torch.save({'sae_state_dict': self.models[name].state_dict()}, save_path)
                print(f"  [🌟 {name}] SAE Loss dropped to {avg_val_loss:.4f} -> Saved!")
            else:
                print(f"  [📉 {name}] SAE Loss: {avg_val_loss:.4f}")
                