import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保能正确导入你 block_trainer 里的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_trainer.config import Config
from block_trainer.extractor import FeatureExtractor
from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE, TokenAuxProj
from block_trainer.trainer import PairDataset, collate_fn, DynamicViewSampler

class SAEEvaluator:
    def __init__(self, method_name):
        self.method = method_name
        Config.train_method = self.method # 确保底层逻辑与当前测试方法一致
        print(f"\n[Evaluator] Initializing Multi-GPU Evaluation for Method: {self.method.upper()}")
        
        self.device_map = {
            "SAE_V": torch.device("cuda:1"),
            "SAE_D": torch.device("cuda:2"),
            "VL_SAE": torch.device("cuda:3")
        }
        
        # 1. 初始化模型
        self.models = {
            "SAE_V": SAE_V(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_V"]),
            "SAE_D": SAE_D(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_D"]),
            "VL_SAE": VL_SAE(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["VL_SAE"])
        }
        
        if self.method == 'asym':
            self.samplers = {
                name: DynamicViewSampler(Config.num_views, Config.gamma).to(self.device_map[name])
                for name in self.models.keys()
            }

        # 2. 联合加载权重 (严格匹配最新 trainer.py 的保存逻辑)
        self.aux_projs = {}
        
        # 2.1 加载唯一的共享 AuxProj
        shared_aux_path = os.path.join(Config.save_dir, f"shared_best_aux_proj_{self.method}.pth")
        if os.path.exists(shared_aux_path):
            shared_state_dict = torch.load(shared_aux_path, map_location='cpu', weights_only=True)
            for name, device in self.device_map.items():
                aux = TokenAuxProj(Config.qwen_hidden_dim).to(device)
                aux.load_state_dict(shared_state_dict)
                aux.eval()
                self.aux_projs[name] = aux
            print(f"  - ✅ Loaded SHARED AuxProj from {shared_aux_path}")
        else:
            print(f"  - ❌ [ERROR] Shared AuxProj not found: {shared_aux_path}")
            sys.exit(1)
            
        # 2.2 加载各自独立的 SAE 权重
        for name in self.models.keys():
            sae_path = os.path.join(Config.save_dir, f"{name}_{self.method}_best_sae.pth")
            if os.path.exists(sae_path):
                ckpt = torch.load(sae_path, map_location=self.device_map[name], weights_only=False)
                self.models[name].load_state_dict(ckpt['sae_state_dict'] if 'sae_state_dict' in ckpt else ckpt)
                self.models[name].eval()
                print(f"  - ✅ Loaded {name} from {sae_path}")
            else:
                print(f"  - ❌ [ERROR] SAE Checkpoint not found: {sae_path}")
                sys.exit(1)

        # 3. 初始化全局指标追踪器（已修复统计学 Bug：使用 sum 和 sum_sq 追踪全局方差）
        self.metrics = {
            name: {
                "mse_v": 0.0, "mse_t": 0.0,
                "sum_v": 0.0, "sum_sq_v": 0.0, "count_v": 0,
                "sum_t": 0.0, "sum_sq_t": 0.0, "count_t": 0,
                "entailment_penalty": 0.0,
                "cosine_sim": 0.0,
                "active_latents_v": torch.zeros(Config.sae_hidden_dim, dtype=torch.bool, device=self.device_map[name]),
                "active_latents_t": torch.zeros(Config.sae_hidden_dim, dtype=torch.bool, device=self.device_map[name]),
                "samples": 0
            } for name in self.models.keys()
        }

    @torch.inference_mode()
    def evaluate_chunk(self, chunk_path):
        data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        dataset = PairDataset(data)
        loader = DataLoader(dataset, batch_size=min(Config.batch_size, 16), collate_fn=collate_fn)

        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, grid_thws, v_len_cpu in loader:
            B = v_pad_cpu.size(0)

            for name, model in self.models.items():
                device = self.device_map[name]
                m = self.metrics[name]
                m["samples"] += B

                v_pad = v_pad_cpu.to(device, non_blocking=True)
                t_pad = t_pad_cpu.to(device, non_blocking=True)
                v_mask = v_mask_cpu.to(device, non_blocking=True)
                t_mask = t_mask_cpu.to(device, non_blocking=True)
                v_len = v_len_cpu.to(device, non_blocking=True)

                if self.method == 'sym':
                    # 【SYM】：投影前压缩 -> 投影 -> SAE
                    v_pool = (v_pad * v_mask.unsqueeze(-1)).sum(dim=1) / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
                    t_pool = (t_pad * t_mask.unsqueeze(-1)).sum(dim=1) / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                    
                    v_global, t_global = self.aux_projs[name](v_pool, t_pool)
                    recon_v, recon_t, latent_v, latent_t = model(vision_embeddings=v_global, text_embeddings=t_global)

                    # 记录误差和方差追踪值
                    m["mse_v"] += F.mse_loss(recon_v, v_global, reduction='sum').item()
                    m["sum_v"] += v_global.sum().item()
                    m["sum_sq_v"] += (v_global ** 2).sum().item()
                    m["count_v"] += v_global.numel()

                    m["mse_t"] += F.mse_loss(recon_t, t_global, reduction='sum').item()
                    m["sum_t"] += t_global.sum().item()
                    m["sum_sq_t"] += (t_global ** 2).sum().item()
                    m["count_t"] += t_global.numel()

                    # 对称蕴含惩罚与相似度
                    m["entailment_penalty"] += F.relu(latent_t - latent_v).sum(dim=-1).sum().item()
                    m["cosine_sim"] += F.cosine_similarity(latent_t, latent_v, dim=-1).sum().item()
                    
                    m["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))

                elif self.method == 'filip':
                    # 【FILIP】：投影 -> 展平测试 EV，同时做全局池化测算交叉相似度
                    v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                    
                    v_proj_flat = v_proj[v_mask]
                    t_proj_flat = t_proj[t_mask]
                    recon_v, recon_t, _, _ = model(vision_embeddings=v_proj_flat, text_embeddings=t_proj_flat)

                    m["mse_v"] += F.mse_loss(recon_v, v_proj_flat, reduction='sum').item()
                    m["sum_v"] += v_proj_flat.sum().item()
                    m["sum_sq_v"] += (v_proj_flat ** 2).sum().item()
                    m["count_v"] += v_proj_flat.numel()

                    m["mse_t"] += F.mse_loss(recon_t, t_proj_flat, reduction='sum').item()
                    m["sum_t"] += t_proj_flat.sum().item()
                    m["sum_sq_t"] += (t_proj_flat ** 2).sum().item()
                    m["count_t"] += t_proj_flat.numel()

                    # 为了测算 FILIP 的跨模态相似度，必须将特征全局池化以解决维度不齐的问题
                    v_global = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1) / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
                    t_global = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1) / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                    _, _, latent_v_global, latent_t_global = model(vision_embeddings=v_global, text_embeddings=t_global)
                    
                    m["entailment_penalty"] += F.relu(latent_t_global - latent_v_global).sum(dim=-1).sum().item()
                    m["cosine_sim"] += F.cosine_similarity(latent_t_global, latent_v_global, dim=-1).sum().item()

                    m["active_latents_v"].logical_or_((latent_v_global > 1e-5).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t_global > 1e-5).any(dim=0))

                elif self.method == 'asym':
                    # 【ASYM】：投影 -> 多视图 -> SAE -> 相对尺度蕴含惩罚
                    v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                    
                    t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
                    t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                    v_views = self.samplers[name](v_proj, v_len, grid_thws) # [B, K, D]

                    recon_v, recon_t, latent_v, latent_t = model(vision_embeddings=v_views, text_embeddings=t_global)

                    m["mse_v"] += F.mse_loss(recon_v, v_views, reduction='sum').item()
                    m["sum_v"] += v_views.sum().item()
                    m["sum_sq_v"] += (v_views ** 2).sum().item()
                    m["count_v"] += v_views.numel()
                    
                    m["mse_t"] += F.mse_loss(recon_t, t_global, reduction='sum').item()
                    m["sum_t"] += t_global.sum().item()
                    m["sum_sq_t"] += (t_global ** 2).sum().item()
                    m["count_t"] += t_global.numel()

                    # 应用严谨的相对尺度蕴含惩罚 (Asym-Norm Entailment) 追踪真实性能
                    t_norm = latent_t / (latent_t.max(dim=-1, keepdim=True)[0] + 1e-8)
                    v_norm = latent_v / (latent_v.max(dim=-1, keepdim=True)[0] + 1e-8)
                    diff = t_norm.unsqueeze(1) - v_norm 
                    penalty = F.relu(diff).sum(dim=-1).min(dim=1)[0].sum().item()
                    m["entailment_penalty"] += penalty

                    m["cosine_sim"] += F.cosine_similarity(latent_t.unsqueeze(1), latent_v, dim=-1).max(dim=1)[0].sum().item()

                    m["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=1).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))

    def print_final_report(self):
        # 准备输出文件路径
        import datetime
        report_file = "evaluation_results_all.txt"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 定义一个内部辅助函数，同时向命令行和文件输出
        def log_and_print(message, f):
            print(message)
            f.write(message + "\n")

        # 以追加模式 'a' 打开文件
        with open(report_file, "a", encoding='utf-8') as f:
            f.write(f"\n\n{'='*60}\n")
            log_and_print(f" 📅 测试时间: {current_time}", f)
            log_and_print(f" 📊 最终测试评估报告 | 方法: {self.method.upper()}", f)
            f.write(f"{'='*60}\n")
            
            for name in self.models.keys():
                m = self.metrics[name]
                samples = m["samples"]
                if samples == 0: continue

                # === 全局方差推导 ===
                mean_v = m["sum_v"] / max(m["count_v"], 1)
                var_v_per_elem = (m["sum_sq_v"] / max(m["count_v"], 1)) - (mean_v ** 2)
                total_var_v = var_v_per_elem * m["count_v"]

                mean_t = m["sum_t"] / max(m["count_t"], 1)
                var_t_per_elem = (m["sum_sq_t"] / max(m["count_t"], 1)) - (mean_t ** 2)
                total_var_t = var_t_per_elem * m["count_t"]

                ev_v = 1.0 - (m["mse_v"] / (total_var_v + 1e-9))
                ev_t = 1.0 - (m["mse_t"] / (total_var_t + 1e-9))

                avg_entailment = m["entailment_penalty"] / samples
                avg_cosine = m["cosine_sim"] / samples

                dead_v = 1.0 - m["active_latents_v"].float().mean().item()
                dead_t = 1.0 - m["active_latents_t"].float().mean().item()

                # 结构化写入
                log_and_print(f"🔹 【 {name} 】 架构表现：", f)
                log_and_print(f"   [重构保真度] 视觉 EV: {ev_v:.4f} | 文本 EV: {ev_t:.4f}", f)
                log_and_print(f"   [概念健康度] 死神经元率 (V): {dead_v:.2%} | (T): {dead_t:.2%}", f)
                log_and_print(f"   [跨模态对齐] 平均余弦相似度: {avg_cosine:.4f}", f)
                log_and_print(f"   [幻觉抑制力] 🌟 非对称蕴含惩罚: {avg_entailment:.4f} (越低越好)", f)
                f.write("-" * 55 + "\n")

        print(f"\n✅ 报告已成功追加至: {os.path.abspath(report_file)}")


def run_evaluation_pipeline(json_path, method_name, chunk_size=100):
    extractor = FeatureExtractor()
    evaluator = SAEEvaluator(method_name=method_name)
    
    print(f"\n[*] Reading test JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    total_images = len(dataset)
    print(f"[*] Total test images: {total_images}")

    for i in range(0, total_images, chunk_size):
        chunk_idx = i // chunk_size
        chunk_data = dataset[i : i + chunk_size]
        
        pt_path = extractor.extract_and_save_chunk(chunk_data, chunk_idx)
        
        if pt_path and os.path.exists(pt_path):
            evaluator.evaluate_chunk(pt_path)
            
            os.remove(pt_path)
            print(f"[Pipeline] Removed temp file {pt_path} to save space.")
            
    evaluator.print_final_report()

if __name__ == "__main__":
    TEST_JSON_PATH = "/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_test_short.json"
    
    # 你可以写一个循环在这里依次测试 'sym', 'filip', 'asym'
    for name in ['sym', 'filip', 'asym']:
        run_evaluation_pipeline(TEST_JSON_PATH, method_name=name, chunk_size=100)