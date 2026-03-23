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
    def __init__(self, method_name, checkpoint_suffix="final.pth"):
        self.method = method_name
        print(f"\n[Evaluator] Initializing Multi-GPU Evaluation for Method: {self.method.upper()}")
        
        # 保持与训练时完全一致的设备映射，防止与 Qwen 抢显存
        self.device_map = {
            "SAE_V": torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"),
            "SAE_D": torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0"),
            "VL_SAE": torch.device("cuda:3" if torch.cuda.device_count() > 3 else "cuda:0")
        }
        
        # 1. 初始化模型与投影层
        self.models = {
            "SAE_V": SAE_V(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_V"]),
            "SAE_D": SAE_D(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_D"]),
            "VL_SAE": VL_SAE(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["VL_SAE"])
        }
        self.aux_projs = {
            name: TokenAuxProj(Config.qwen_hidden_dim).to(self.device_map[name]) for name in self.models.keys()
        }
        
        if self.method == 'asym':
            self.samplers = {
                name: DynamicViewSampler(Config.num_views, Config.gamma).to(self.device_map[name])
                for name in self.models.keys()
            }

        # 2. 联合加载权重 (SAE + AuxProj)
        for name in self.models.keys():
            ckpt_path = os.path.join("../block_trainer/", Config.save_dir, f"{name}_{self.method}_SAE_V_{checkpoint_suffix}")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device_map[name], weights_only=False)
                if 'sae_state_dict' in ckpt:
                    self.models[name].load_state_dict(ckpt['sae_state_dict'])
                    self.aux_projs[name].load_state_dict(ckpt['aux_proj_state_dict'])
                else:
                    print(f"  [警告] {name} 缺失 aux_proj 权重，测试结果将失效！请先修复 trainer 的 save_checkpoint 并重新训练。")
                    self.models[name].load_state_dict(ckpt)
                
                self.models[name].eval()
                self.aux_projs[name].eval()
                print(f"  - ✅ Loaded {name} from {ckpt_path}")
            else:
                print(f"  - ❌ [ERROR] Checkpoint not found: {ckpt_path}")
                sys.exit(1)

        # 3. 初始化全局指标追踪器
        self.metrics = {
            name: {
                "mse_v": 0.0, "mse_t": 0.0,
                "var_v": 0.0, "var_t": 0.0,
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
        # 建议测试时使用较小的 batch_size 防止极端数据导致 OOM
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

                v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)

                if self.method == 'asym':
                    # 【非对称模型测试】
                    t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
                    t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                    v_views = self.samplers[name](v_proj, v_len, grid_thws) # [B, K, D]

                    recon_v, recon_t, latent_v, latent_t = model(vision_embeddings=v_views, text_embeddings=t_global)

                    # MSE 与 方差计算
                    m["mse_v"] += F.mse_loss(recon_v, v_views, reduction='sum').item()
                    m["var_v"] += ((v_views - v_views.mean(dim=0, keepdim=True))**2).sum().item()
                    
                    m["mse_t"] += F.mse_loss(recon_t, t_global, reduction='sum').item()
                    m["var_t"] += ((t_global - t_global.mean(dim=0, keepdim=True))**2).sum().item()

                    # 非对称蕴含惩罚: min_k(ReLU(Z_t - Z_v_k))
                    diff = latent_t.unsqueeze(1) - latent_v  # [B, K, D_sae]
                    penalty = F.relu(diff).sum(dim=-1).min(dim=1)[0].sum().item()
                    m["entailment_penalty"] += penalty

                    # 提取最大匹配的余弦相似度
                    cos_sim = F.cosine_similarity(latent_t.unsqueeze(1), latent_v, dim=-1).max(dim=1)[0].sum().item()
                    m["cosine_sim"] += cos_sim

                    m["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=1).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))

                else:
                    # 【对称基线模型测试】
                    t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
                    t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                    v_sum = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1)
                    v_global = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)

                    recon_v, recon_t, latent_v, latent_t = model(vision_embeddings=v_global, text_embeddings=t_global)

                    m["mse_v"] += F.mse_loss(recon_v, v_global, reduction='sum').item()
                    m["var_v"] += ((v_global - v_global.mean(dim=0, keepdim=True))**2).sum().item()
                    m["mse_t"] += F.mse_loss(recon_t, t_global, reduction='sum').item()
                    m["var_t"] += ((t_global - t_global.mean(dim=0, keepdim=True))**2).sum().item()

                    # 简单的对称蕴含惩罚
                    penalty = F.relu(latent_t - latent_v).sum(dim=-1).sum().item()
                    m["entailment_penalty"] += penalty

                    m["cosine_sim"] += F.cosine_similarity(latent_t, latent_v, dim=-1).sum().item()

                    m["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))
                
                # 释放当前卡的显存
                del v_pad, t_pad, v_proj, t_proj, recon_v, recon_t, latent_v, latent_t

    def print_final_report(self):
        print(f"\n{'='*55}")
        print(f" 📊 最终测试评估报告 | 方法: {self.method.upper()}")
        print(f"{'='*55}")
        
        for name in self.models.keys():
            m = self.metrics[name]
            samples = m["samples"]
            if samples == 0: continue

            # 解释方差 Explained Variance
            ev_v = 1.0 - (m["mse_v"] / (m["var_v"] + 1e-9))
            ev_t = 1.0 - (m["mse_t"] / (m["var_t"] + 1e-9))

            avg_entailment = m["entailment_penalty"] / samples
            avg_cosine = m["cosine_sim"] / samples

            dead_v = 1.0 - m["active_latents_v"].float().mean().item()
            dead_t = 1.0 - m["active_latents_t"].float().mean().item()

            print(f"🔹 【 {name} 】 架构表现：")
            print(f"   [重构保真度] 视觉 EV: {ev_v:.4f} | 文本 EV: {ev_t:.4f}")
            print(f"   [概念健康度] 死神经元率 (V): {dead_v:.2%} | (T): {dead_t:.2%}")
            print(f"   [跨模态对齐] 平均余弦相似度: {avg_cosine:.4f}")
            print(f"   [幻觉抑制力] 🌟 非对称蕴含惩罚: {avg_entailment:.4f} (越低越好)")
            print("-" * 55)

def run_evaluation_pipeline(json_path, method_name, chunk_size=100):
    # 1. 实例化现成的 Extractor (自动加载 Qwen2.5-VL)
    extractor = FeatureExtractor()
    
    # 2. 实例化评估器 (自动加载 SAE 权重)
    evaluator = SAEEvaluator(method_name=method_name)
    
    # 3. 读取 JSON 文件
    print(f"\n[*] Reading test JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    total_images = len(dataset)
    print(f"[*] Total test images: {total_images}")

    # 4. 分块流水线处理
    for i in range(0, total_images, chunk_size):
        chunk_idx = i // chunk_size
        chunk_data = dataset[i : i + chunk_size]
        
        # 4.1 提取特征并保存为临时 .pt 文件 (直接复用你的原版代码)
        pt_path = extractor.extract_and_save_chunk(chunk_data, chunk_idx)
        
        if pt_path and os.path.exists(pt_path):
            # 4.2 运行 SAE 评估
            evaluator.evaluate_chunk(pt_path)
            
            # 4.3 阅后即焚，节省硬盘
            os.remove(pt_path)
            print(f"[Pipeline] Removed temp file {pt_path} to save space.")
            
    # 5. 打印最终合并指标
    evaluator.print_final_report()

if __name__ == "__main__":
    TEST_JSON_PATH = "/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_test.json"
    
    # 根据你需要测试的方法切换 'asym' 或 'sym'
    run_evaluation_pipeline(TEST_JSON_PATH, method_name="asym", chunk_size=100)