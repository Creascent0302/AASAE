import os
import sys
import json
import datetime
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure block_trainer modules are on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_trainer.config import Config
from block_trainer.extractor import FeatureExtractor
from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE, TokenAuxProj
from block_trainer.trainer import PairDataset, collate_fn, DynamicViewSampler


class SAEEvaluator:
    def __init__(
        self,
        method_name,
        eval_batch_size=8,
        max_tokens=0,
        report_file="evaluation_results_all.txt",
        score_align_weight=0.6,
        score_entail_weight=0.4,
    ):
        self.method = method_name
        self.eval_batch_size = int(eval_batch_size)
        self.max_tokens = int(max_tokens)
        self.report_file = report_file
        self.score_align_weight = float(score_align_weight)
        self.score_entail_weight = float(score_entail_weight)

        Config.train_method = self.method
        print(f"\n[Evaluator] Initializing Evaluation for Method: {self.method.upper()} | Top-K: {Config.topk}")

        def _pick_device(idx):
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                return torch.device(f"cuda:{min(idx, max(count - 1, 0))}")
            return torch.device("cpu")

        self.device_map = {
            "SAE_V": _pick_device(1),
            "SAE_D": _pick_device(2),
            "VL_SAE": _pick_device(3),
        }

        self.models = {
            "SAE_V": SAE_V(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_V"]),
            "SAE_D": SAE_D(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["SAE_D"]),
            "VL_SAE": VL_SAE(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(self.device_map["VL_SAE"]),
        }

        if self.method == "asym":
            self.samplers = {
                name: DynamicViewSampler(Config.num_views, Config.gamma).to(self.device_map[name])
                for name in self.models.keys()
            }
        else:
            self.samplers = {}

        self.aux_projs = {}
        shared_aux_path = os.path.join(Config.save_dir, f"shared_best_aux_proj_{self.method}.pth")
        if os.path.exists(shared_aux_path):
            shared_state_dict = torch.load(shared_aux_path, map_location="cpu", weights_only=True)
            for name, device in self.device_map.items():
                aux = TokenAuxProj(Config.qwen_hidden_dim).to(device)
                aux.load_state_dict(shared_state_dict)
                aux.eval()
                self.aux_projs[name] = aux
            print(f"  - Loaded shared AuxProj: {shared_aux_path}")
        else:
            print(f"  - [ERROR] Shared AuxProj not found: {shared_aux_path}")
            sys.exit(1)

        for name in self.models.keys():
            sae_path = os.path.join(Config.save_dir, f"{name}_{self.method}_new_best_sae.pth")
            if not os.path.exists(sae_path):
                sae_path = os.path.join(Config.save_dir, f"{name}_{self.method}_best_sae.pth")

            if os.path.exists(sae_path):
                ckpt = torch.load(sae_path, map_location=self.device_map[name], weights_only=False)
                self.models[name].load_state_dict(ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt)
                self.models[name].eval()
                print(f"  - Loaded {name} from {sae_path}")
            else:
                print(f"  - [ERROR] SAE Checkpoint not found: {sae_path}")
                sys.exit(1)

        self.metrics = {
            name: {
                "mse_v": 0.0,
                "mse_t": 0.0,
                "sum_v": 0.0,
                "sum_sq_v": 0.0,
                "count_v": 0,
                "sum_t": 0.0,
                "sum_sq_t": 0.0,
                "count_t": 0,
                "entailment_ratio_sum": 0.0,
                "entailment_ratio_count": 0,
                "align_i2t_r1": 0,
                "align_i2t_r5": 0,
                "align_t2i_r1": 0,
                "align_t2i_r5": 0,
                "align_total": 0,
                "align_pos_sum": 0.0,
                "align_pos_count": 0,
                "cosine_sim": 0.0,
                "active_latents_v": torch.zeros(Config.sae_hidden_dim, dtype=torch.bool, device=self.device_map[name]),
                "active_latents_t": torch.zeros(Config.sae_hidden_dim, dtype=torch.bool, device=self.device_map[name]),
                "samples": 0,
            }
            for name in self.models.keys()
        }

    def _select_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.max_tokens <= 0 or tokens.size(0) <= self.max_tokens:
            return tokens
        scores = tokens.pow(2).sum(dim=-1)
        topk = min(self.max_tokens, scores.size(0))
        idx = torch.topk(scores, k=topk, dim=0).indices
        return tokens[idx]

    def _update_retrieval(self, metric, score_matrix: torch.Tensor):
        B = score_matrix.size(0)
        if B == 0:
            return
        labels = torch.arange(B, device=score_matrix.device)
        top1_i2t = (score_matrix.argmax(dim=1) == labels).sum().item()
        top1_t2i = (score_matrix.argmax(dim=0) == labels).sum().item()
        k = min(5, B)
        top5_i2t = (score_matrix.topk(k=k, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        top5_t2i = (score_matrix.topk(k=k, dim=0).indices == labels.unsqueeze(0)).any(dim=0).sum().item()

        metric["align_i2t_r1"] += top1_i2t
        metric["align_i2t_r5"] += top5_i2t
        metric["align_t2i_r1"] += top1_t2i
        metric["align_t2i_r5"] += top5_t2i
        metric["align_total"] += B
        metric["align_pos_sum"] += score_matrix.diag().sum().item()
        metric["align_pos_count"] += B

    def _cosine_score_matrix(self, v_latent: torch.Tensor, t_latent: torch.Tensor) -> torch.Tensor:
        v_norm = F.normalize(v_latent, dim=-1)
        t_norm = F.normalize(t_latent, dim=-1)
        return v_norm @ t_norm.transpose(0, 1)

    def _filip_score_matrix(self, v_latents: List[torch.Tensor], t_latents: List[torch.Tensor]) -> torch.Tensor:
        B = len(v_latents)
        if B == 0:
            return torch.zeros(0, 0)
        scores = torch.zeros(B, B, device=v_latents[0].device)
        for i in range(B):
            v_i = self._select_tokens(v_latents[i])
            if v_i.numel() == 0:
                continue
            v_i = F.normalize(v_i, dim=-1)
            for j in range(B):
                t_j = self._select_tokens(t_latents[j])
                if t_j.numel() == 0:
                    continue
                t_j = F.normalize(t_j, dim=-1)
                sim = v_i @ t_j.transpose(0, 1)
                sim_v = sim.max(dim=1).values.mean()
                sim_t = sim.max(dim=0).values.mean()
                scores[i, j] = (sim_v + sim_t) / 2.0
        return scores

    def _asym_score_matrix(self, v_latent: torch.Tensor, t_latent: torch.Tensor) -> torch.Tensor:
        B, _, _ = v_latent.shape
        v_norm = F.normalize(v_latent, dim=-1)
        t_norm = F.normalize(t_latent, dim=-1)
        scores = torch.zeros(B, B, device=v_latent.device)
        for i in range(B):
            sim = torch.einsum("kd,bd->kb", v_norm[i], t_norm)
            scores[i] = sim.max(dim=0).values
        return scores

    @torch.inference_mode()
    def evaluate_chunk(self, chunk_path):
        data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        dataset = PairDataset(data)
        batch_size = min(self.eval_batch_size, 16) if self.eval_batch_size > 0 else min(Config.batch_size, 16)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

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

                if self.method == "sym":
                    v_pool = (v_pad * v_mask.unsqueeze(-1)).sum(dim=1) / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
                    t_pool = (t_pad * t_mask.unsqueeze(-1)).sum(dim=1) / (t_mask.sum(dim=1, keepdim=True) + 1e-6)

                    v_global, t_global = self.aux_projs[name](v_pool, t_pool)
                    recon_v, recon_t, latent_v, latent_t = model(vision_embeddings=v_global, text_embeddings=t_global)

                    m["mse_v"] += F.mse_loss(recon_v, v_global, reduction="sum").item()
                    m["sum_v"] += v_global.sum().item()
                    m["sum_sq_v"] += (v_global ** 2).sum().item()
                    m["count_v"] += v_global.numel()

                    m["mse_t"] += F.mse_loss(recon_t, t_global, reduction="sum").item()
                    m["sum_t"] += t_global.sum().item()
                    m["sum_sq_t"] += (t_global ** 2).sum().item()
                    m["count_t"] += t_global.numel()

                    diff = latent_t - latent_v
                    penalty = F.relu(diff).sum(dim=-1)
                    denom = latent_t.abs().sum(dim=-1).clamp(min=1e-6)
                    m["entailment_ratio_sum"] += (penalty / denom).sum().item()
                    m["entailment_ratio_count"] += B

                    score_matrix = self._cosine_score_matrix(latent_v, latent_t)
                    self._update_retrieval(m, score_matrix)
                    m["cosine_sim"] += F.cosine_similarity(latent_t, latent_v, dim=-1).sum().item()

                    m["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))

                elif self.method == "filip":
                    v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)

                    v_proj_flat = v_proj[v_mask]
                    t_proj_flat = t_proj[t_mask]
                    recon_v, recon_t, latent_v_flat, latent_t_flat = model(
                        vision_embeddings=v_proj_flat, text_embeddings=t_proj_flat
                    )

                    m["mse_v"] += F.mse_loss(recon_v, v_proj_flat, reduction="sum").item()
                    m["sum_v"] += v_proj_flat.sum().item()
                    m["sum_sq_v"] += (v_proj_flat ** 2).sum().item()
                    m["count_v"] += v_proj_flat.numel()

                    m["mse_t"] += F.mse_loss(recon_t, t_proj_flat, reduction="sum").item()
                    m["sum_t"] += t_proj_flat.sum().item()
                    m["sum_sq_t"] += (t_proj_flat ** 2).sum().item()
                    m["count_t"] += t_proj_flat.numel()

                    batch_cosine_sim = 0.0
                    batch_penalty_ratio = 0.0
                    v_latents_list = []
                    t_latents_list = []

                    v_idx, t_idx = 0, 0
                    for b in range(B):
                        lv = v_mask[b].sum().item()
                        lt = t_mask[b].sum().item()

                        lv_latents = latent_v_flat[v_idx : v_idx + lv]
                        lt_latents = latent_t_flat[t_idx : t_idx + lt]

                        if lv_latents.numel() == 0:
                            lv_latents = latent_v_flat.new_zeros(1, latent_v_flat.size(-1))
                        if lt_latents.numel() == 0:
                            lt_latents = latent_t_flat.new_zeros(1, latent_t_flat.size(-1))

                        sim_matrix = F.cosine_similarity(
                            lv_latents.unsqueeze(1), lt_latents.unsqueeze(0), dim=-1
                        )
                        sim_v = sim_matrix.max(dim=1)[0].mean().item()
                        sim_t = sim_matrix.max(dim=0)[0].mean().item()
                        batch_cosine_sim += (sim_v + sim_t) / 2.0

                        v_union = lv_latents.max(dim=0)[0]
                        diff = lt_latents - v_union.unsqueeze(0)
                        penalty = F.relu(diff).sum()
                        denom = lt_latents.abs().sum().clamp(min=1e-6)
                        batch_penalty_ratio += (penalty / denom).item()

                        v_latents_list.append(lv_latents)
                        t_latents_list.append(lt_latents)

                        v_idx += lv
                        t_idx += lt

                    m["cosine_sim"] += batch_cosine_sim
                    m["entailment_ratio_sum"] += batch_penalty_ratio
                    m["entailment_ratio_count"] += B

                    score_matrix = self._filip_score_matrix(v_latents_list, t_latents_list)
                    self._update_retrieval(m, score_matrix)

                    m["active_latents_v"].logical_or_((latent_v_flat > 1e-5).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t_flat > 1e-5).any(dim=0))

                elif self.method == "asym":
                    v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)

                    t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
                    t_global = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                    v_views = self.samplers[name](v_proj, v_len, grid_thws)

                    recon_v, recon_t, latent_v, latent_t = model(
                        vision_embeddings=v_views, text_embeddings=t_global
                    )

                    m["mse_v"] += F.mse_loss(recon_v, v_views, reduction="sum").item()
                    m["sum_v"] += v_views.sum().item()
                    m["sum_sq_v"] += (v_views ** 2).sum().item()
                    m["count_v"] += v_views.numel()

                    m["mse_t"] += F.mse_loss(recon_t, t_global, reduction="sum").item()
                    m["sum_t"] += t_global.sum().item()
                    m["sum_sq_t"] += (t_global ** 2).sum().item()
                    m["count_t"] += t_global.numel()

                    sim_matrix = F.cosine_similarity(latent_t.unsqueeze(1), latent_v, dim=-1)
                    m["cosine_sim"] += sim_matrix.max(dim=1)[0].sum().item()

                    latent_v_union = latent_v.max(dim=1)[0]
                    diff = latent_t - latent_v_union
                    penalty = F.relu(diff).sum(dim=-1)
                    denom = latent_t.abs().sum(dim=-1).clamp(min=1e-6)
                    m["entailment_ratio_sum"] += (penalty / denom).sum().item()
                    m["entailment_ratio_count"] += B

                    score_matrix = self._asym_score_matrix(latent_v, latent_t)
                    self._update_retrieval(m, score_matrix)

                    m["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=1).any(dim=0))
                    m["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))

    def print_final_report(self):
        report_file = self.report_file
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def log_and_print(message, f):
            print(message)
            f.write(message + "\n")

        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*60}\n")
            log_and_print(f" Test Time: {current_time}", f)
            log_and_print(f" Evaluation Report | Method: {self.method.upper()} | Top-K: {Config.topk}", f)
            f.write(f"{'='*60}\n")

            for name in self.models.keys():
                m = self.metrics[name]
                samples = m["samples"]
                if samples == 0:
                    continue

                mean_v = m["sum_v"] / max(m["count_v"], 1)
                var_v_per_elem = (m["sum_sq_v"] / max(m["count_v"], 1)) - (mean_v ** 2)
                total_var_v = var_v_per_elem * m["count_v"]

                mean_t = m["sum_t"] / max(m["count_t"], 1)
                var_t_per_elem = (m["sum_sq_t"] / max(m["count_t"], 1)) - (mean_t ** 2)
                total_var_t = var_t_per_elem * m["count_t"]

                ev_v = 1.0 - (m["mse_v"] / (total_var_v + 1e-9))
                ev_t = 1.0 - (m["mse_t"] / (total_var_t + 1e-9))

                avg_entailment_ratio = m["entailment_ratio_sum"] / max(m["entailment_ratio_count"], 1)
                avg_cosine = m["cosine_sim"] / samples

                align_total = max(m["align_total"], 1)
                i2t_r1 = m["align_i2t_r1"] / align_total
                i2t_r5 = m["align_i2t_r5"] / align_total
                t2i_r1 = m["align_t2i_r1"] / align_total
                t2i_r5 = m["align_t2i_r5"] / align_total
                align_pos = m["align_pos_sum"] / max(m["align_pos_count"], 1)

                align_r1 = 0.5 * (i2t_r1 + t2i_r1)
                align_r5 = 0.5 * (i2t_r5 + t2i_r5)
                align_score = 0.5 * (align_r1 + align_r5)
                entail_score = 1.0 - avg_entailment_ratio
                primary_score = (self.score_align_weight * align_score) + (self.score_entail_weight * entail_score)

                dead_v = 1.0 - m["active_latents_v"].float().mean().item()
                dead_t = 1.0 - m["active_latents_t"].float().mean().item()

                log_and_print(f"[ {name} ]", f)
                log_and_print(f"  EV (V/T): {ev_v:.4f} / {ev_t:.4f}", f)
                log_and_print(f"  Dead Latents (V/T): {dead_v:.2%} / {dead_t:.2%}", f)
                log_and_print(f"  Align: AvgCos {avg_cosine:.4f} | PosSim {align_pos:.4f}", f)
                log_and_print(f"  Align R@1 (I2T/T2I): {i2t_r1:.4f} / {t2i_r1:.4f}", f)
                log_and_print(f"  Align R@5 (I2T/T2I): {i2t_r5:.4f} / {t2i_r5:.4f}", f)
                log_and_print(f"  Entail Ratio: {avg_entailment_ratio:.4f} | Coverage: {entail_score:.4f}", f)
                log_and_print(
                    f"  Primary Score: {primary_score:.4f} (Align {self.score_align_weight:.2f} + Entail {self.score_entail_weight:.2f})",
                    f,
                )
                f.write("-" * 55 + "\n")

        print(f"\nReport appended to: {os.path.abspath(report_file)}")


def run_evaluation_pipeline(
    json_path,
    method_name,
    chunk_size=100,
    eval_batch_size=8,
    max_items=0,
    max_tokens=0,
    report_file="evaluation_results_all.txt",
    score_align_weight=0.6,
    score_entail_weight=0.4,
):
    Config.train_method = method_name
    extractor = FeatureExtractor()
    evaluator = SAEEvaluator(
        method_name=method_name,
        eval_batch_size=eval_batch_size,
        max_tokens=max_tokens,
        report_file=report_file,
        score_align_weight=score_align_weight,
        score_entail_weight=score_entail_weight,
    )

    print(f"\n[*] Reading test JSON from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if max_items > 0:
        dataset = dataset[:max_items]
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
    import argparse

    parser = argparse.ArgumentParser(description="SAE Evaluation Pipeline")
    parser.add_argument("--test-json", default="/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_test_short.json")
    parser.add_argument("--model-path", default=Config.model_path)
    parser.add_argument("--image-folder", default=Config.image_folder)
    parser.add_argument("--save-dir", default=Config.save_dir)
    parser.add_argument("--target-layer-name", default=Config.target_layer_name)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--methods", default="sym,filip,asym")
    parser.add_argument("--topk-sym", type=int, default=256)
    parser.add_argument("--topk-filip", type=int, default=512)
    parser.add_argument("--topk-asym", type=int, default=512)
    parser.add_argument("--report-file", default="evaluation_results_all.txt")
    parser.add_argument("--score-align-weight", type=float, default=0.6)
    parser.add_argument("--score-entail-weight", type=float, default=0.4)
    args = parser.parse_args()

    Config.model_path = args.model_path
    Config.image_folder = args.image_folder
    Config.save_dir = args.save_dir
    Config.target_layer_name = args.target_layer_name

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    topk_map = {
        "sym": args.topk_sym,
        "filip": args.topk_filip,
        "asym": args.topk_asym,
    }

    for name in methods:
        if name not in topk_map:
            print(f"[WARN] Unknown method '{name}', skipping.")
            continue
        Config.topk = topk_map[name]
        run_evaluation_pipeline(
            args.test_json,
            method_name=name,
            chunk_size=args.chunk_size,
            eval_batch_size=args.eval_batch_size,
            max_items=args.max_items,
            max_tokens=args.max_tokens,
            report_file=args.report_file,
            score_align_weight=args.score_align_weight,
            score_entail_weight=args.score_entail_weight,
        )
