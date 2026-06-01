#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_trainer.config import Config
from block_trainer.extractor import FeatureExtractor
from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE, TokenAuxProj
from block_trainer.trainer import PairDataset, collate_fn, DynamicViewSampler


def _parse_list(text: str, cast=float) -> List:
    items = [x.strip() for x in text.split(",") if x.strip()]
    return [cast(x) for x in items]


def _pool_text_tokens(
    t_proj: torch.Tensor,
    t_mask: torch.Tensor,
    pool_mode: str,
    temp: float,
    topk: int = 0,
) -> torch.Tensor:
    if pool_mode == "mean":
        t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
        return t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)

    if temp <= 0:
        temp = 1.0
    scores = t_proj.pow(2).sum(dim=-1)
    scores = scores.masked_fill(~t_mask, -1e9)
    if topk and topk > 0:
        k = min(int(topk), scores.size(1))
        topk_idx = torch.topk(scores, k=k, dim=1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(1, topk_idx, True)
        scores = scores.masked_fill(~keep, -1e9)
    weights = torch.softmax(scores / temp, dim=1)
    return (t_proj * weights.unsqueeze(-1)).sum(dim=1)


def _apply_noise(x: torch.Tensor, std: float, rng: torch.Generator) -> torch.Tensor:
    if std <= 0:
        return x
    return x + torch.randn(x, generator=rng, device=x.device, dtype=x.dtype) * std


def _apply_token_dropout(x: torch.Tensor, mask: torch.Tensor, drop: float, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    if drop <= 0:
        return x, mask
    keep = (torch.rand(mask.shape, device=x.device, generator=rng) > drop) & mask
    # Ensure at least one token kept per sample
    for b in range(keep.size(0)):
        if keep[b].sum() == 0:
            idx = mask[b].nonzero(as_tuple=False)
            if idx.numel() > 0:
                keep[b, idx[0].item()] = True
    x = x * keep.unsqueeze(-1)
    return x, keep


def _apply_token_budget(x: torch.Tensor, mask: torch.Tensor, frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if frac >= 1.0:
        return x, mask
    keep = torch.zeros_like(mask)
    for b in range(mask.size(0)):
        idx = mask[b].nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(math.ceil(idx.numel() * frac)))
        norms = x[b, idx].pow(2).sum(dim=-1)
        topk = torch.topk(norms, k=k).indices
        keep_idx = idx[topk]
        keep[b, keep_idx] = True
    x = x * keep.unsqueeze(-1)
    return x, keep


def _update_retrieval(metric: Dict[str, float], score_matrix: torch.Tensor):
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


def _init_metric(device: torch.device, hidden: int) -> Dict[str, object]:
    return {
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
        "active_latents_v": torch.zeros(hidden, dtype=torch.bool, device=device),
        "active_latents_t": torch.zeros(hidden, dtype=torch.bool, device=device),
        "samples": 0,
    }


def _finalize_metric(metric: Dict[str, object]) -> Dict[str, float]:
    samples = metric["samples"]
    mean_v = metric["sum_v"] / max(metric["count_v"], 1)
    var_v = (metric["sum_sq_v"] / max(metric["count_v"], 1)) - (mean_v ** 2)
    total_var_v = var_v * metric["count_v"]
    ev_v = 1.0 - (metric["mse_v"] / (total_var_v + 1e-9))

    mean_t = metric["sum_t"] / max(metric["count_t"], 1)
    var_t = (metric["sum_sq_t"] / max(metric["count_t"], 1)) - (mean_t ** 2)
    total_var_t = var_t * metric["count_t"]
    ev_t = 1.0 - (metric["mse_t"] / (total_var_t + 1e-9))

    avg_entail = metric["entailment_ratio_sum"] / max(metric["entailment_ratio_count"], 1)
    avg_cos = metric["cosine_sim"] / max(samples, 1)

    align_total = max(metric["align_total"], 1)
    i2t_r1 = metric["align_i2t_r1"] / align_total
    t2i_r1 = metric["align_t2i_r1"] / align_total
    i2t_r5 = metric["align_i2t_r5"] / align_total
    t2i_r5 = metric["align_t2i_r5"] / align_total

    align_r1 = 0.5 * (i2t_r1 + t2i_r1)
    align_r5 = 0.5 * (i2t_r5 + t2i_r5)
    align_score = 0.5 * (align_r1 + align_r5)

    return {
        "ev_v": float(ev_v),
        "ev_t": float(ev_t),
        "dead_v": float(1.0 - metric["active_latents_v"].float().mean().item()),
        "dead_t": float(1.0 - metric["active_latents_t"].float().mean().item()),
        "align_avgcos": float(avg_cos),
        "align_possim": float(metric["align_pos_sum"] / max(metric["align_pos_count"], 1)),
        "r1_i2t": float(i2t_r1),
        "r1_t2i": float(t2i_r1),
        "r5_i2t": float(i2t_r5),
        "r5_t2i": float(t2i_r5),
        "entail_ratio": float(avg_entail),
        "coverage": float(1.0 - avg_entail),
        "align_score": float(align_score),
    }


def _apply_condition(
    v_pad: torch.Tensor,
    t_pad: torch.Tensor,
    v_mask: torch.Tensor,
    t_mask: torch.Tensor,
    condition: Tuple[str, float],
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cond_type, level = condition
    v = v_pad.clone()
    t = t_pad.clone()
    vm = v_mask.clone()
    tm = t_mask.clone()

    if cond_type == "noise":
        v = _apply_noise(v, level, rng)
        t = _apply_noise(t, level, rng)
    elif cond_type == "dropout":
        v, vm = _apply_token_dropout(v, vm, level, rng)
        t, tm = _apply_token_dropout(t, tm, level, rng)
    elif cond_type == "budget":
        v, vm = _apply_token_budget(v, vm, level)
        t, tm = _apply_token_budget(t, tm, level)
    return v, t, vm, tm


def _cosine_score_matrix(v_latent: torch.Tensor, t_latent: torch.Tensor) -> torch.Tensor:
    v_norm = F.normalize(v_latent, dim=-1)
    t_norm = F.normalize(t_latent, dim=-1)
    return v_norm @ t_norm.transpose(0, 1)


def _asym_score_matrix(v_latent: torch.Tensor, t_latent: torch.Tensor) -> torch.Tensor:
    B, _, _ = v_latent.shape
    v_norm = F.normalize(v_latent, dim=-1)
    t_norm = F.normalize(t_latent, dim=-1)
    scores = torch.zeros(B, B, device=v_latent.device)
    for i in range(B):
        sim = torch.einsum("kd,bd->kb", v_norm[i], t_norm)
        scores[i] = sim.max(dim=0).values
    return scores


def _build_models(method: str, model_names: List[str], save_dir: str, device_map: Dict[str, torch.device]):
    models = {}
    for name in model_names:
        if name == "SAE_V":
            model = SAE_V(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(device_map[name])
        elif name == "SAE_D":
            model = SAE_D(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(device_map[name])
        else:
            model = VL_SAE(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk).to(device_map[name])

        ckpt_path = os.path.join(save_dir, f"{name}_{method}_new_best_sae.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(save_dir, f"{name}_{method}_best_sae.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device_map[name], weights_only=False)
        model.load_state_dict(ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt)
        model.eval()
        models[name] = model
    return models


def _build_aux(method: str, model_names: List[str], save_dir: str, device_map: Dict[str, torch.device]):
    shared_aux_path = os.path.join(save_dir, f"shared_best_aux_proj_{method}.pth")
    if not os.path.exists(shared_aux_path):
        raise FileNotFoundError(f"Missing AuxProj checkpoint: {shared_aux_path}")
    shared_state = torch.load(shared_aux_path, map_location="cpu", weights_only=True)

    aux = {}
    for name in model_names:
        proj = TokenAuxProj(Config.qwen_hidden_dim).to(device_map[name])
        proj.load_state_dict(shared_state)
        proj.eval()
        aux[name] = proj
    return aux


def evaluate(
    test_json: str,
    save_dir: str,
    methods: List[str],
    model_names: List[str],
    conditions: List[Tuple[str, float]],
    out_csv: str,
    chunk_size: int,
    eval_batch_size: int,
    seed: int,
):
    Config.train_method = "asym"
    extractor = FeatureExtractor()

    device_map = {
        "SAE_V": torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"),
        "SAE_D": torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0"),
        "VL_SAE": torch.device("cuda:3" if torch.cuda.device_count() > 3 else "cuda:0"),
    }

    evaluators = {}
    samplers = {}
    for method in methods:
        Config.train_method = method
        evaluators[method] = {
            "models": _build_models(method, model_names, save_dir, device_map),
            "aux": _build_aux(method, model_names, save_dir, device_map),
        }
        if method == "asym" and Config.asym_use_views:
            samplers[method] = {
                name: DynamicViewSampler(Config.num_views, Config.gamma).to(device_map[name])
                for name in model_names
            }

    metrics: Dict[Tuple[str, str, str, float], Dict[str, object]] = {}
    for method in methods:
        for cond in conditions:
            for model in model_names:
                metrics[(method, cond[0], f"{cond[1]}", model)] = _init_metric(device_map[model], Config.sae_hidden_dim)

    with open(test_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_images = len(dataset)
    for start in range(0, total_images, chunk_size):
        chunk_idx = start // chunk_size
        chunk_data = dataset[start : start + chunk_size]
        pt_path = extractor.extract_and_save_chunk(chunk_data, chunk_idx)
        if not pt_path or not os.path.exists(pt_path):
            continue
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        loader = DataLoader(PairDataset(data), batch_size=eval_batch_size, collate_fn=collate_fn)

        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, grid_thws, v_len_cpu in tqdm(loader, desc=f"Eval chunk {chunk_idx}"):
            for condition in conditions:
                cond_type, level = condition
                for method in methods:
                    for model_name in model_names:
                        device = device_map[model_name]
                        rng = torch.Generator(device=device)
                        rng.manual_seed(seed + int(abs(hash((cond_type, level, method, model_name))) % 100000))

                        v_pad = v_pad_cpu.to(device, non_blocking=True)
                        t_pad = t_pad_cpu.to(device, non_blocking=True)
                        v_mask = v_mask_cpu.to(device, non_blocking=True)
                        t_mask = t_mask_cpu.to(device, non_blocking=True)
                        v_len = v_len_cpu.to(device, non_blocking=True)

                        v_pad, t_pad, v_mask, t_mask = _apply_condition(v_pad, t_pad, v_mask, t_mask, condition, rng)

                        metric = metrics[(method, cond_type, f"{level}", model_name)]
                        metric["samples"] += v_pad.size(0)

                        model = evaluators[method]["models"][model_name]
                        aux = evaluators[method]["aux"][model_name]

                        with torch.inference_mode():
                            if method == "sym":
                                v_sum = (v_pad * v_mask.unsqueeze(-1)).sum(dim=1)
                                v_pool = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
                                t_sum = (t_pad * t_mask.unsqueeze(-1)).sum(dim=1)
                                t_pool = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
                                v_global, t_global = aux(v_pool, t_pool)

                                recon_v, recon_t, latent_v, latent_t = model(
                                    vision_embeddings=v_global, text_embeddings=t_global
                                )

                                metric["mse_v"] += F.mse_loss(recon_v, v_global, reduction="sum").item()
                                metric["sum_v"] += v_global.sum().item()
                                metric["sum_sq_v"] += (v_global ** 2).sum().item()
                                metric["count_v"] += v_global.numel()

                                metric["mse_t"] += F.mse_loss(recon_t, t_global, reduction="sum").item()
                                metric["sum_t"] += t_global.sum().item()
                                metric["sum_sq_t"] += (t_global ** 2).sum().item()
                                metric["count_t"] += t_global.numel()

                                diff = latent_t - latent_v
                                penalty = F.relu(diff).sum(dim=-1)
                                denom = latent_t.abs().sum(dim=-1).clamp(min=1e-6)
                                metric["entailment_ratio_sum"] += (penalty / denom).sum().item()
                                metric["entailment_ratio_count"] += v_pad.size(0)

                                score_matrix = _cosine_score_matrix(latent_v, latent_t)
                                _update_retrieval(metric, score_matrix)
                                metric["cosine_sim"] += F.cosine_similarity(latent_t, latent_v, dim=-1).sum().item()
                                metric["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=0))
                                metric["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))

                            elif method == "asym" and not Config.asym_use_views:
                                v_proj, t_proj = aux(v_pad, t_pad)

                                v_proj_flat = v_proj[v_mask]
                                t_proj_flat = t_proj[t_mask]
                                recon_v, recon_t, latent_v, latent_t = model(
                                    vision_embeddings=v_proj_flat, text_embeddings=t_proj_flat
                                )

                                metric["mse_v"] += F.mse_loss(recon_v, v_proj_flat, reduction="sum").item()
                                metric["sum_v"] += v_proj_flat.sum().item()
                                metric["sum_sq_v"] += (v_proj_flat ** 2).sum().item()
                                metric["count_v"] += v_proj_flat.numel()

                                metric["mse_t"] += F.mse_loss(recon_t, t_proj_flat, reduction="sum").item()
                                metric["sum_t"] += t_proj_flat.sum().item()
                                metric["sum_sq_t"] += (t_proj_flat ** 2).sum().item()
                                metric["count_t"] += t_proj_flat.numel()

                                v_idx, t_idx = 0, 0
                                v_union_list = []
                                t_union_list = []
                                penalty_ratio = 0.0

                                for b in range(v_mask.size(0)):
                                    lv = int(v_mask[b].sum().item())
                                    lt = int(t_mask[b].sum().item())
                                    if lv > 0:
                                        v_union = latent_v[v_idx : v_idx + lv].max(dim=0)[0]
                                    else:
                                        v_union = torch.zeros(latent_v.size(-1), device=latent_v.device)
                                    if lt > 0:
                                        t_union = latent_t[t_idx : t_idx + lt].max(dim=0)[0]
                                        diff = latent_t[t_idx : t_idx + lt].detach() - v_union.unsqueeze(0)
                                        penalty = F.relu(diff).sum(dim=-1)
                                        denom = latent_t[t_idx : t_idx + lt].abs().sum(dim=-1).clamp(min=1e-6)
                                        penalty_ratio += (penalty / denom).mean().item()
                                    else:
                                        t_union = torch.zeros(latent_t.size(-1), device=latent_t.device)

                                    v_union_list.append(v_union)
                                    t_union_list.append(t_union)
                                    v_idx += lv
                                    t_idx += lt

                                if v_union_list:
                                    v_union_mat = torch.stack(v_union_list, dim=0)
                                    t_union_mat = torch.stack(t_union_list, dim=0)
                                    score_matrix = _cosine_score_matrix(v_union_mat, t_union_mat)
                                    _update_retrieval(metric, score_matrix)
                                    metric["cosine_sim"] += F.cosine_similarity(
                                        t_union_mat, v_union_mat, dim=-1
                                    ).sum().item()

                                metric["entailment_ratio_sum"] += penalty_ratio
                                metric["entailment_ratio_count"] += v_pad.size(0)
                                metric["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=0))
                                metric["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))

                            elif method == "asym":
                                v_proj, t_proj = aux(v_pad, t_pad)
                                pool_mode = getattr(Config, "asym_text_pool", "mean")
                                text_temp = getattr(Config, "asym_text_temp", 1.0)
                                text_topk = getattr(Config, "asym_text_topk", 0)
                                t_global = _pool_text_tokens(t_proj, t_mask, pool_mode, text_temp, text_topk)

                                sampler = samplers[method][model_name]
                                v_views = sampler(v_proj, v_len, grid_thws)
                                recon_v, recon_t, latent_v, latent_t = model(
                                    vision_embeddings=v_views, text_embeddings=t_global
                                )

                                metric["mse_v"] += F.mse_loss(recon_v, v_views, reduction="sum").item()
                                metric["sum_v"] += v_views.sum().item()
                                metric["sum_sq_v"] += (v_views ** 2).sum().item()
                                metric["count_v"] += v_views.numel()

                                metric["mse_t"] += F.mse_loss(recon_t, t_global, reduction="sum").item()
                                metric["sum_t"] += t_global.sum().item()
                                metric["sum_sq_t"] += (t_global ** 2).sum().item()
                                metric["count_t"] += t_global.numel()

                                sim = F.cosine_similarity(latent_t.unsqueeze(1), latent_v, dim=-1)
                                metric["cosine_sim"] += sim.max(dim=1)[0].sum().item()

                                union_temp = getattr(Config, "asym_union_temp", 0.0)
                                if union_temp and union_temp > 0:
                                    v_union = torch.logsumexp(latent_v / union_temp, dim=1) * union_temp
                                    v_union = v_union - union_temp * math.log(latent_v.size(1))
                                else:
                                    v_union = latent_v.max(dim=1)[0]
                                diff = latent_t - v_union
                                penalty = F.relu(diff).sum(dim=-1)
                                denom = latent_t.abs().sum(dim=-1).clamp(min=1e-6)
                                metric["entailment_ratio_sum"] += (penalty / denom).sum().item()
                                metric["entailment_ratio_count"] += v_pad.size(0)

                                score_matrix = _asym_score_matrix(latent_v, latent_t)
                                _update_retrieval(metric, score_matrix)
                                metric["active_latents_v"].logical_or_((latent_v > 1e-5).any(dim=1).any(dim=0))
                                metric["active_latents_t"].logical_or_((latent_t > 1e-5).any(dim=0))
                            else:
                                raise ValueError(f"Unsupported method: {method}")

        os.remove(pt_path)

    rows = []
    for (method, cond_type, level, model_name), metric in metrics.items():
        summary = _finalize_metric(metric)
        summary.update({
            "method": method,
            "condition": cond_type,
            "level": float(level),
            "model": model_name,
            "primary": 0.6 * summary["align_score"] + 0.4 * summary["coverage"],
        })
        rows.append(summary)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    import csv

    cols = [
        "method",
        "condition",
        "level",
        "model",
        "ev_v",
        "ev_t",
        "dead_v",
        "dead_t",
        "align_avgcos",
        "align_possim",
        "r1_i2t",
        "r1_t2i",
        "r5_i2t",
        "r5_t2i",
        "entail_ratio",
        "coverage",
        "align_score",
        "primary",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved robustness metrics to: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", default="/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_test_short.json")
    parser.add_argument("--save-dir", default="/home/liuzonghao/AASAE/qwen/block_trainer/checkpoints_sae")
    parser.add_argument("--methods", default="sym,asym")
    parser.add_argument("--models", default="SAE_V,SAE_D,VL_SAE")
    parser.add_argument("--noise-stds", default="0,0.02,0.05,0.1")
    parser.add_argument("--dropout-rates", default="0,0.2,0.5")
    parser.add_argument("--budget-fracs", default="1.0,0.5,0.25")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="/home/liuzonghao/AASAE/qwen/eval/robustness_metrics.csv")
    parser.add_argument(
        "--asym_use_views",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether ASYM uses view sampling (1) or token-level only (0).",
    )
    args = parser.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    # methods = [m.upper() for m in methods]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    Config.asym_use_views = bool(args.asym_use_views)

    conditions: List[Tuple[str, float]] = []
    for std in _parse_list(args.noise_stds, float):
        conditions.append(("noise", float(std)))
    for dr in _parse_list(args.dropout_rates, float):
        conditions.append(("dropout", float(dr)))
    for frac in _parse_list(args.budget_fracs, float):
        conditions.append(("budget", float(frac)))

    evaluate(
        test_json=args.test_json,
        save_dir=args.save_dir,
        methods=methods,
        model_names=models,
        conditions=conditions,
        out_csv=args.out_csv,
        chunk_size=args.chunk_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
