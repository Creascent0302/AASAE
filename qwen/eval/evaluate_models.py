"""
evaluator.py — SAE evaluation pipeline with image-level dead latent metrics.

Core changes vs. original
──────────────────────────
1. Dead-latent metric is now **image-level** (not flat-token-level).

   Old (broken): active if ANY token in the entire flat batch activates the
   feature.  With FILIP's ~6400 tokens/batch every feature looks active → 0%.

   New: for each image b, check whether ANY of its own tokens activates
   feature k; aggregate over all images.  This is meaningful and comparable
   between SYM and FILIP because the counting unit is always "one image".

2. Three dead-latent thresholds are reported for rigour:
     dead_strict : never active for ANY image in the eval set
     dead_1pct   : active for < 1 % of images
     dead_5pct   : active for < 5 % of images

3. View-sampling code and 'asym' evaluation branch removed.
4. Evaluation modes: 'sym' and 'filip' (same logic regardless of lambda_align).
"""

import os
import sys
import json
import datetime
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_trainer.config import Config
from block_trainer.extractor import FeatureExtractor
from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE, TokenAuxProj
from block_trainer.trainer import PairDataset, collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SAEEvaluator:
    """
    Evaluates three SAE architectures (SAE_V, SAE_D, VL_SAE) for a
    given training method ('sym' or 'filip') on cross-modal retrieval,
    reconstruction quality, feature coverage, and image-level sparsity.
    """

    def __init__(
        self,
        method_name: str,
        eval_batch_size: int = 8,
        report_file: str = "evaluation_results_all.txt",
        score_align_weight: float = 0.6,
        score_entail_weight: float = 0.4,
    ) -> None:
        self.method             = method_name
        self.eval_batch_size    = int(eval_batch_size)
        self.report_file        = report_file
        self.score_align_w      = float(score_align_weight)
        self.score_entail_w     = float(score_entail_weight)

        Config.train_method = self.method
        print(f"\n[Evaluator] Method: {self.method.upper()} | TopK: {Config.topk}")

        def _dev(i):
            if torch.cuda.is_available():
                c = torch.cuda.device_count()
                return torch.device(f"cuda:{min(i, c - 1)}")
            return torch.device("cpu")

        self.device_map = {"SAE_V": _dev(1), "SAE_D": _dev(2), "VL_SAE": _dev(3)}

        # ── Load SAE models ─────────────────────────────────────────────────
        self.models = {
            "SAE_V":  SAE_V( Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk)
                           .to(self.device_map["SAE_V"]),
            "SAE_D":  SAE_D( Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk)
                           .to(self.device_map["SAE_D"]),
            "VL_SAE": VL_SAE(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk)
                           .to(self.device_map["VL_SAE"]),
        }
        for name in self.models:
            p = os.path.join(Config.save_dir, f"{name}_{self.method}_new_best_sae.pth")
            if not os.path.exists(p):
                p = os.path.join(Config.save_dir, f"{name}_{self.method}_best_sae.pth")
            if not os.path.exists(p):
                print(f"  [ERROR] {p} not found"); sys.exit(1)
            ckpt = torch.load(p, map_location=self.device_map[name], weights_only=False)
            self.models[name].load_state_dict(
                ckpt.get("sae_state_dict", ckpt))
            self.models[name].eval()
            print(f"  Loaded {name} ← {p}")

        # ── Load shared AuxProj ─────────────────────────────────────────────
        self.aux_projs = {}
        aux_p = os.path.join(Config.save_dir, f"shared_best_aux_proj_{self.method}.pth")
        if not os.path.exists(aux_p):
            print(f"  [ERROR] {aux_p} not found"); sys.exit(1)
        sd = torch.load(aux_p, map_location="cpu", weights_only=True)
        for name, dev in self.device_map.items():
            aux = TokenAuxProj(Config.qwen_hidden_dim).to(dev)
            aux.load_state_dict(sd);  aux.eval()
            self.aux_projs[name] = aux
        print(f"  Loaded AuxProj ← {aux_p}")

        # ── Metrics dict ────────────────────────────────────────────────────
        # image_active_sum_{v,t}: float tensor [D_sae] counting how many
        # eval images activated each feature.  Used to compute all three
        # dead-latent thresholds in the final report.
        self.metrics = {
            name: {
                # Reconstruction (token-level for FILIP, global for SYM)
                "mse_v": 0.0,   "mse_t": 0.0,
                "sum_v": 0.0,   "sum_sq_v": 0.0,  "count_v": 0,
                "sum_t": 0.0,   "sum_sq_t": 0.0,  "count_t": 0,
                # Cross-modal entailment
                "entail_sum": 0.0,  "entail_count": 0,
                # Retrieval
                "i2t_r1": 0, "i2t_r5": 0,
                "t2i_r1": 0, "t2i_r5": 0,
                "align_total": 0,
                "pos_sim_sum": 0.0, "pos_sim_count": 0,
                "cosine_sum": 0.0,
                # Image-level dead-latent tracking
                "image_active_sum_v": torch.zeros(
                    Config.sae_hidden_dim, dtype=torch.float,
                    device=self.device_map[name]),
                "image_active_sum_t": torch.zeros(
                    Config.sae_hidden_dim, dtype=torch.float,
                    device=self.device_map[name]),
                "image_total": 0,
                "samples": 0,
            }
            for name in self.models
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_retrieval(self, m: dict, score_matrix: torch.Tensor) -> None:
        B = score_matrix.size(0)
        if B == 0:
            return
        labels = torch.arange(B, device=score_matrix.device)
        k5     = min(5, B)
        m["i2t_r1"]        += (score_matrix.argmax(1)  == labels).sum().item()
        m["t2i_r1"]        += (score_matrix.argmax(0)  == labels).sum().item()
        m["i2t_r5"]        += (score_matrix.topk(k5, 1).indices == labels.unsqueeze(1)).any(1).sum().item()
        m["t2i_r5"]        += (score_matrix.topk(k5, 0).indices == labels.unsqueeze(0)).any(0).sum().item()
        m["align_total"]   += B
        m["pos_sim_sum"]   += score_matrix.diag().sum().item()
        m["pos_sim_count"] += B

    def _cosine_matrix(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(v, dim=-1) @ F.normalize(t, dim=-1).T

    def _filip_matrix(
        self,
        v_list: List[torch.Tensor],
        t_list: List[torch.Tensor],
    ) -> torch.Tensor:
        B = len(v_list)
        if B == 0:
            return torch.zeros(0, 0)
        scores = torch.zeros(B, B, device=v_list[0].device)
        for i in range(B):
            vi = F.normalize(v_list[i], dim=-1)
            for j in range(B):
                tj = F.normalize(t_list[j], dim=-1)
                sim = vi @ tj.T
                scores[i, j] = (sim.max(1).values.mean() + sim.max(0).values.mean()) / 2.0
        return scores

    # ─────────────────────────────────────────────────────────────────────────
    # Main evaluation chunk
    # ─────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def evaluate_chunk(self, chunk_path: str) -> None:
        data    = torch.load(chunk_path, map_location="cpu", weights_only=False)
        bs      = min(self.eval_batch_size, 16) or min(Config.batch_size, 16)
        loader  = DataLoader(PairDataset(data), batch_size=bs, collate_fn=collate_fn)

        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _ in loader:
            B = v_pad_cpu.size(0)

            for name, model in self.models.items():
                dev = self.device_map[name]
                m   = self.metrics[name]
                m["samples"] += B

                v_pad  = v_pad_cpu.to(dev,  non_blocking=True)
                t_pad  = t_pad_cpu.to(dev,  non_blocking=True)
                v_mask = v_mask_cpu.to(dev, non_blocking=True)
                t_mask = t_mask_cpu.to(dev, non_blocking=True)

                if self.method == "sym":
                    self._eval_sym(name, model, m, v_pad, t_pad, v_mask, t_mask, B)
                else:  # 'filip'
                    self._eval_filip(name, model, m, v_pad, t_pad, v_mask, t_mask, B)

    # ─── SYM evaluation ──────────────────────────────────────────────────────

    def _eval_sym(self, name, model, m, v_pad, t_pad, v_mask, t_mask, B):
        v_g = (v_pad * v_mask.unsqueeze(-1)).sum(1) / (v_mask.sum(1, keepdim=True) + 1e-6)
        t_g = (t_pad * t_mask.unsqueeze(-1)).sum(1) / (t_mask.sum(1, keepdim=True) + 1e-6)
        v_g, t_g    = self.aux_projs[name](v_g, t_g)
        _, _, lv, lt = model(vision_embeddings=v_g, text_embeddings=t_g)

        # Reconstruction EV
        recon_v, recon_t, _, _ = model(vision_embeddings=v_g, text_embeddings=t_g)
        m["mse_v"]    += F.mse_loss(recon_v, v_g, reduction="sum").item()
        m["sum_v"]    += v_g.sum().item();  m["sum_sq_v"] += (v_g ** 2).sum().item()
        m["count_v"]  += v_g.numel()
        m["mse_t"]    += F.mse_loss(recon_t, t_g, reduction="sum").item()
        m["sum_t"]    += t_g.sum().item();  m["sum_sq_t"] += (t_g ** 2).sum().item()
        m["count_t"]  += t_g.numel()

        # Entailment
        diff   = lv - lt
        pen    = F.relu(diff).sum(-1)
        denom  = lt.abs().sum(-1).clamp(min=1e-6)
        m["entail_sum"]   += (pen / denom).sum().item()
        m["entail_count"] += B

        # Retrieval
        self._update_retrieval(m, self._cosine_matrix(lv, lt))
        m["cosine_sum"] += F.cosine_similarity(lt, lv, dim=-1).sum().item()

        # Image-level dead-latent tracking (one global vector = one image)
        m["image_active_sum_v"] += (lv > 1e-5).float().sum(0)   # [D_sae] count of images
        m["image_active_sum_t"] += (lt > 1e-5).float().sum(0)
        m["image_total"]        += B

    # ─── FILIP evaluation ─────────────────────────────────────────────────────

    def _eval_filip(self, name, model, m, v_pad, t_pad, v_mask, t_mask, B):
        v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)

        # Flatten all tokens for joint SAE forward (reconstruction EV stays token-level)
        v_flat = v_proj[v_mask]
        t_flat = t_proj[t_mask]
        recon_v, recon_t, lv_flat, lt_flat = model(
            vision_embeddings=v_flat, text_embeddings=t_flat)

        m["mse_v"]   += F.mse_loss(recon_v, v_flat, reduction="sum").item()
        m["sum_v"]   += v_flat.sum().item();  m["sum_sq_v"] += (v_flat ** 2).sum().item()
        m["count_v"] += v_flat.numel()
        m["mse_t"]   += F.mse_loss(recon_t, t_flat, reduction="sum").item()
        m["sum_t"]   += t_flat.sum().item();  m["sum_sq_t"] += (t_flat ** 2).sum().item()
        m["count_t"] += t_flat.numel()

        # Per-image analysis: retrieval, entailment, dead-latent
        v_latents_list: List[torch.Tensor] = []
        t_latents_list: List[torch.Tensor] = []
        cosine_sum = entail_sum = 0.0
        v_idx = t_idx = 0

        for b in range(B):
            lv = int(v_mask[b].sum().item())
            lt = int(t_mask[b].sum().item())

            lv_lat = lv_flat[v_idx: v_idx + lv]
            lt_lat = lt_flat[t_idx: t_idx + lt]

            if lv_lat.numel() == 0:
                lv_lat = lv_flat.new_zeros(1, lv_flat.size(-1))
            if lt_lat.numel() == 0:
                lt_lat = lt_flat.new_zeros(1, lt_flat.size(-1))

            # Per-image cosine similarity (bidirectional FILIP)
            sim  = F.cosine_similarity(lv_lat.unsqueeze(1), lt_lat.unsqueeze(0), dim=-1)
            cosine_sum += (sim.max(1).values.mean() + sim.max(0).values.mean()).item() / 2.0

            # Per-image asymmetric entailment (text coverage by visual tokens)
            v_union = lv_lat.max(dim=0).values
            diff    = lt_lat - v_union.unsqueeze(0)
            pen     = F.relu(diff).sum()
            denom   = lt_lat.abs().sum().clamp(min=1e-6)
            entail_sum += (pen / denom).item()

            v_latents_list.append(lv_lat)
            t_latents_list.append(lt_lat)

            # ── Image-level dead-latent tracking ───────────────────────────
            # A feature is "active for this image" if it fires for at least
            # one of the image's tokens.  Using (any, dim=0) gives a [D_sae]
            # bool mask that is comparable to the SYM global-vector flag.
            if lv > 0:
                m["image_active_sum_v"] += (lv_lat > 1e-5).any(dim=0).float()
            if lt > 0:
                m["image_active_sum_t"] += (lt_lat > 1e-5).any(dim=0).float()
            m["image_total"] += 1

            v_idx += lv
            t_idx += lt

        m["cosine_sum"]   += cosine_sum
        m["entail_sum"]   += entail_sum
        m["entail_count"] += B

        self._update_retrieval(m, self._filip_matrix(v_latents_list, t_latents_list))

    # ─────────────────────────────────────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────────────────────────────────────

    def print_final_report(self) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def lp(msg, f):
            print(msg);  f.write(msg + "\n")

        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*65}\n")
            lp(f" Test Time : {now}", f)
            lp(f" Method    : {self.method.upper()} | TopK: {Config.topk}", f)
            lp(f" λ_align   : {Config.lambda_align}", f)
            f.write(f"{'='*65}\n")

            for name in self.models:
                m = self.metrics[name]
                if m["samples"] == 0:
                    continue

                # ── Reconstruction EV ────────────────────────────────────
                def ev(mse, s, sq, cnt):
                    mu  = s / max(cnt, 1)
                    var = (sq / max(cnt, 1)) - mu ** 2
                    return 1.0 - mse / (var * cnt + 1e-9)

                ev_v = ev(m["mse_v"], m["sum_v"], m["sum_sq_v"], m["count_v"])
                ev_t = ev(m["mse_t"], m["sum_t"], m["sum_sq_t"], m["count_t"])

                # ── Retrieval ────────────────────────────────────────────
                tot   = max(m["align_total"], 1)
                i2t_r1 = m["i2t_r1"] / tot;  i2t_r5 = m["i2t_r5"] / tot
                t2i_r1 = m["t2i_r1"] / tot;  t2i_r5 = m["t2i_r5"] / tot
                pos_sim = m["pos_sim_sum"] / max(m["pos_sim_count"], 1)
                avg_cos = m["cosine_sum"]  / max(m["samples"], 1)

                align_r1    = 0.5 * (i2t_r1 + t2i_r1)
                align_r5    = 0.5 * (i2t_r5 + t2i_r5)
                align_score = 0.5 * (align_r1 + align_r5)

                # ── Entailment / Coverage ────────────────────────────────
                entail_ratio  = m["entail_sum"] / max(m["entail_count"], 1)
                coverage      = 1.0 - entail_ratio
                primary       = self.score_align_w * align_score + self.score_entail_w * coverage

                # ── Image-level dead-latent metrics ──────────────────────
                # Act_freq[k] = fraction of eval images that activate feature k.
                # Three thresholds for rigour (strict / 1% / 5%).
                n_img        = max(m["image_total"], 1)
                act_freq_v   = m["image_active_sum_v"] / n_img
                act_freq_t   = m["image_active_sum_t"] / n_img

                dead_v_strict = (m["image_active_sum_v"] == 0).float().mean().item()
                dead_t_strict = (m["image_active_sum_t"] == 0).float().mean().item()
                dead_v_1pct   = (act_freq_v < 0.01).float().mean().item()
                dead_t_1pct   = (act_freq_t < 0.01).float().mean().item()
                dead_v_5pct   = (act_freq_v < 0.05).float().mean().item()
                dead_t_5pct   = (act_freq_t < 0.05).float().mean().item()
                # Median activation frequency: "typical" feature activates for X% of images
                med_freq_v    = act_freq_v.median().item()
                med_freq_t    = act_freq_t.median().item()

                lp(f"[ {name} ]", f)
                lp(f"  EV (V/T)              : {ev_v:.4f} / {ev_t:.4f}", f)
                lp(f"  ──── Image-level dead latents ────", f)
                lp(f"  Dead strict  (V/T)    : {dead_v_strict:.2%} / {dead_t_strict:.2%}  "
                   f"(never active in eval set)", f)
                lp(f"  Dead <1% img (V/T)    : {dead_v_1pct:.2%} / {dead_t_1pct:.2%}  "
                   f"(active for <1% of images)", f)
                lp(f"  Dead <5% img (V/T)    : {dead_v_5pct:.2%} / {dead_t_5pct:.2%}  "
                   f"(active for <5% of images)", f)
                lp(f"  Median act-freq (V/T) : {med_freq_v:.3f} / {med_freq_t:.3f}  "
                   f"(typical feature activates for X% of images)", f)
                lp(f"  ──── Retrieval ────", f)
                lp(f"  AvgCos / PosSim       : {avg_cos:.4f} / {pos_sim:.4f}", f)
                lp(f"  R@1 I2T / T2I         : {i2t_r1:.4f} / {t2i_r1:.4f}", f)
                lp(f"  R@5 I2T / T2I         : {i2t_r5:.4f} / {t2i_r5:.4f}", f)
                lp(f"  ──── Coverage / Primary ────", f)
                lp(f"  Entail Ratio          : {entail_ratio:.4f}", f)
                lp(f"  Coverage              : {coverage:.4f}", f)
                lp(
                    f"  Primary Score         : {primary:.4f}  "
                    f"(Align×{self.score_align_w:.2f} + Coverage×{self.score_entail_w:.2f})",
                    f,
                )
                f.write("-" * 55 + "\n")

        print(f"\n[Report] appended → {os.path.abspath(self.report_file)}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_pipeline(
    json_path: str,
    method_name: str,
    chunk_size: int = 100,
    eval_batch_size: int = 8,
    max_items: int = 0,
    report_file: str = "evaluation_results_all.txt",
    score_align_weight: float = 0.6,
    score_entail_weight: float = 0.4,
) -> None:
    Config.train_method = method_name
    extractor = FeatureExtractor()
    evaluator = SAEEvaluator(
        method_name        = method_name,
        eval_batch_size    = eval_batch_size,
        report_file        = report_file,
        score_align_weight = score_align_weight,
        score_entail_weight= score_entail_weight,
    )

    with open(json_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    if max_items > 0:
        dataset = dataset[:max_items]
    print(f"[*] Eval images: {len(dataset)}")

    for i in range(0, len(dataset), chunk_size):
        chunk_data = dataset[i: i + chunk_size]
        pt_path    = extractor.extract_and_save_chunk(chunk_data, i // chunk_size)
        if pt_path and os.path.exists(pt_path):
            evaluator.evaluate_chunk(pt_path)
            os.remove(pt_path)

    evaluator.print_final_report()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAE Evaluation Pipeline")
    parser.add_argument("--test-json", default="/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_test_short.json")
    parser.add_argument("--model-path",      default=Config.model_path)
    parser.add_argument("--image-folder",    default=Config.image_folder)
    parser.add_argument("--save-dir",        default=Config.save_dir)
    parser.add_argument("--target-layer",    default=Config.target_layer_name)
    parser.add_argument("--chunk-size",      type=int,   default=100)
    parser.add_argument("--eval-batch-size", type=int,   default=8)
    parser.add_argument("--max-items",       type=int,   default=0)
    parser.add_argument("--methods",         default="sym,filip",
                        help="Comma-separated list of methods to evaluate.")
    parser.add_argument("--topk",            type=int,   default=64,
                        help="TopK sparsity (same for all methods – fair comparison).")
    parser.add_argument("--lambda-align",    type=float, default=0.0,
                        help="Entailment weight used during training (informational only).")
    parser.add_argument("--report-file",     default="evaluation_results_all.txt")
    parser.add_argument("--score-align",     type=float, default=0.6)
    parser.add_argument("--score-entail",    type=float, default=0.4)
    args = parser.parse_args()

    Config.model_path        = args.model_path
    Config.image_folder      = args.image_folder
    Config.save_dir          = args.save_dir
    Config.target_layer_name = args.target_layer
    Config.topk              = args.topk
    Config.lambda_align      = args.lambda_align

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        run_evaluation_pipeline(
            json_path           = args.test_json,
            method_name         = method,
            chunk_size          = args.chunk_size,
            eval_batch_size     = args.eval_batch_size,
            max_items           = args.max_items,
            report_file         = args.report_file,
            score_align_weight  = args.score_align,
            score_entail_weight = args.score_entail,
        )