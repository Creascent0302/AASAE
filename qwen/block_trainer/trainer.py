"""
trainer.py — Phase-1 (AuxProj alignment) and Phase-2 (SAE dictionary) training.

Core changes vs. original
──────────────────────────
1. DynamicViewSampler and all view-sampling code removed.
2. Training methods simplified to two modes:
     • 'sym'  – global-average-pool contrastive baseline
     • 'filip' – token-level FILIP alignment (Phase 1) +
                 per-image SAE training with optional asymmetric
                 entailment constraint (Phase 2)
3. Per-image SAE training loop (FILIP mode):
     – Each image's tokens are processed independently instead of
       being flattened into a giant batch.
     – Dead-latent update is called ONCE per outer batch (not per token),
       giving semantically meaningful, SYM-comparable dead-latent counts.
4. Asymmetric entailment (Config.lambda_align > 0):
     – For each image: v_union = max(SAE(v_tokens), dim=0)
       penalty = ReLU(t_latents − v_union) / ||t_latents||₁
     – Enforces "text concepts must be covered by visual features" per image.
"""

import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    from config import Config
    from sae_model import SAE_V, SAE_D, VL_SAE, CAFECore, TokenAuxProj
except ImportError:
    from block_trainer.config import Config
    from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE, CAFECore, TokenAuxProj


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        grid = self.data[i].get("grid_thw", None)
        return self.data[i]["vision"].float(), self.data[i]["text"].float(), grid


def collate_fn(batch):
    v_list    = [b[0] for b in batch]
    t_list    = [b[1] for b in batch]
    grid_thws = [b[2] for b in batch]
    v_len     = torch.tensor([len(v) for v in v_list])
    t_len     = torch.tensor([len(t) for t in t_list])
    v_pad     = pad_sequence(v_list, batch_first=True)
    t_pad     = pad_sequence(t_list, batch_first=True)
    v_mask    = torch.arange(v_pad.size(1))[None, :] < v_len[:, None]
    t_mask    = torch.arange(t_pad.size(1))[None, :] < t_len[:, None]
    return v_pad, t_pad, v_mask, t_mask, grid_thws, v_len


# ─────────────────────────────────────────────────────────────────────────────
# Alignment loss functions (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def global_contrastive_loss(
    v_global: torch.Tensor,
    t_global: torch.Tensor,
    temp: float = 0.07,
) -> torch.Tensor:
    """Standard symmetric InfoNCE on global vectors (SYM baseline)."""
    v = F.normalize(v_global, dim=-1)
    t = F.normalize(t_global, dim=-1)
    sim = torch.matmul(v, t.T) / temp
    labels = torch.arange(v.shape[0], device=v.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def batch_filip_loss(
    v_proj: torch.Tensor,
    t_proj: torch.Tensor,
    v_mask: torch.Tensor,
    t_mask: torch.Tensor,
    temp: float = 0.07,
) -> torch.Tensor:
    """
    Memory-efficient token-level FILIP contrastive loss.

    For each image b, compute its bidirectional maximum-similarity score
    against all texts in the batch, avoiding materialising the full
    [B × Lv × B × Lt] tensor.
    """
    B, Lv, D = v_proj.shape
    v_norm = F.normalize(v_proj, dim=-1)
    t_norm = F.normalize(t_proj, dim=-1)
    align_score = torch.zeros(B, B, device=v_proj.device)

    for b in range(B):
        # sim_b: [Lv, B, Lt]
        sim_b = torch.einsum("id,cjd->icj", v_norm[b], t_norm) / temp

        # Vision → text direction
        sim_v = sim_b.clone()
        sim_v.masked_fill_(~t_mask.unsqueeze(0), -1e4)
        max_sim_v = sim_v.max(dim=2).values            # [Lv, B]
        valid_v   = v_mask[b].sum()
        align_v   = (
            (max_sim_v * v_mask[b].unsqueeze(1)).sum(0) / valid_v
            if valid_v > 0
            else torch.zeros(B, device=v_proj.device)
        )

        # Text → vision direction
        sim_t = sim_b.clone()
        sim_t.masked_fill_(~v_mask[b].view(Lv, 1, 1), -1e4)
        max_sim_t = sim_t.max(dim=0).values            # [B, Lt]
        align_t   = (max_sim_t * t_mask).sum(1) / t_mask.sum(1).clamp(min=1)

        align_score[b] = (align_v + align_t) / 2.0

    labels = torch.arange(B, device=v_proj.device)
    return 0.5 * (F.cross_entropy(align_score, labels) + F.cross_entropy(align_score.T, labels))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: AuxProj pre-alignment trainer
# ─────────────────────────────────────────────────────────────────────────────

class AuxProjTrainer:
    """
    Train the shared TokenAuxProj projection on the contrastive objective.
    The best checkpoint is frozen and reused in Phase 2.
    """

    def __init__(self) -> None:
        print(
            f"\n[Phase 1: AuxProj] Initialising "
            f"({Config.train_method.upper()}) token alignment training …"
        )
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.shared_aux     = TokenAuxProj(Config.qwen_hidden_dim).to(self.device)
        self.optimizer      = optim.Adam(
            self.shared_aux.parameters(),
            lr=Config.initial_lr,
            weight_decay=Config.weight_decay,
        )
        self.scaler         = GradScaler("cuda")
        self.best_val_loss  = float("inf")

    # ── Internal ────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        v_proj: torch.Tensor,
        t_proj: torch.Tensor,
        v_mask: torch.Tensor,
        t_mask: torch.Tensor,
    ) -> torch.Tensor:
        if Config.train_method == "sym":
            v_g = (v_proj * v_mask.unsqueeze(-1)).sum(1) / (v_mask.sum(1, keepdim=True) + 1e-6)
            t_g = (t_proj * t_mask.unsqueeze(-1)).sum(1) / (t_mask.sum(1, keepdim=True) + 1e-6)
            return global_contrastive_loss(v_g, t_g)
        # 'filip': token-level bidirectional FILIP
        return batch_filip_loss(v_proj, t_proj, v_mask, t_mask)

    # ── Public API ──────────────────────────────────────────────────────────

    def train_on_chunk(
        self,
        train_chunk_path: str,
        val_chunk_path: str,
        chunk_idx: int,
    ) -> None:
        print(f"\n[Phase 1] AuxProj – chunk {chunk_idx}")
        train_data = torch.load(train_chunk_path, map_location="cpu", weights_only=False)
        val_data   = torch.load(val_chunk_path,   map_location="cpu", weights_only=False)
        train_loader = DataLoader(PairDataset(train_data), batch_size=Config.batch_size,
                                  shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(PairDataset(val_data),   batch_size=Config.batch_size,
                                  shuffle=False, collate_fn=collate_fn)

        # ── Training ────────────────────────────────────────────────────────
        self.shared_aux.train()
        pbar = tqdm(train_loader, desc=f"Phase 1: train (chunk {chunk_idx})")
        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _ in pbar:
            try:
                v_pad  = v_pad_cpu.to(self.device,  non_blocking=True)
                t_pad  = t_pad_cpu.to(self.device,  non_blocking=True)
                v_mask = v_mask_cpu.to(self.device, non_blocking=True)
                t_mask = t_mask_cpu.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                with autocast("cuda"):
                    v_proj, t_proj = self.shared_aux(v_pad, t_pad)
                    loss = self._compute_loss(v_proj, t_proj, v_mask, t_mask)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                del loss, v_proj, t_proj, v_pad, t_pad
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()

        # ── Validation ──────────────────────────────────────────────────────
        self.shared_aux.eval()
        val_total = 0.0
        with torch.no_grad():
            for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _ in tqdm(
                val_loader, desc="Phase 1: val", leave=False
            ):
                try:
                    v_pad  = v_pad_cpu.to(self.device,  non_blocking=True)
                    t_pad  = t_pad_cpu.to(self.device,  non_blocking=True)
                    v_mask = v_mask_cpu.to(self.device, non_blocking=True)
                    t_mask = t_mask_cpu.to(self.device, non_blocking=True)
                    with autocast("cuda"):
                        v_proj, t_proj = self.shared_aux(v_pad, t_pad)
                        val_total += self._compute_loss(v_proj, t_proj, v_mask, t_mask).item()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

        avg = val_total / max(len(val_loader), 1)
        print(f"  Phase 1 val loss: {avg:.4f} (best {self.best_val_loss:.4f})")
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            os.makedirs(Config.save_dir, exist_ok=True)
            path = os.path.join(Config.save_dir, f"shared_best_aux_proj_{Config.train_method}.pth")
            torch.save(self.shared_aux.state_dict(), path)
            print(f"  ✅ Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: SAE dictionary trainer
# ─────────────────────────────────────────────────────────────────────────────

class SAETrainer:
    """
    Train SAE dictionaries on top of frozen AuxProj features.

    Two training modes
    ──────────────────
    'sym'   : global-pool → SAE reconstruction only (baseline)
    'filip' : per-image token-level loop → SAE reconstruction +
              optional asymmetric entailment (Config.lambda_align > 0)
    """

    def __init__(self) -> None:
        print(
            f"\n[Phase 2: SAE] Initialising "
            f"({Config.train_method.upper()}) SAE dictionary training …"
        )
        self.device_map = {
            "SAE_V":  torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"),
            "SAE_D":  torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0"),
            "VL_SAE": torch.device("cuda:3" if torch.cuda.device_count() > 3 else "cuda:0"),
        }

        sae_cfg = {
            "input_unit_norm":      Config.input_unit_norm,
            "l1_coeff":             Config.l1_coeff,
            "aux_penalty":          Config.aux_penalty,
            "top_k_aux":            Config.top_k_aux,
            "n_batches_to_dead":    Config.n_batches_to_dead,
            "use_threshold_in_eval": Config.use_threshold_in_eval,
        }
        self.models = {
            "SAE_V":  SAE_V( Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk, cfg=sae_cfg)
                          .to(self.device_map["SAE_V"]),
            "SAE_D":  SAE_D( Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk, cfg=sae_cfg)
                          .to(self.device_map["SAE_D"]),
            "VL_SAE": VL_SAE(Config.qwen_hidden_dim, Config.sae_hidden_dim, Config.topk, cfg=sae_cfg)
                          .to(self.device_map["VL_SAE"]),
        }

        # ── Load frozen AuxProj ─────────────────────────────────────────────
        self.aux_projs = {}
        load_path = os.path.join(Config.save_dir, f"shared_best_aux_proj_{Config.train_method}.pth")
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"[Phase 2] Shared AuxProj not found at {load_path}. "
                "Run Phase 1 first."
            )
        shared_sd = torch.load(load_path, map_location="cpu", weights_only=True)
        for name, device in self.device_map.items():
            aux = TokenAuxProj(Config.qwen_hidden_dim).to(device)
            aux.load_state_dict(shared_sd)
            aux.eval()
            for p in aux.parameters():
                p.requires_grad_(False)
            self.aux_projs[name] = aux
        print(f"  Loaded AuxProj: {load_path}")

        self.optimizers = {
            n: optim.Adam(m.parameters(), lr=Config.initial_lr, weight_decay=Config.weight_decay)
            for n, m in self.models.items()
        }
        self.scalers          = {n: GradScaler("cuda") for n in self.models}
        self.best_val_loss    = {n: float("inf")        for n in self.models}
        self._b_dec_init_done = False

    # ── Utility: get per-modality SAE cores ─────────────────────────────────

    def _get_sae_cores(self, name: str) -> Tuple[CAFECore, CAFECore]:
        """
        Return (v_core, t_core).  For SAE_V both are the same object
        (shared core); callers must handle identity before double-updating.
        """
        model = self.models[name]
        if hasattr(model, "v_core"):       # VL_SAE, SAE_D
            return model.v_core, model.t_core
        return model.core, model.core      # SAE_V

    # ── b_dec warm-start ────────────────────────────────────────────────────

    @torch.no_grad()
    def _init_b_dec_from_val(self, val_chunk_path: str) -> None:
        if self._b_dec_init_done:
            return
        val_data   = torch.load(val_chunk_path, map_location="cpu", weights_only=False)
        val_loader = DataLoader(PairDataset(val_data), batch_size=Config.batch_size,
                                shuffle=False, collate_fn=collate_fn)
        device    = self.device_map["SAE_V"]
        aux       = self.aux_projs["SAE_V"]
        sum_v = sum_t = None
        cnt_v = cnt_t = 0

        for idx, (v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _) in enumerate(val_loader):
            if idx >= Config.init_b_dec_batches:
                break
            v_pad  = v_pad_cpu.to(device);  t_pad  = t_pad_cpu.to(device)
            v_mask = v_mask_cpu.to(device); t_mask = t_mask_cpu.to(device)
            v_proj, t_proj = aux(v_pad, t_pad)

            if Config.train_method == "sym":
                v_in = (v_proj * v_mask.unsqueeze(-1)).sum(1) / (v_mask.sum(1, keepdim=True) + 1e-6)
                t_in = (t_proj * t_mask.unsqueeze(-1)).sum(1) / (t_mask.sum(1, keepdim=True) + 1e-6)
            else:  # filip: flat tokens
                v_in = v_proj[v_mask]
                t_in = t_proj[t_mask]

            sum_v = v_in.sum(0) if sum_v is None else sum_v + v_in.sum(0);  cnt_v += v_in.shape[0]
            sum_t = t_in.sum(0) if sum_t is None else sum_t + t_in.sum(0);  cnt_t += t_in.shape[0]

        mean_v = (sum_v / max(cnt_v, 1)).cpu() if sum_v is not None else None
        mean_t = (sum_t / max(cnt_t, 1)).cpu() if sum_t is not None else None
        for model in self.models.values():
            model.set_b_dec_from_mean(mean_v, mean_t)
        self._b_dec_init_done = True

    # ── Loss computation: SYM ───────────────────────────────────────────────

    def _compute_sae_loss_sym(
        self,
        name:   str,
        v_proj: torch.Tensor,
        t_proj: torch.Tensor,
        v_mask: torch.Tensor,
        t_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Global-pool → SAE reconstruction.  Standard SYM baseline."""
        v_g = (v_proj * v_mask.unsqueeze(-1)).sum(1) / (v_mask.sum(1, keepdim=True) + 1e-6)
        t_g = (t_proj * t_mask.unsqueeze(-1)).sum(1) / (t_mask.sum(1, keepdim=True) + 1e-6)
        _, _, _, _, loss_v, loss_t = self.models[name](
            vision_embeddings=v_g, text_embeddings=t_g, return_loss=True
        )
        return (loss_v["loss"] if loss_v else 0.0) + (loss_t["loss"] if loss_t else 0.0)

    # ── Loss computation: FILIP (per-image) ─────────────────────────────────

    def _compute_sae_loss_filip(
        self,
        name:   str,
        v_proj: torch.Tensor,
        t_proj: torch.Tensor,
        v_mask: torch.Tensor,
        t_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-image SAE training with optional asymmetric entailment.

        ┌─ For each image b ─────────────────────────────────────────────┐
        │  1. Extract v_tokens [Lv, D]  and  t_tokens [Lt, D]           │
        │  2. SAE forward (update_dead=False – batch-level update later) │
        │  3. Reconstruction loss: L2 + L1 + aux                        │
        │  4. Asymmetric entailment (if lambda > 0):                    │
        │       v_union = max_{i}(SAE_v(v_tok_i))    [D_sae]           │
        │       penalty = ReLU(t_lat_j − v_union) per text token j     │
        │       loss_ent = mean_j( penalty_j / ||t_lat_j||₁ )          │
        └────────────────────────────────────────────────────────────────┘
        After loop: update_inactive_from_flags() once per outer batch.
        This gives dead-latent counts comparable to SYM mode.
        """
        v_core, t_core = self._get_sae_cores(name)
        B      = v_proj.shape[0]
        D_sae  = v_core.hidden_dim
        device = v_proj.device

        losses: list         = []
        batch_act_v          = torch.zeros(D_sae, dtype=torch.bool, device=device)
        batch_act_t          = torch.zeros(D_sae, dtype=torch.bool, device=device)

        for b in range(B):
            lv = int(v_mask[b].sum().item())
            lt = int(t_mask[b].sum().item())
            if lv == 0 or lt == 0:
                continue

            v_tok = v_proj[b, :lv]   # [lv, D_input]
            t_tok = t_proj[b, :lt]   # [lt, D_input]

            # ── SAE forward (no internal dead-latent update) ─────────────
            _, latent_v, loss_v = v_core(v_tok, return_loss=True, update_dead=False)
            _, latent_t, loss_t = t_core(t_tok, return_loss=True, update_dead=False)

            img_loss = (
                (loss_v["loss"] if loss_v else torch.tensor(0.0, device=device))
                + (loss_t["loss"] if loss_t else torch.tensor(0.0, device=device))
            )

            # ── Asymmetric entailment constraint ─────────────────────────
            if Config.lambda_align > 0 and latent_v.numel() > 0 and latent_t.numel() > 0:
                v_union   = latent_v.detach().max(dim=0).values      # [D_sae]
                diff      = latent_t - v_union.unsqueeze(0)           # [lt, D_sae]
                penalty   = F.relu(diff).sum(dim=-1)                  # [lt]
                denom     = latent_t.detach().abs().sum(dim=-1).clamp(min=1e-6)
                ent_loss  = (penalty / denom).mean()
                img_loss  = img_loss + Config.lambda_align * ent_loss

            losses.append(img_loss)

            # ── Accumulate image-level activity flags ────────────────────
            batch_act_v.logical_or_((latent_v.detach() > 1e-5).any(dim=0))
            batch_act_t.logical_or_((latent_t.detach() > 1e-5).any(dim=0))

        # ── Update dead-latent counters ONCE per outer batch ─────────────
        # This is semantically equivalent to SYM's once-per-batch update.
        if v_core is t_core:
            # SAE_V: shared core, combine both modality flags
            v_core.update_inactive_from_flags(batch_act_v | batch_act_t)
        else:
            v_core.update_inactive_from_flags(batch_act_v)
            t_core.update_inactive_from_flags(batch_act_t)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()

    # ── Dispatcher ──────────────────────────────────────────────────────────

    def _compute_sae_loss(
        self,
        name:   str,
        v_proj: torch.Tensor,
        t_proj: torch.Tensor,
        v_mask: torch.Tensor,
        t_mask: torch.Tensor,
    ) -> torch.Tensor:
        if Config.train_method == "sym":
            return self._compute_sae_loss_sym(name, v_proj, t_proj, v_mask, t_mask)
        return self._compute_sae_loss_filip(name, v_proj, t_proj, v_mask, t_mask)

    # ── Training loop ────────────────────────────────────────────────────────

    def train_on_chunk(
        self,
        train_chunk_path: str,
        val_chunk_path: str,
        chunk_idx: int,
    ) -> None:
        print(f"\n[Phase 2] SAE – chunk {chunk_idx} | λ_align={Config.lambda_align:.5f}")
        self._init_b_dec_from_val(val_chunk_path)

        train_data   = torch.load(train_chunk_path, map_location="cpu", weights_only=False)
        val_data     = torch.load(val_chunk_path,   map_location="cpu", weights_only=False)
        train_loader = DataLoader(PairDataset(train_data), batch_size=Config.batch_size,
                                  shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(PairDataset(val_data),   batch_size=Config.batch_size,
                                  shuffle=False, collate_fn=collate_fn)

        # ── Training ────────────────────────────────────────────────────────
        for m in self.models.values():
            m.train()
        pbar = tqdm(train_loader, desc=f"Phase 2: train (chunk {chunk_idx})")

        for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _ in pbar:
            for name, dev in self.device_map.items():
                try:
                    v_pad  = v_pad_cpu.to(dev,  non_blocking=True)
                    t_pad  = t_pad_cpu.to(dev,  non_blocking=True)
                    v_mask = v_mask_cpu.to(dev, non_blocking=True)
                    t_mask = t_mask_cpu.to(dev, non_blocking=True)

                    self.optimizers[name].zero_grad()
                    with autocast("cuda"):
                        with torch.no_grad():
                            v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                        loss = self._compute_sae_loss(name, v_proj, t_proj, v_mask, t_mask)

                    self.scalers[name].scale(loss).backward()
                    self.scalers[name].unscale_(self.optimizers[name])
                    torch.nn.utils.clip_grad_norm_(self.models[name].parameters(), Config.max_grad_norm)
                    self.models[name].make_decoder_weights_and_grad_unit_norm()
                    self.scalers[name].step(self.optimizers[name])
                    self.scalers[name].update()

                    if name == "VL_SAE":
                        pbar.set_postfix({"sae_loss": f"{loss.item():.4f}"})
                    del loss, v_proj, t_proj, v_pad, t_pad
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.optimizers[name].zero_grad()

        # ── Validation ──────────────────────────────────────────────────────
        for m in self.models.values():
            m.eval()
        val_losses = {n: 0.0 for n in self.models}

        with torch.no_grad():
            for v_pad_cpu, t_pad_cpu, v_mask_cpu, t_mask_cpu, _, _ in tqdm(
                val_loader, desc="Phase 2: val", leave=False
            ):
                for name, dev in self.device_map.items():
                    try:
                        v_pad  = v_pad_cpu.to(dev,  non_blocking=True)
                        t_pad  = t_pad_cpu.to(dev,  non_blocking=True)
                        v_mask = v_mask_cpu.to(dev, non_blocking=True)
                        t_mask = t_mask_cpu.to(dev, non_blocking=True)
                        with autocast("cuda"):
                            v_proj, t_proj = self.aux_projs[name](v_pad, t_pad)
                            val_losses[name] += self._compute_sae_loss(
                                name, v_proj, t_proj, v_mask, t_mask).item()
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()

        # ── Save best checkpoints ────────────────────────────────────────────
        os.makedirs(Config.save_dir, exist_ok=True)
        print(f"--- Phase 2 validation (chunk {chunk_idx}) ---")
        for name in self.models:
            avg = val_losses[name] / max(len(val_loader), 1)
            if avg < self.best_val_loss[name]:
                self.best_val_loss[name] = avg
                path = os.path.join(Config.save_dir,
                                    f"{name}_{Config.train_method}_new_best_sae.pth")
                torch.save({"sae_state_dict": self.models[name].state_dict()}, path)
                print(f"  ✅ [{name}] {avg:.4f} → saved")
            else:
                print(f"  📉 [{name}] {avg:.4f} (best {self.best_val_loss[name]:.4f})")