"""
sae_model.py — Multimodal Sparse Autoencoder architectures.

Key modifications vs. original:
- CAFECore.forward(): added `update_dead` param (default True).
  In per-image FILIP training, pass update_dead=False and call
  update_inactive_from_flags() once per outer batch to give
  semantically correct dead-latent counts comparable to SYM.
- CAFECore.update_inactive_from_flags(): new method for batch-level update.
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

class TokenAuxProj(nn.Module):
    """
    Token-level auxiliary projection layer.

    A single linear map per modality that projects the frozen backbone
    features into a shared alignment space *without* collapsing the
    sequence dimension.  This preserves local spatial structure for the
    subsequent FILIP contrastive alignment and per-image SAE training.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.v_proj = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(dim, dim)

    def forward(
        self, v: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.v_proj(v), self.t_proj(t)


def _unit_norm_rows(w: torch.Tensor) -> torch.Tensor:
    return w / (w.norm(dim=-1, keepdim=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Core SAE module
# ─────────────────────────────────────────────────────────────────────────────

class CAFECore(nn.Module):
    """
    CaFE-style TopK Sparse Autoencoder core.

    Architecture
    ────────────
    x  →  (x - b_dec)  →  W_enc  →  ReLU  →  TopK  →  W_dec  →  + b_dec  →  x̂

    Design choices
    ──────────────
    • TopK hard-gates L0 = k exactly, avoiding L1 magnitude shrinkage.
    • Decoder columns are unit-normalised after every gradient step so
      the dot product S_m = ReLU((x−b_dec)·W_enc[:,k]) is a pure
      geometric alignment score (see feature_viz.py for attribution).
    • b_dec absorbs the data mean so the dictionary learns residual
      directions rather than the global mean.
    • Auxiliary loss revives dead features by fitting current residuals.

    Dead-latent tracking (per-image)
    ────────────────────────────────
    In SYM training each batch produces one global vector per image, so
    calling update_inactive_features() once per batch naturally counts
    "image-batches without activation".  In FILIP training hundreds of
    tokens arrive in a single flat batch, making *every* feature appear
    active → dead count stays zero.

    To fix this, FILIP training calls forward() with update_dead=False
    and calls update_inactive_from_flags() once per outer batch with a
    pre-computed image-level activity mask.  This ensures dead-latent
    counts are comparable between the two modes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        topk: int,
        cfg: Optional[Dict] = None,
        shared_W_enc: Optional[nn.Parameter] = None,
        shared_b_dec: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.topk = topk

        self.input_unit_norm    = bool(cfg.get("input_unit_norm",    False))
        self.l1_coeff           = float(cfg.get("l1_coeff",          0.0))
        self.aux_penalty        = float(cfg.get("aux_penalty",        0.0))
        self.top_k_aux          = int(cfg.get("top_k_aux",           topk))
        self.n_batches_to_dead  = int(cfg.get("n_batches_to_dead",   20))
        self.use_threshold_in_eval = bool(cfg.get("use_threshold_in_eval", False))

        # ── Encoder ────────────────────────────────────────────────────────
        if shared_W_enc is None:
            self.W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim))
            nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        else:
            self.W_enc = shared_W_enc

        if shared_b_dec is None:
            self.b_dec = nn.Parameter(torch.zeros(input_dim))
        else:
            self.b_dec = shared_b_dec

        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))

        # ── Decoder ────────────────────────────────────────────────────────
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._init_decoder_from_encoder()

        # ── Buffers ─────────────────────────────────────────────────────────
        # threshold: running minimum of active TopK activations (for fast
        # inference without the O(N log N) sort).
        self.register_buffer("threshold", torch.tensor(0.0))
        # num_batches_not_active[k]: number of consecutive outer-batch steps
        # (or per-image steps, depending on update mode) since feature k
        # last fired.  Feature is "dead" when this exceeds n_batches_to_dead.
        self.register_buffer("num_batches_not_active", torch.zeros(hidden_dim))

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_decoder_from_encoder(self) -> None:
        self.W_dec.data.copy_(self.W_enc.t().data)
        self.W_dec.data.copy_(_unit_norm_rows(self.W_dec.data))

    # ── Input pre/post processing ───────────────────────────────────────────

    def preprocess_input(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.input_unit_norm:
            return x, None, None
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True)
        x = x / (x_std + 1e-5)
        return x, x_mean, x_std

    def postprocess_output(
        self,
        x_recon: torch.Tensor,
        x_mean: Optional[torch.Tensor],
        x_std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.input_unit_norm:
            return x_recon
        return x_recon * x_std + x_mean

    # ── Activation ─────────────────────────────────────────────────────────

    def compute_activations(
        self, x_cent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pre_topk_acts, topk_acts)."""
        pre_acts = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)

        if self.training or not self.use_threshold_in_eval:
            k = min(self.topk, acts.shape[-1])
            topk_vals, topk_idx = torch.topk(acts, k=k, dim=-1)
            acts_topk = torch.zeros_like(acts).scatter(-1, topk_idx, topk_vals)
            if self.training:
                self.update_threshold(acts_topk)
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

        return acts, acts_topk

    # ── Public encode / decode ──────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_proc, _, _ = self.preprocess_input(x)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])
        x_cent = x_flat - self.b_dec
        _, acts_topk = self.compute_activations(x_cent)
        if len(orig_shape) == 3:
            return acts_topk.reshape(orig_shape[0], orig_shape[1], -1)
        return acts_topk

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
        update_dead: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Parameters
        ----------
        x           : input tensor [..., input_dim]
        return_loss : whether to compute and return the loss dict
        update_dead : whether to call update_inactive_features() on this pass.
                      Set to False in per-image FILIP training; call
                      update_inactive_from_flags() manually at batch end.
        """
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        x_cent    = x_flat - self.b_dec
        acts, acts_topk = self.compute_activations(x_cent)

        recon_flat = acts_topk @ self.W_dec + self.b_dec
        recon      = self.postprocess_output(recon_flat.reshape(orig_shape), x_mean, x_std)

        if len(orig_shape) == 3:
            acts_topk_out = acts_topk.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            acts_topk_out = acts_topk

        loss_dict = None
        if return_loss:
            if update_dead:
                self.update_inactive_features(acts_topk)
            loss_dict = self.get_loss_dict(x_flat, recon_flat, acts, acts_topk)

        return recon, acts_topk_out, loss_dict

    # ── Loss computation ────────────────────────────────────────────────────

    def get_loss_dict(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        acts: torch.Tensor,
        acts_topk: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l2_loss  = (x_recon.float() - x.float()).pow(2).mean()
        l1_norm  = acts_topk.float().abs().sum(-1).mean()
        l1_loss  = self.l1_coeff * l1_norm
        l0_norm  = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_recon, acts)
        loss     = l2_loss + l1_loss + aux_loss
        n_dead   = (self.num_batches_not_active > self.n_batches_to_dead).sum()

        return {
            "loss":             loss,
            "l2_loss":          l2_loss,
            "l1_loss":          l1_loss,
            "l0_norm":          l0_norm,
            "l1_norm":          l1_norm,
            "aux_loss":         aux_loss,
            "num_dead_features": n_dead,
            "threshold":        self.threshold,
        }

    def get_auxiliary_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        acts: torch.Tensor,
    ) -> torch.Tensor:
        """Fit dead features to the current reconstruction residual."""
        if self.aux_penalty <= 0:
            return torch.tensor(0.0, device=x.device)
        dead = self.num_batches_not_active >= self.n_batches_to_dead
        if dead.sum() == 0:
            return torch.tensor(0.0, device=x.device)

        residual      = x.float() - x_recon.float()
        aux_topk      = torch.topk(acts[:, dead], min(self.top_k_aux, int(dead.sum().item())), dim=-1)
        acts_aux      = torch.zeros_like(acts[:, dead]).scatter(-1, aux_topk.indices, aux_topk.values)
        x_recon_aux   = acts_aux @ self.W_dec[dead]
        return self.aux_penalty * (x_recon_aux.float() - residual).pow(2).mean()

    # ── Dead-latent tracking ────────────────────────────────────────────────

    @torch.no_grad()
    def update_inactive_features(self, acts_topk: torch.Tensor) -> None:
        """
        Standard per-batch update.
        Used by SYM (one global vector per image) and during validation.
        Feature k is considered active if *any* row in acts_topk fires it.
        """
        active = (acts_topk.sum(0) > 0)
        self.num_batches_not_active[active]  = 0
        self.num_batches_not_active[~active] += 1

    @torch.no_grad()
    def update_inactive_from_flags(self, active: torch.Tensor) -> None:
        """
        Per-image-batch update for FILIP training.

        `active`: bool tensor [hidden_dim] – True if this feature fired
        for **any image** in the outer training batch (aggregated across
        all of their tokens).

        Calling this *once per outer batch* (not once per token) makes
        the dead-latent counter semantically equivalent to the SYM-mode
        counter: one increment = one outer optimisation step.
        """
        self.num_batches_not_active[active]  = 0
        self.num_batches_not_active[~active] += 1

    # ── Decoder normalisation (call after each optimiser step) ─────────────

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        if self.W_dec.grad is None:
            return
        W_normed   = _unit_norm_rows(self.W_dec)
        grad_proj  = (self.W_dec.grad * W_normed).sum(-1, keepdim=True) * W_normed
        self.W_dec.grad -= grad_proj
        self.W_dec.data.copy_(W_normed)

    @torch.no_grad()
    def update_threshold(self, acts_topk: torch.Tensor, lr: float = 0.01) -> None:
        pos = acts_topk > 0
        if pos.any():
            self.threshold = (1 - lr) * self.threshold + lr * acts_topk[pos].min()

    @torch.no_grad()
    def set_b_dec_from_mean(self, mean_vec: torch.Tensor) -> None:
        self.b_dec.data.copy_(mean_vec.to(self.b_dec.device))


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal SAE wrappers
# ─────────────────────────────────────────────────────────────────────────────

class VL_SAE(nn.Module):
    """
    Shared encoder + independent decoders  (recommended for multimodal SAE).

    Both modalities share W_enc: they compete for the *same* set of
    concept directions.  This forces cross-modal concept alignment at
    the dictionary level.  Independent decoders let each modality
    reconstruct back to its own dense distribution without coupling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        topk: int = 256,
        cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        shared_W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.kaiming_uniform_(shared_W_enc, a=math.sqrt(5))
        shared_b_dec = nn.Parameter(torch.zeros(input_dim))

        self.v_core = CAFECore(input_dim, hidden_dim, topk, cfg,
                               shared_W_enc=shared_W_enc, shared_b_dec=shared_b_dec)
        self.t_core = CAFECore(input_dim, hidden_dim, topk, cfg,
                               shared_W_enc=shared_W_enc, shared_b_dec=shared_b_dec)

    def forward(
        self,
        vision_embeddings: Optional[torch.Tensor] = None,
        text_embeddings:   Optional[torch.Tensor] = None,
        return_loss: bool = False,
        update_dead: bool = True,
    ):
        recon_v = recon_t = latent_v = latent_t = loss_v = loss_t = None
        if vision_embeddings is not None:
            recon_v, latent_v, loss_v = self.v_core(
                vision_embeddings, return_loss=return_loss, update_dead=update_dead)
        if text_embeddings is not None:
            recon_t, latent_t, loss_t = self.t_core(
                text_embeddings,   return_loss=return_loss, update_dead=update_dead)
        if return_loss:
            return recon_v, recon_t, latent_v, latent_t, loss_v, loss_t
        return recon_v, recon_t, latent_v, latent_t

    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        self.v_core.make_decoder_weights_and_grad_unit_norm()
        self.t_core.make_decoder_weights_and_grad_unit_norm()

    def set_b_dec_from_mean(
        self,
        mean_v: Optional[torch.Tensor],
        mean_t: Optional[torch.Tensor],
    ) -> None:
        mean = (0.5 * (mean_v + mean_t)) if (mean_v is not None and mean_t is not None) \
               else (mean_v if mean_v is not None else mean_t)
        if mean is not None:
            self.v_core.set_b_dec_from_mean(mean)
            self.t_core.set_b_dec_from_mean(mean)


class SAE_D(nn.Module):
    """
    Fully independent encoder + decoder for each modality.

    Used as a negative-control ablation to demonstrate that *without*
    shared W_enc the cross-modal latent spaces diverge and retrieval
    collapses, proving that the shared encoder in VL_SAE is necessary.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        topk: int = 256,
        cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        self.v_core = CAFECore(input_dim, hidden_dim, topk, cfg)
        self.t_core = CAFECore(input_dim, hidden_dim, topk, cfg)

    def forward(
        self,
        vision_embeddings: Optional[torch.Tensor] = None,
        text_embeddings:   Optional[torch.Tensor] = None,
        return_loss: bool = False,
        update_dead: bool = True,
    ):
        recon_v = recon_t = latent_v = latent_t = loss_v = loss_t = None
        if vision_embeddings is not None:
            recon_v, latent_v, loss_v = self.v_core(
                vision_embeddings, return_loss=return_loss, update_dead=update_dead)
        if text_embeddings is not None:
            recon_t, latent_t, loss_t = self.t_core(
                text_embeddings,   return_loss=return_loss, update_dead=update_dead)
        if return_loss:
            return recon_v, recon_t, latent_v, latent_t, loss_v, loss_t
        return recon_v, recon_t, latent_v, latent_t

    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        self.v_core.make_decoder_weights_and_grad_unit_norm()
        self.t_core.make_decoder_weights_and_grad_unit_norm()

    def set_b_dec_from_mean(
        self,
        mean_v: Optional[torch.Tensor],
        mean_t: Optional[torch.Tensor],
    ) -> None:
        if mean_v is not None:
            self.v_core.set_b_dec_from_mean(mean_v)
        if mean_t is not None:
            self.t_core.set_b_dec_from_mean(mean_t)


class SAE_V(nn.Module):
    """
    Fully shared encoder + shared decoder.

    All parameters are shared across modalities. Used as a baseline to
    show that while cross-modal alignment is good, independent decoders
    (VL_SAE) better preserve modality-specific reconstruction fidelity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        topk: int = 256,
        cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        self.core = CAFECore(input_dim, hidden_dim, topk, cfg)

    def forward(
        self,
        vision_embeddings: Optional[torch.Tensor] = None,
        text_embeddings:   Optional[torch.Tensor] = None,
        return_loss: bool = False,
        update_dead: bool = True,
    ):
        recon_v = recon_t = latent_v = latent_t = loss_v = loss_t = None
        if vision_embeddings is not None:
            recon_v, latent_v, loss_v = self.core(
                vision_embeddings, return_loss=return_loss, update_dead=update_dead)
        if text_embeddings is not None:
            recon_t, latent_t, loss_t = self.core(
                text_embeddings,   return_loss=return_loss, update_dead=update_dead)
        if return_loss:
            return recon_v, recon_t, latent_v, latent_t, loss_v, loss_t
        return recon_v, recon_t, latent_v, latent_t

    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        self.core.make_decoder_weights_and_grad_unit_norm()

    def set_b_dec_from_mean(
        self,
        mean_v: Optional[torch.Tensor],
        mean_t: Optional[torch.Tensor],
    ) -> None:
        mean = (0.5 * (mean_v + mean_t)) if (mean_v is not None and mean_t is not None) \
               else (mean_v if mean_v is not None else mean_t)
        if mean is not None:
            self.core.set_b_dec_from_mean(mean)