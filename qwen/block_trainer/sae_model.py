import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAuxProj(nn.Module):
    """
    Token 级独立投影层。替代原论文中处理全局向量的 AuxiliaryAE。
    保持特征的细粒度空间结构，为后续的 FILIP 对齐损失做准备。
    """
    def __init__(self, dim):
        super().__init__()
        self.v_proj = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(dim, dim)

    def forward(self, v, t):
        return self.v_proj(v), self.t_proj(t)


def _unit_norm_rows(w: torch.Tensor) -> torch.Tensor:
    return w / (w.norm(dim=-1, keepdim=True) + 1e-8)

class CAFECore(nn.Module):
    """
    CaFE-style TopK SAE core.
    - Linear encoder/decoder
    - Optional input unit-norm preprocessing
    - Top-k sparsification
    - Decoder unit-norm projection
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        topk: int,
        cfg: Optional[Dict] = None,
        shared_W_enc: Optional[nn.Parameter] = None,
        shared_b_dec: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        cfg = cfg or {}

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.topk = topk

        self.input_unit_norm = bool(cfg.get("input_unit_norm", False))
        self.l1_coeff = float(cfg.get("l1_coeff", 0.0))
        self.aux_penalty = float(cfg.get("aux_penalty", 0.0))
        self.top_k_aux = int(cfg.get("top_k_aux", topk))
        self.n_batches_to_dead = int(cfg.get("n_batches_to_dead", 20))
        self.use_threshold_in_eval = bool(cfg.get("use_threshold_in_eval", False))

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
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._init_decoder_from_encoder()

        self.register_buffer("threshold", torch.tensor(0.0)) # 这是在训练时通过滑动平均学习到的 Top-K 的平均激活值，用于推理过程筛选特征，因为topk操作涉及到排序耗时，直接用threshold筛选可以大幅提升推理速度，也能灵活分配特征数量
        self.register_buffer("num_batches_not_active", torch.zeros(hidden_dim))

    def _init_decoder_from_encoder(self) -> None:
        self.W_dec.data.copy_(self.W_enc.t().data)
        self.W_dec.data.copy_(_unit_norm_rows(self.W_dec.data))

    def preprocess_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.input_unit_norm:
            return x, None, None
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True)
        x = x / (x_std + 1e-5)
        return x, x_mean, x_std

    def postprocess_output(
        self,
        x_reconstruct: torch.Tensor,
        x_mean: Optional[torch.Tensor],
        x_std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.input_unit_norm:
            return x_reconstruct
        return x_reconstruct * x_std + x_mean

    def compute_activations(self, x_cent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_acts = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)

        if self.training or not self.use_threshold_in_eval:
            k = min(self.topk, acts.shape[-1])
            topk_vals, topk_indices = torch.topk(acts, k=k, dim=-1)
            acts_topk = torch.zeros_like(acts).scatter(-1, topk_indices, topk_vals)
            if self.training:
                self.update_threshold(acts_topk) # 训练过程平滑更新threshold值
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts)) # 推理过程直接使用threshold

        return acts, acts_topk

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_proc, _, _ = self.preprocess_input(x)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])
        x_cent = x_flat - self.b_dec
        _, acts_topk = self.compute_activations(x_cent)
        if len(orig_shape) == 3:
            return acts_topk.reshape(orig_shape[0], orig_shape[1], -1)
        return acts_topk

    def decode(
        self,
        acts_topk: torch.Tensor,
        x_mean: Optional[torch.Tensor],
        x_std: Optional[torch.Tensor],
        orig_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        acts_flat = acts_topk.reshape(-1, acts_topk.shape[-1])
        recon_flat = acts_flat @ self.W_dec + self.b_dec
        recon = recon_flat.reshape(orig_shape)
        return self.postprocess_output(recon, x_mean, x_std)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        orig_shape = x.shape
        x_proc, x_mean, x_std = self.preprocess_input(x)
        x_flat = x_proc.reshape(-1, x_proc.shape[-1])

        x_cent = x_flat - self.b_dec # 减掉解码器偏置，让编码器专注于学习残差特征。我们希望 SAE 的字典特征（W_dec）去学习数据中有意义的变化（方差），而不是浪费特征容量去记住整个数据集的全局平均特征。因此，先用 b_dec 把输入平移到原点附近，编码解码完成后，再把 b_dec 加回来恢复原样。
        acts, acts_topk = self.compute_activations(x_cent)
        recon_flat = acts_topk @ self.W_dec + self.b_dec
        recon = recon_flat.reshape(orig_shape)
        recon = self.postprocess_output(recon, x_mean, x_std)

        if len(orig_shape) == 3:
            acts_topk_out = acts_topk.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            acts_topk_out = acts_topk

        loss_dict = None
        if return_loss:
            self.update_inactive_features(acts_topk)
            loss_dict = self.get_loss_dict(x_flat, recon_flat, acts, acts_topk)

        return recon, acts_topk_out, loss_dict

    def get_loss_dict(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        acts: torch.Tensor,
        acts_topk: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.l1_coeff * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (self.num_batches_not_active > self.n_batches_to_dead).sum()

        return {
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "num_dead_features": num_dead_features,
            "threshold": self.threshold,
        }

    def get_auxiliary_loss(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        acts: torch.Tensor,
    ) -> torch.Tensor:
        if self.aux_penalty <= 0:
            return torch.tensor(0.0, device=x.device)
        dead_features = self.num_batches_not_active >= self.n_batches_to_dead
        if dead_features.sum() == 0:
            return torch.tensor(0.0, device=x.device)

        residual = x.float() - x_reconstruct.float()
        acts_topk_aux = torch.topk(
            acts[:, dead_features],
            min(self.top_k_aux, int(dead_features.sum().item())),
            dim=-1,
        )
        acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
            -1, acts_topk_aux.indices, acts_topk_aux.values
        )
        x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
        return self.aux_penalty * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        if self.W_dec.grad is None:
            return
        W_dec_normed = _unit_norm_rows(self.W_dec)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data.copy_(W_dec_normed)

    @torch.no_grad()
    def update_inactive_features(self, acts_topk: torch.Tensor) -> None:
        self.num_batches_not_active += (acts_topk.sum(0) == 0).float()
        self.num_batches_not_active[acts_topk.sum(0) > 0] = 0

    @torch.no_grad()
    def update_threshold(self, acts_topk: torch.Tensor, lr: float = 0.01) -> None:
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive

    @torch.no_grad()
    def set_b_dec_from_mean(self, mean_vec: torch.Tensor) -> None:
        self.b_dec.data.copy_(mean_vec.to(self.b_dec.device))


class VL_SAE(nn.Module):
    """
    共享编码器，独立解码器。
    CaFE 风格：线性编码 + TopK 稀疏化。
    """
    def __init__(self, input_dim, hidden_dim, topk=256, dropout=0, cfg: Optional[Dict] = None):
        super().__init__()
        cfg = cfg or {}

        shared_W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.kaiming_uniform_(shared_W_enc, a=math.sqrt(5))
        shared_b_dec = nn.Parameter(torch.zeros(input_dim))

        self.v_core = CAFECore(
            input_dim,
            hidden_dim,
            topk,
            cfg=cfg,
            shared_W_enc=shared_W_enc,
            shared_b_dec=shared_b_dec,
        )
        self.t_core = CAFECore(
            input_dim,
            hidden_dim,
            topk,
            cfg=cfg,
            shared_W_enc=shared_W_enc,
            shared_b_dec=shared_b_dec,
        )

    def forward(self, vision_embeddings=None, text_embeddings=None, return_loss: bool = False):
        recon_vision_embeddings, recon_text_embeddings = None, None
        latent_v, latent_t = None, None
        loss_v, loss_t = None, None

        if vision_embeddings is not None:
            recon_vision_embeddings, latent_v, loss_v = self.v_core(
                vision_embeddings, return_loss=return_loss
            )
        if text_embeddings is not None:
            recon_text_embeddings, latent_t, loss_t = self.t_core(
                text_embeddings, return_loss=return_loss
            )

        if return_loss:
            return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t, loss_v, loss_t
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t

    def make_decoder_weights_and_grad_unit_norm(self):
        self.v_core.make_decoder_weights_and_grad_unit_norm()
        self.t_core.make_decoder_weights_and_grad_unit_norm()

    def set_b_dec_from_mean(self, mean_v: Optional[torch.Tensor], mean_t: Optional[torch.Tensor]):
        if mean_v is not None and mean_t is not None:
            mean = 0.5 * (mean_v + mean_t)
        else:
            mean = mean_v if mean_v is not None else mean_t
        if mean is not None:
            self.v_core.set_b_dec_from_mean(mean)
            self.t_core.set_b_dec_from_mean(mean)


class SAE_D(nn.Module):
    """
    独立编码器，独立解码器 (双流 SAE)。
    """
    def __init__(self, input_dim, hidden_dim, topk=256, dropout=0.1, cfg: Optional[Dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.v_core = CAFECore(input_dim, hidden_dim, topk, cfg=cfg)
        self.t_core = CAFECore(input_dim, hidden_dim, topk, cfg=cfg)

    def forward(self, vision_embeddings=None, text_embeddings=None, return_loss: bool = False):
        recon_vision_embeddings, recon_text_embeddings = None, None
        latent_v, latent_t = None, None
        loss_v, loss_t = None, None

        if vision_embeddings is not None:
            recon_vision_embeddings, latent_v, loss_v = self.v_core(
                vision_embeddings, return_loss=return_loss
            )
        if text_embeddings is not None:
            recon_text_embeddings, latent_t, loss_t = self.t_core(
                text_embeddings, return_loss=return_loss
            )

        if return_loss:
            return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t, loss_v, loss_t
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t

    def make_decoder_weights_and_grad_unit_norm(self):
        self.v_core.make_decoder_weights_and_grad_unit_norm()
        self.t_core.make_decoder_weights_and_grad_unit_norm()

    def set_b_dec_from_mean(self, mean_v: Optional[torch.Tensor], mean_t: Optional[torch.Tensor]):
        if mean_v is not None:
            self.v_core.set_b_dec_from_mean(mean_v)
        if mean_t is not None:
            self.t_core.set_b_dec_from_mean(mean_t)


class SAE_V(nn.Module):
    """
    共享编码器，共享解码器 (单流 SAE)。
    """
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0.1, cfg: Optional[Dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.core = CAFECore(input_dim, hidden_dim, topk, cfg=cfg)

    def forward(self, vision_embeddings=None, text_embeddings=None, return_loss: bool = False):
        recon_vision_embeddings, recon_text_embeddings = None, None
        latent_v, latent_t = None, None
        loss_v, loss_t = None, None

        if vision_embeddings is not None:
            recon_vision_embeddings, latent_v, loss_v = self.core(
                vision_embeddings, return_loss=return_loss
            )
        if text_embeddings is not None:
            recon_text_embeddings, latent_t, loss_t = self.core(
                text_embeddings, return_loss=return_loss
            )

        if return_loss:
            return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t, loss_v, loss_t
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t

    def make_decoder_weights_and_grad_unit_norm(self):
        self.core.make_decoder_weights_and_grad_unit_norm()

    def set_b_dec_from_mean(self, mean_v: Optional[torch.Tensor], mean_t: Optional[torch.Tensor]):
        if mean_v is not None and mean_t is not None:
            mean = 0.5 * (mean_v + mean_t)
        else:
            mean = mean_v if mean_v is not None else mean_t
        if mean is not None:
            self.core.set_b_dec_from_mean(mean)