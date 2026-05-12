import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from config import Config
    from hooks import InputHook
    from sae_model import SAE_V, SAE_D, VL_SAE
    from sae_model import CAFECore
except ImportError:
    from block_trainer.config import Config
    from block_trainer.hooks import InputHook
    from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE
    from block_trainer.sae_model import CAFECore


@dataclass
class SampleItem:
    dim: int
    seq_idx: int
    input_name: str
    caption: str
    modality: str
    group: str
    row_id: int
    view_center_x: Optional[float]
    view_center_y: Optional[float]


def deterministic_centers(num_views: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = int(math.ceil(math.sqrt(num_views)))
    xs = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
    ys = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return centers[:num_views]


def parse_layer_index(target_layer_name: str) -> int:
    parts = target_layer_name.split(".")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot parse layer index from '{target_layer_name}'")


def force_eager_attention(model) -> None:
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(model.config, attr):
            setattr(model.config, attr, "eager")


def compute_attn_rollout(
    attentions: Tuple[torch.Tensor, ...],
    score: torch.Tensor,
    layer_end: int,
    valid_len: int,
) -> Optional[torch.Tensor]:
    if not attentions:
        return None

    layer_end = min(layer_end, len(attentions) - 1)
    cams: List[torch.Tensor] = []
    for layer_idx in range(layer_end + 1):
        attn = attentions[layer_idx]
        if attn is None:
            continue
        attn = attn[0, :, :valid_len, :valid_len]
        grad = torch.autograd.grad(score, attn, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            continue
        cam = torch.relu(grad * attn)
        cam = cam.mean(dim=0)
        cam = cam + torch.eye(cam.shape[-1], device=cam.device)
        cam = cam / cam.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        cams.append(cam)

    if not cams:
        return None

    R = torch.eye(cams[0].shape[-1], device=cams[0].device)
    for cam in cams:
        R = cam @ R
    return R


def infer_sae_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    for key, value in state_dict.items():
        if key.endswith("W_enc") and value.ndim == 2:
            return value.shape[0], value.shape[1]
    raise ValueError("Cannot infer SAE dims from state_dict.")


def load_sae_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "sae_state_dict" in ckpt:
        return ckpt["sae_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported SAE checkpoint format")


def build_sae_model(sae_type: str, state_dict: Dict[str, torch.Tensor], topk: int, cfg: Dict) -> torch.nn.Module:
    input_dim, hidden_dim = infer_sae_dims(state_dict)
    if sae_type == "SAE_V":
        model = SAE_V(input_dim, hidden_dim, topk=topk, cfg=cfg)
    elif sae_type == "SAE_D":
        model = SAE_D(input_dim, hidden_dim, topk=topk, cfg=cfg)
    elif sae_type == "VL_SAE":
        model = VL_SAE(input_dim, hidden_dim, topk=topk, cfg=cfg)
    else:
        raise ValueError(f"Unknown sae_type: {sae_type}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_aux_proj(path: str, device: torch.device):
    from sae_model import TokenAuxProj
    proj = TokenAuxProj(Config.qwen_hidden_dim).to(device)
    proj.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    proj.eval()
    for p in proj.parameters():
        p.requires_grad = False
    return proj


def get_sae_params(sae: torch.nn.Module, sae_type: str, modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if sae_type == "SAE_V":
        core = sae.core
    elif sae_type == "SAE_D":
        core = sae.v_core if modality == "vision" else sae.t_core
    elif sae_type == "VL_SAE":
        core = sae.v_core if modality == "vision" else sae.t_core
    else:
        raise ValueError(f"Unknown sae_type: {sae_type}")
    return core.W_enc, core.b_dec


def gaussian_views(
    v_proj: torch.Tensor,
    grid_thw: torch.Tensor,
    centers: Optional[torch.Tensor],
    num_views: int,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Lv, D = v_proj.shape
    device = v_proj.device
    H = int(grid_thw[1].item())
    W = int(grid_thw[2].item())
    L = min(H * W, Lv)

    y_coords = (torch.arange(H, device=device) + 0.5) / H
    x_coords = (torch.arange(W, device=device) + 0.5) / W
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)[:L]

    if centers is None:
        centers = deterministic_centers(num_views, device, coords.dtype)
    if centers.ndim == 2:
        centers = centers.unsqueeze(0)
    if centers.shape[0] == 1 and B > 1:
        centers = centers.expand(B, -1, -1)

    diff = centers.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    m = torch.exp(-gamma * dist_sq)

    v_views = torch.zeros(B, m.shape[1], D, device=device)
    for b in range(B):
        numerator = torch.mm(m[b], v_proj[b, :L, :])
        denominator = m[b].sum(dim=1, keepdim=True) + 1e-6
        v_views[b] = numerator / denominator

    return v_views, m


def normalize_heatmap(h: torch.Tensor) -> torch.Tensor:
    h_min = h.min()
    h_max = h.max()
    if (h_max - h_min) < 1e-8:
        return torch.zeros_like(h)
    return (h - h_min) / (h_max - h_min)


def gaussian_heatmap_2d(H: int, W: int, center_r: int, center_c: int, sigma: float = 1.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist_sq = (yy - center_r) ** 2 + (xx - center_c) ** 2
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    import matplotlib.cm as cm

    cmap = cm.get_cmap("hot")
    heat_rgb = cmap(heatmap)[:, :, :3]
    heat_rgb = (heat_rgb * 255).astype(np.uint8)

    base = image.astype(np.float32)
    overlay = base * (1 - alpha) + heat_rgb.astype(np.float32) * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def build_naive_vision_overlay(
    image_path: str,
    grid_thw: Optional[torch.Tensor],
    seq_idx: int,
    view_center_x: Optional[float],
    view_center_y: Optional[float],
) -> Optional[np.ndarray]:
    if grid_thw is None:
        return None
    H = int(grid_thw[1].item())
    W = int(grid_thw[2].item())
    if H <= 0 or W <= 0:
        return None

    if view_center_x is not None and view_center_y is not None:
        center_r = int(round(view_center_y * H - 0.5))
        center_c = int(round(view_center_x * W - 0.5))
    else:
        if seq_idx < 0 or seq_idx >= H * W:
            return None
        center_r, center_c = divmod(seq_idx, W)

    center_r = max(0, min(H - 1, center_r))
    center_c = max(0, min(W - 1, center_c))

    heat = gaussian_heatmap_2d(H, W, center_r, center_c, sigma=1.0)
    heat = normalize_heatmap(heat)
    heat = F.interpolate(heat[None, None], size=(224, 224), mode="nearest").squeeze().cpu().numpy()

    image = Image.open(image_path).convert("RGB")
    image = np.array(image.resize((224, 224)))
    return overlay_heatmap(image, heat)


def build_text_heatmap_html(tokens: List[str], scores: np.ndarray, highlight_idx: int) -> str:
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    html = ["<html><body style='font-family: monospace;'>"]
    for i, (tok, s) in enumerate(zip(tokens, scores)):
        bg = int(255 * (1 - s))
        color = f"rgb(255,{bg},{bg})"
        border = "1px solid #000" if i == highlight_idx else "none"
        safe_tok = tok.replace(" ", "&nbsp;")
        html.append(f"<span style='background:{color};border:{border};padding:2px;margin:1px;'>" + safe_tok + "</span>")
    html.append("</body></html>")
    return "".join(html)


def write_text_topk_files(
    out_dir: str,
    base_name: str,
    tokens: List[str],
    scores: np.ndarray,
    highlight_idx: int,
    topk: int,
) -> None:
    order = np.argsort(scores)[::-1]
    topk = min(topk, len(order))
    top_idx = order[:topk]

    payload = []
    for idx in top_idx:
        payload.append(
            {
                "idx": int(idx),
                "token": tokens[int(idx)],
                "score": float(scores[int(idx)]),
                "highlight": bool(idx == highlight_idx),
            }
        )

    json_path = os.path.join(out_dir, f"{base_name}_topk.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, f"{base_name}_topk.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "token", "score", "highlight"])
        writer.writeheader()
        writer.writerows(payload)


def select_samples(df: pd.DataFrame, dim: int) -> List[SampleItem]:
    df_dim = df[df["dim"] == dim]
    df_sorted = df_dim.sort_values("value", ascending=False)
    top10 = df_sorted.head(10)
    top_ids = set(top10["row_id"].tolist())

    bucket_names = [f"bucket_{i:02d}" for i in range(1, 10)] + ["bucket_random"]
    bucket_samples = []
    for b in bucket_names:
        cand = df_dim[(df_dim["group"] == b) & (~df_dim["row_id"].isin(top_ids))]
        cand = cand.sort_values("value", ascending=False).head(2)
        bucket_samples.append(cand)

    selected = pd.concat([top10] + bucket_samples, ignore_index=True)
    samples = []
    def _opt_float(v):
        if pd.isna(v):
            return None
        return float(v)

    for _, row in selected.iterrows():
        samples.append(
            SampleItem(
                dim=int(row["dim"]),
                seq_idx=int(row["seq_idx"]),
                input_name=str(row["input_name"]),
                caption=str(row.get("caption", "")) if not pd.isna(row.get("caption", "")) else "",
                modality=str(row.get("modality", "vision")),
                group=str(row["group"]),
                row_id=int(row["row_id"]),
                view_center_x=_opt_float(row.get("view_center_x")),
                view_center_y=_opt_float(row.get("view_center_y")),
            )
        )
    return samples


def compute_attribution(
    model,
    processor,
    aux_proj,
    sae,
    sae_type: str,
    item: SampleItem,
    method: str,
    target_layer_name: str,
    train_method: str,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Dict[str, object]]]:
    image_path = item.input_name
    caption = item.caption

    if not os.path.exists(image_path):
        return None, None, None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": caption},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    grid_thw = inputs.get("image_grid_thw")
    grid_thw = grid_thw[0].detach().cpu() if grid_thw is not None else None

    input_ids = inputs["input_ids"][0]
    attn_mask = inputs["attention_mask"][0]
    vision_start_indices = torch.where(input_ids == model.config.vision_start_token_id)[0]
    vision_end_indices = torch.where(input_ids == model.config.vision_end_token_id)[0]
    if len(vision_start_indices) == 0 or len(vision_end_indices) == 0:
        return None, None, None

    img_st = vision_start_indices[0].item()
    img_ed = vision_end_indices[0].item()
    valid_len = attn_mask.sum().item()

    text_start = img_ed + 1
    text_end = valid_len
    token_ids = inputs["input_ids"][0][text_start:text_end].tolist()
    tokenizer = processor.tokenizer
    token_strs = [tokenizer.decode([t]) for t in token_ids]

    method = method.lower()
    if method in ("integrated_gradients", "ig"):
        method = "ig"
    if method in ("attnlrp", "lrp"):
        method = "attnlrp"

    if method == "naive":
        if item.modality == "vision":
            overlay = build_naive_vision_overlay(
                image_path,
                grid_thw,
                item.seq_idx,
                item.view_center_x,
                item.view_center_y,
            )
            return overlay, None, None

        if not token_strs:
            return None, None, None

        highlight_idx = min(item.seq_idx, len(token_strs) - 1)
        scores = np.zeros(len(token_strs), dtype=np.float32)
        scores[highlight_idx] = 1.0
        html = build_text_heatmap_html(token_strs, scores, highlight_idx)
        meta = {
            "tokens": token_strs,
            "scores": scores,
            "highlight_idx": highlight_idx,
        }
        return None, html, meta

    if method == "attnlrp":
        layer_index = parse_layer_index(target_layer_name)
        if layer_index == 0:
            return None, None, None
        force_eager_attention(model)

        with torch.enable_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        attentions = outputs.attentions
        hidden_states = outputs.hidden_states
        if attentions is None or hidden_states is None:
            return None, None, None
        if layer_index >= len(hidden_states):
            return None, None, None

        hidden = hidden_states[layer_index]
        v_feat = hidden[0, img_st + 1 : img_ed, :]
        t_feat = hidden[0, img_ed + 1 : valid_len, :] if valid_len > img_ed + 1 else None
        if t_feat is None or t_feat.numel() == 0:
            return None, None, None

        v_mask = torch.ones(1, v_feat.shape[0], device=device, dtype=torch.bool)
        t_mask = torch.ones(1, t_feat.shape[0], device=device, dtype=torch.bool)

        v_proj, t_proj = aux_proj(v_feat.unsqueeze(0), t_feat.unsqueeze(0))

        view_mask = None
        if train_method == "sym":
            v_sum = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1)
            v_tokens = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
            t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
            t_tokens = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
            v_tokens = v_tokens.unsqueeze(1)
            t_tokens = t_tokens.unsqueeze(1)
        elif train_method == "filip":
            v_tokens = v_proj
            t_tokens = t_proj
        else:
            if grid_thw is None:
                v_tokens = v_proj[:, :1, :]
            else:
                centers = deterministic_centers(Config.num_views, device, v_proj.dtype)
                v_tokens, view_mask = gaussian_views(
                    v_proj,
                    grid_thw,
                    centers,
                    Config.num_views,
                    Config.gamma,
                )
            t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
            t_tokens = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
            t_tokens = t_tokens.unsqueeze(1)

        W_enc, b_dec = get_sae_params(sae, sae_type, item.modality)
        W_enc = W_enc.to(device)
        b_dec = b_dec.to(device)

        if item.modality == "vision":
            tokens = v_tokens
        else:
            tokens = t_tokens

        if item.seq_idx >= tokens.shape[1]:
            return None, None, None

        tok = tokens[0, item.seq_idx]
        score = (tok - b_dec) @ W_enc[:, item.dim]

        layer_end = max(layer_index - 1, 0)
        R = compute_attn_rollout(attentions, score, layer_end, valid_len)
        if R is None:
            return None, None, None

        vision_indices = torch.arange(img_st + 1, img_ed, device=R.device)
        text_indices = torch.arange(text_start, text_end, device=R.device)
        if len(vision_indices) == 0 or len(text_indices) == 0:
            return None, None, None

        if train_method == "sym":
            if item.modality == "vision":
                row = R[vision_indices].mean(dim=0)
            else:
                row = R[text_indices].mean(dim=0)
        elif train_method == "asym" and item.modality == "vision" and view_mask is not None:
            if item.seq_idx >= view_mask.shape[1]:
                return None, None, None
            L = min(view_mask.shape[2], len(vision_indices))
            weights = view_mask[0, item.seq_idx, :L]
            weights = weights / weights.sum().clamp(min=1e-6)
            row = (weights.unsqueeze(0) @ R[vision_indices[:L]]).squeeze(0)
        elif train_method == "asym" and item.modality == "text":
            row = R[text_indices].mean(dim=0)
        else:
            if item.modality == "vision":
                if item.seq_idx >= len(vision_indices):
                    return None, None, None
                target_abs = vision_indices[item.seq_idx]
            else:
                if item.seq_idx >= len(text_indices):
                    return None, None, None
                target_abs = text_indices[item.seq_idx]
            row = R[target_abs]

        if item.modality == "vision":
            if grid_thw is None:
                return None, None, None
            H = int(grid_thw[1].item())
            W = int(grid_thw[2].item())
            L = min(H * W, len(vision_indices))
            token_scores = row[vision_indices[:L]].reshape(1, 1, H, W)
            token_scores = normalize_heatmap(token_scores.squeeze())
            token_scores = token_scores.unsqueeze(0).unsqueeze(0)
            heat = F.interpolate(token_scores, size=(224, 224), mode="nearest").squeeze().cpu().numpy()

            image = Image.open(image_path).convert("RGB")
            image = np.array(image.resize((224, 224)))
            overlay = overlay_heatmap(image, heat)
            return overlay, None, None

        token_scores = row[text_indices].detach().cpu().numpy()
        highlight_idx = min(item.seq_idx, len(token_strs) - 1)
        html = build_text_heatmap_html(token_strs, token_scores, highlight_idx)
        meta = {
            "tokens": token_strs,
            "scores": token_scores,
            "highlight_idx": highlight_idx,
        }
        return None, html, meta

    with InputHook(model, outputs=[target_layer_name], as_tensor=True) as h:
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, use_cache=True)
        hidden = h.layer_outputs[target_layer_name]
        if isinstance(hidden, tuple):
            hidden = hidden[0]
    if not torch.is_tensor(hidden):
        return None, None, None

    v_feat = hidden[0, img_st + 1 : img_ed, :]
    t_feat = hidden[0, img_ed + 1 : valid_len, :] if valid_len > img_ed + 1 else None
    if t_feat is None or t_feat.numel() == 0:
        return None, None, None

    proj_device = next(aux_proj.parameters()).device
    proj_dtype = next(aux_proj.parameters()).dtype
    v_feat = v_feat.to(proj_device, dtype=proj_dtype)
    t_feat = t_feat.to(proj_device, dtype=proj_dtype)

    v_mask = torch.ones(1, v_feat.shape[0], device=proj_device, dtype=torch.bool)
    t_mask = torch.ones(1, t_feat.shape[0], device=proj_device, dtype=torch.bool) if t_feat is not None else None

    v_proj, t_proj = aux_proj(v_feat.unsqueeze(0), t_feat.unsqueeze(0))

    v_proj = v_proj.detach().requires_grad_(True)
    t_proj = t_proj.detach().requires_grad_(True)

    if train_method == "sym":
        v_sum = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1)
        v_tokens = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
        t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
        t_tokens = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
        v_tokens = v_tokens.unsqueeze(1)
        t_tokens = t_tokens.unsqueeze(1)
        view_mask = None
    elif train_method == "filip":
        v_tokens = v_proj
        t_tokens = t_proj
        view_mask = None
    else:
        if grid_thw is None:
            v_tokens = v_proj[:, :1, :]
            view_mask = None
        else:
            centers = deterministic_centers(Config.num_views, proj_device, v_proj.dtype)
            v_tokens, view_mask = gaussian_views(
                v_proj,
                grid_thw,
                centers,
                Config.num_views,
                Config.gamma,
            )
        t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
        t_tokens = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
        t_tokens = t_tokens.unsqueeze(1)

    W_enc, b_dec = get_sae_params(sae, sae_type, item.modality)
    W_enc = W_enc.to(proj_device)
    b_dec = b_dec.to(proj_device)

    if item.modality == "vision":
        tokens = v_tokens
    else:
        tokens = t_tokens

    if item.seq_idx >= tokens.shape[1]:
        return None, None, None

    tok = tokens[0, item.seq_idx]
    score = (tok - b_dec) @ W_enc[:, item.dim]

    target_proj = v_proj if item.modality == "vision" else t_proj
    grad_tokens = torch.autograd.grad(score, target_proj, retain_graph=False, create_graph=False, allow_unused=True)[0]
    if grad_tokens is None:
        return None, None, None

    proj_tokens = target_proj.detach()[0]
    if method == "grad":
        attr_tokens = grad_tokens[0]
    elif method == "ig":
        attr_tokens = grad_tokens[0] * proj_tokens
    elif method == "attnlrp":
        # Token-level LRP-style relevance (positive contributions only)
        centered = proj_tokens - b_dec
        attr_tokens = torch.relu(grad_tokens[0] * centered)
    else:
        raise ValueError(f"Unknown method: {method}")

    if item.modality == "vision":
        if grid_thw is None:
            return None, None, None
        H = int(grid_thw[1].item())
        W = int(grid_thw[2].item())
        if attr_tokens.shape[0] < H * W:
            return None, None, None
        token_scores = attr_tokens[: H * W].norm(dim=-1).reshape(1, 1, H, W)
        token_scores = normalize_heatmap(token_scores.squeeze())
        token_scores = token_scores.unsqueeze(0).unsqueeze(0)
        heat = F.interpolate(token_scores, size=(224, 224), mode="nearest").squeeze().cpu().numpy()

        image = Image.open(image_path).convert("RGB")
        image = np.array(image.resize((224, 224)))
        overlay = overlay_heatmap(image, heat)
        return overlay, None, None

    # text
    token_scores = attr_tokens.norm(dim=-1).detach().cpu().numpy()
    highlight_idx = min(item.seq_idx, len(token_strs) - 1)
    html = build_text_heatmap_html(token_strs, token_scores, highlight_idx)
    meta = {
        "tokens": token_strs,
        "scores": token_scores,
        "highlight_idx": highlight_idx,
    }
    return None, html, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-layer-name", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--sae-type", choices=["SAE_V", "SAE_D", "VL_SAE"], default="VL_SAE")
    parser.add_argument("--train-method", choices=["filip", "asym", "sym"], default="filip")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--aux-proj-path", default="")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--method", default="grad,ig,attnlrp")
    parser.add_argument("--modality", choices=["vision", "text"], default="vision")
    parser.add_argument("--dims", default="")
    parser.add_argument("--max-dims", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-topk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.save_dir:
        Config.save_dir = args.save_dir

    device = torch.device(args.device)
    cfg = {
        "input_unit_norm": Config.input_unit_norm,
        "l1_coeff": Config.l1_coeff,
        "aux_penalty": Config.aux_penalty,
        "top_k_aux": Config.top_k_aux,
        "n_batches_to_dead": Config.n_batches_to_dead,
        "use_threshold_in_eval": Config.use_threshold_in_eval,
    }

    df = pd.read_csv(args.csv_path)
    df = df[df["modality"] == args.modality]

    if args.dims:
        dims = [int(x) for x in args.dims.split(",") if x.strip()]
    else:
        dims = sorted(df["dim"].unique().tolist())
        if args.max_dims > 0:
            dims = dims[: args.max_dims]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    sae_state = load_sae_checkpoint(args.sae_checkpoint)
    sae = build_sae_model(args.sae_type, sae_state, args.topk, cfg).to(device)

    aux_path = args.aux_proj_path or os.path.join(
        Config.save_dir, f"shared_best_aux_proj_{args.train_method}.pth"
    )
    aux_proj = load_aux_proj(aux_path, device)

    methods = [m.strip() for m in args.method.split(",") if m.strip()]

    for dim in tqdm(dims, desc="dims"):
        samples = select_samples(df, dim)
        for method in methods:
            for item in samples:
                overlay, html, text_meta = compute_attribution(
                    model,
                    processor,
                    aux_proj,
                    sae,
                    args.sae_type,
                    item,
                    method,
                    args.target_layer_name,
                    args.train_method,
                    device,
                )

                out_dir = os.path.join(args.output_dir, f"concept_{dim}", method, item.modality)
                os.makedirs(out_dir, exist_ok=True)

                base_name = f"{item.group}_id{item.row_id}_seq{item.seq_idx}"
                if item.modality == "vision" and overlay is not None:
                    fname = f"{base_name}.png"
                    Image.fromarray(overlay).save(os.path.join(out_dir, fname))
                if item.modality == "text" and html is not None:
                    fname = f"{base_name}.html"
                    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                        f.write(html)
                    if text_meta is not None:
                        write_text_topk_files(
                            out_dir,
                            base_name,
                            text_meta["tokens"],
                            text_meta["scores"],
                            text_meta["highlight_idx"],
                            args.text_topk,
                        )


if __name__ == "__main__":
    main()
