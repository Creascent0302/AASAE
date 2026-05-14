import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from config import Config
    from hooks import InputHook, OutputHook
    from sae_model import SAE_V, SAE_D, VL_SAE
except ImportError:
    from block_trainer.config import Config
    from block_trainer.hooks import InputHook, OutputHook
    from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE


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


@dataclass
class OverlayConfig:
    alpha: float
    gamma: float
    cmap: str
    clip_low: float
    clip_high: float


def extract_tensor(value: object) -> Optional[torch.Tensor]:
    stack = [value]
    while stack:
        current = stack.pop()
        if torch.is_tensor(current):
            return current
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return None


def make_pair_id(input_name: str, caption: str) -> str:
    base = Path(input_name).stem
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    if caption:
        digest = hashlib.md5(caption.encode("utf-8")).hexdigest()[:8]
    else:
        digest = "nocap"
    return f"{base}_{digest}"


def infer_sae_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    for key, value in state_dict.items():
        if key.endswith("W_enc") and value.ndim == 2:
            return value.shape[0], value.shape[1]
    sample_keys = list(state_dict.keys())[:8]
    raise ValueError(
        "Cannot infer SAE dims from state_dict. Expected a 2D 'W_enc' tensor. "
        f"Sample keys: {sample_keys}"
    )


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


def normalize_heatmap(h: torch.Tensor) -> torch.Tensor:
    h_min = h.min()
    h_max = h.max()
    if (h_max - h_min) < 1e-8:
        return torch.zeros_like(h)
    return (h - h_min) / (h_max - h_min)


def stretch_heatmap(heatmap: np.ndarray, clip_low: float, clip_high: float) -> np.ndarray:
    lo = float(np.quantile(heatmap, clip_low))
    hi = float(np.quantile(heatmap, clip_high))
    if hi - lo < 1e-8:
        return np.zeros_like(heatmap)
    return np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)


def tokens_to_heatmap(token_scores: torch.Tensor, H: int, W: int) -> torch.Tensor:
    L = token_scores.shape[0]
    if L == H * W:
        grid = token_scores.view(1, 1, H, W)
    else:
        aspect = W / max(H, 1)
        W_eff = max(1, int(round(math.sqrt(L * aspect))))
        H_eff = max(1, math.ceil(L / W_eff))
        pad = H_eff * W_eff - L
        if pad > 0:
            token_scores = F.pad(token_scores, (0, pad))
        grid = token_scores.view(1, 1, H_eff, W_eff)
        grid = F.interpolate(grid, size=(H, W), mode="nearest")
    return normalize_heatmap(grid.squeeze())


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "jet",
) -> np.ndarray:
    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)
    heat_rgb = cmap(heatmap)[:, :, :3]
    heat_rgb = (heat_rgb * 255).astype(np.uint8)

    base = image.astype(np.float32)
    overlay = base * (1 - alpha) + heat_rgb.astype(np.float32) * alpha
    return overlay.clip(0, 255).astype(np.uint8)
def enhance_heatmap(heatmap: np.ndarray, gamma: float) -> np.ndarray:
    heat = np.clip(heatmap, 0.0, 1.0)
    if gamma <= 0:
        return heat
    return np.power(heat, gamma)


def render_text_heatmap_image(
    tokens: List[str],
    scores: np.ndarray,
    cmap_name: str,
    max_width: int = 1024,
    pad: int = 4,
    line_spacing: int = 6,
) -> Image.Image:
    import matplotlib.cm as cm

    font = ImageFont.load_default()
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    def _tok_size(tok: str) -> Tuple[int, int]:
        bbox = draw.textbbox((0, 0), tok, font=font)
        w = (bbox[2] - bbox[0]) + pad * 2
        h = (bbox[3] - bbox[1]) + pad * 2
        return w, h

    lines: List[List[Tuple[int, str, int, int]]] = []
    current: List[Tuple[int, str, int, int]] = []
    width = 0
    max_line_w = 0

    for idx, tok in enumerate(tokens):
        tok = tok.replace("\n", "\\n")
        w, h = _tok_size(tok)
        if current and width + w > max_width:
            lines.append(current)
            max_line_w = max(max_line_w, width)
            current = []
            width = 0
        current.append((idx, tok, w, h))
        width += w

    if current:
        lines.append(current)
        max_line_w = max(max_line_w, width)

    line_heights = [max(h for _, _, _, h in line) for line in lines]
    img_h = sum(line_heights) + line_spacing * max(len(lines) - 1, 0) + pad * 2
    img_w = max(max_line_w + pad * 2, 1)

    image = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    cmap = cm.get_cmap(cmap_name)

    y = pad
    for line, line_h in zip(lines, line_heights):
        x = pad
        for idx, tok, w, _ in line:
            score = float(scores[idx]) if idx < len(scores) else 0.0
            r, g, b, _ = cmap(score)
            rgb = (int(r * 255), int(g * 255), int(b * 255))
            lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
            txt_color = (0, 0, 0) if lum > 0.5 else (255, 255, 255)
            draw.rectangle([x, y, x + w, y + line_h], fill=rgb)
            draw.text((x + pad, y + pad), tok, fill=txt_color, font=font)
            x += w
        y += line_h + line_spacing

    return image


def build_text_heatmap_html(
    tokens: List[str],
    scores: np.ndarray,
    highlight_idx: int,
    cmap_name: str,
) -> str:
    import matplotlib.cm as cm

    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    cmap = cm.get_cmap(cmap_name)
    html = ["<html><body style='font-family: monospace;'>"]
    for i, (tok, s) in enumerate(zip(tokens, scores)):
        r, g, b, _ = cmap(float(s))
        color = f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
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
    device: torch.device,
    overlay_cfg: OverlayConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[Dict[str, object]]]:
    image_path = item.input_name
    caption = item.caption

    if not os.path.exists(image_path):
        return None, None, None, None

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
        return None, None, None, None

    img_st = vision_start_indices[0].item()
    img_ed = vision_end_indices[0].item()
    valid_len = attn_mask.sum().item()

    text_start = img_ed + 1
    text_end = valid_len
    token_ids = inputs["input_ids"][0][text_start:text_end].tolist()
    tokenizer = processor.tokenizer
    token_strs = [tokenizer.decode([t]) for t in token_ids]
    if item.modality == "text" and not token_strs:
        return None, None, None, None

    method = method.lower()
    if method in ("activation", "act"):
        method = "act"
    if method != "act":
        raise ValueError("Only 'act' (activation) is supported for visualization.")

    with InputHook(model, outputs=[target_layer_name], as_tensor=True) as h:
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, use_cache=True)
        hidden = extract_tensor(h.layer_outputs.get(target_layer_name))
    if hidden is None:
        with OutputHook(model, outputs=[target_layer_name], as_tensor=True) as h:
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=1, use_cache=True)
            hidden = extract_tensor(h.layer_outputs.get(target_layer_name))
    if hidden is None:
        return None, None, None, None

    v_feat = hidden[0, img_st + 1 : img_ed, :]
    t_feat = hidden[0, img_ed + 1 : valid_len, :] if valid_len > img_ed + 1 else None
    if (t_feat is None or t_feat.numel() == 0) and item.modality == "text":
        return None, None, None, None

    proj_device = next(aux_proj.parameters()).device
    proj_dtype = next(aux_proj.parameters()).dtype
    v_feat = v_feat.to(proj_device, dtype=proj_dtype)
    if t_feat is None or t_feat.numel() == 0:
        t_feat = torch.zeros(1, v_feat.shape[1], device=proj_device, dtype=proj_dtype)
    else:
        t_feat = t_feat.to(proj_device, dtype=proj_dtype)

    v_proj, t_proj = aux_proj(v_feat.unsqueeze(0), t_feat.unsqueeze(0))
    v_proj = v_proj.float()
    t_proj = t_proj.float()

    W_enc, b_dec = get_sae_params(sae, sae_type, item.modality)
    W_enc = W_enc.to(proj_device, dtype=v_proj.dtype)
    b_dec = b_dec.to(proj_device, dtype=v_proj.dtype)
    target_proj = v_proj[0] if item.modality == "vision" else t_proj[0]
    token_scores = (target_proj - b_dec) @ W_enc[:, item.dim]
    token_scores = torch.relu(token_scores)

    if item.modality == "vision":
        if grid_thw is None:
            L = int(token_scores.numel())
            side = max(1, int(math.ceil(math.sqrt(L))))
            H = side
            W = side
        else:
            H = int(grid_thw[1].item())
            W = int(grid_thw[2].item())
        if token_scores.numel() == 0:
            return None, None, None, None
        token_scores = tokens_to_heatmap(token_scores, H, W)
        heat = F.interpolate(
            token_scores.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="nearest",
        ).squeeze().detach().cpu().numpy()
        heat = stretch_heatmap(heat, overlay_cfg.clip_low, overlay_cfg.clip_high)
        heat = enhance_heatmap(heat, overlay_cfg.gamma)

        image = Image.open(image_path).convert("RGB")
        image = np.array(image.resize((224, 224)))
        overlay = overlay_heatmap(
            image,
            heat,
            alpha=overlay_cfg.alpha,
            cmap_name=overlay_cfg.cmap,
        )
        return overlay, None, None, None

    # text
    token_scores = token_scores.detach().cpu().numpy()
    if len(token_scores) != len(token_strs):
        n = min(len(token_scores), len(token_strs))
        token_scores = token_scores[:n]
        token_strs = token_strs[:n]
    token_scores = stretch_heatmap(token_scores, overlay_cfg.clip_low, overlay_cfg.clip_high)
    token_scores = enhance_heatmap(token_scores, overlay_cfg.gamma)
    highlight_idx = min(item.seq_idx, len(token_strs) - 1)
    html = build_text_heatmap_html(token_strs, token_scores, highlight_idx, overlay_cfg.cmap)
    text_img = render_text_heatmap_image(token_strs, token_scores, overlay_cfg.cmap)
    meta = {
        "tokens": token_strs,
        "scores": token_scores,
        "highlight_idx": highlight_idx,
    }
    return None, text_img, html, meta


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
    parser.add_argument("--method", default="act")
    parser.add_argument("--modality", choices=["vision", "text"], default="vision")
    parser.add_argument("--dims", default="")
    parser.add_argument("--max-dims", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-topk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--overlay-gamma", type=float, default=0.6)
    parser.add_argument("--overlay-cmap", default="jet")
    parser.add_argument("--overlay-clip-low", type=float, default=0.05)
    parser.add_argument("--overlay-clip-high", type=float, default=0.95)
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

    if args.train_method != "sym":
        raise ValueError("Visualization is simplified to sym baseline. Set --train-method sym.")

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
    for m in methods:
        if m not in ("act", "activation"):
            raise ValueError("Only 'act' (activation) is supported in the simplified visualizer.")

    overlay_cfg = OverlayConfig(
        alpha=args.overlay_alpha,
        gamma=args.overlay_gamma,
        cmap=args.overlay_cmap,
        clip_low=args.overlay_clip_low,
        clip_high=args.overlay_clip_high,
    )

    for dim in tqdm(dims, desc="dims"):
        samples = select_samples(df, dim)
        for method in methods:
            for item in samples:
                overlay, text_img, html, text_meta = compute_attribution(
                    model,
                    processor,
                    aux_proj,
                    sae,
                    args.sae_type,
                    item,
                    method,
                    args.target_layer_name,
                    device,
                    overlay_cfg,
                )

                out_dir = os.path.join(args.output_dir, f"concept_{dim}", method, item.modality)
                os.makedirs(out_dir, exist_ok=True)
                pair_id = make_pair_id(item.input_name, item.caption)
                pair_dir = os.path.join(args.output_dir, f"concept_{dim}", method, "pairs", pair_id)
                os.makedirs(pair_dir, exist_ok=True)

                base_name = f"{pair_id}_{item.group}_seq{item.seq_idx}"
                if item.modality == "vision" and overlay is not None:
                    fname = f"{base_name}.png"
                    Image.fromarray(overlay).save(os.path.join(out_dir, fname))
                    Image.fromarray(overlay).save(
                        os.path.join(pair_dir, f"vision_seq{item.seq_idx}.png")
                    )
                if item.modality == "text" and html is not None:
                    fname = f"{base_name}.html"
                    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                        f.write(html)
                    if text_img is not None:
                        text_img.save(os.path.join(out_dir, f"{base_name}.png"))
                        text_img.save(os.path.join(pair_dir, f"text_seq{item.seq_idx}.png"))
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
