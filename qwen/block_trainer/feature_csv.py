import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from config import Config
    from hooks import InputHook
    from sae_model import SAE_V, SAE_D, VL_SAE
except ImportError:
    from block_trainer.config import Config
    from block_trainer.hooks import InputHook
    from block_trainer.sae_model import SAE_V, SAE_D, VL_SAE


def load_dataset(file_path: str) -> List[dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def deterministic_centers(num_views: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = int(math.ceil(math.sqrt(num_views)))
    xs = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
    ys = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return centers[:num_views]


def infer_sae_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    for key, value in state_dict.items():
        if key.endswith("W_enc") and value.ndim == 2:
            return value.shape[0], value.shape[1]
    sample_keys = list(state_dict.keys())[:8]
    raise ValueError(
        "Cannot infer SAE dims from state_dict. Expected a 2D 'W_enc' tensor. "
        f"Sample keys: {sample_keys}"
    )


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


def load_sae_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "sae_state_dict" in ckpt:
        return ckpt["sae_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported SAE checkpoint format")


def load_aux_proj(path: str, device: torch.device):
    from sae_model import TokenAuxProj
    proj = TokenAuxProj(Config.qwen_hidden_dim).to(device)
    proj.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    proj.eval()
    for p in proj.parameters():
        p.requires_grad = False
    return proj


def get_modality_tokens(
    v_proj: torch.Tensor,
    t_proj: torch.Tensor,
    v_mask: torch.Tensor,
    t_mask: torch.Tensor,
    train_method: str,
    grid_thw: Optional[torch.Tensor],
    num_views: int,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    if train_method == "sym":
        v_sum = (v_proj * v_mask.unsqueeze(-1)).sum(dim=1)
        v_tokens = v_sum / (v_mask.sum(dim=1, keepdim=True) + 1e-6)
        t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
        t_tokens = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
        return v_tokens.unsqueeze(1), t_tokens.unsqueeze(1), None

    if train_method == "filip":
        return v_proj, t_proj, None

    # asym
    if grid_thw is None:
        v_tokens = v_proj[:, :1, :]
        t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
        t_tokens = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)
        return v_tokens, t_tokens.unsqueeze(1), None

    B, Lv, D = v_proj.shape
    device = v_proj.device
    centers = deterministic_centers(num_views, device, v_proj.dtype).unsqueeze(0).expand(B, -1, -1)

    H = int(grid_thw[1].item())
    W = int(grid_thw[2].item())
    L = H * W
    L = min(L, Lv)

    y_coords = (torch.arange(H, device=device) + 0.5) / H
    x_coords = (torch.arange(W, device=device) + 0.5) / W
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)[:L]

    diff = centers.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    m = torch.exp(-gamma * dist_sq)

    v_tokens = torch.zeros(B, num_views, D, device=device, dtype=v_proj.dtype)
    for b in range(B):
        numerator = torch.mm(m[b], v_proj[b, :L, :])
        denominator = m[b].sum(dim=1, keepdim=True) + 1e-6
        v_tokens[b] = numerator / denominator

    t_sum = (t_proj * t_mask.unsqueeze(-1)).sum(dim=1)
    t_tokens = t_sum / (t_mask.sum(dim=1, keepdim=True) + 1e-6)

    centers_cpu = centers.detach().cpu().numpy()
    return v_tokens, t_tokens.unsqueeze(1), centers_cpu


def update_dim_dicts(
    dim_dicts: List[dict],
    dim_nonzero_count: List[int],
    acts: torch.Tensor,
    input_name: str,
    caption: str,
    label: int,
    row_id_start: int,
    sample_indices: List[int],
    modality: str,
    centers: Optional[np.ndarray],
) -> int:
    row_id = row_id_start
    for seq_i in sample_indices:
        vec = acts[seq_i].detach().cpu()
        nz = torch.nonzero(vec > 0, as_tuple=False).view(-1)
        for d_idx in nz.tolist():
            v = float(vec[d_idx])
            dim_nonzero_count[d_idx] += 1
            key = (modality, input_name)
            old = dim_dicts[d_idx].get(key)
            if old is None or v > old[0]:
                center_x = None
                center_y = None
                if centers is not None and seq_i < centers.shape[1]:
                    center_x = float(centers[0, seq_i, 0])
                    center_y = float(centers[0, seq_i, 1])
                dim_dicts[d_idx][key] = (v, row_id, seq_i, label, modality, caption, center_x, center_y)
        row_id += 1
    return row_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--target-layer-name", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--sae-type", choices=["SAE_V", "SAE_D", "VL_SAE"], default="VL_SAE")
    parser.add_argument("--train-method", choices=["filip", "asym", "sym"], default="filip")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--aux-proj-path", default="")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-per-seq", type=int, default=257)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-k", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--output-csv", required=True)
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

    dataset = load_dataset(args.dataset_file)
    if args.max_items > 0:
        dataset = dataset[: args.max_items]

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
    proj_device = next(aux_proj.parameters()).device
    proj_dtype = next(aux_proj.parameters()).dtype
    sae = sae.to(proj_device)

    vision_start_token_id = model.config.vision_start_token_id
    vision_end_token_id = model.config.vision_end_token_id

    dict_size = None
    for key, value in sae_state.items():
        if key.endswith("W_enc"):
            dict_size = value.shape[1]
            break
    if dict_size is None:
        raise ValueError("Cannot infer dict size")

    dim_dicts = [dict() for _ in range(dict_size)]
    dim_nonzero_count = [0] * dict_size
    global_row_id = 0

    for line in tqdm(dataset):
        image_file = line["key"] + ".jpg"
        image_path = os.path.join(args.image_folder, image_file)
        caption = line.get("caption", "")

        if not os.path.exists(image_path):
            continue

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

        with InputHook(model, outputs=[args.target_layer_name], as_tensor=True) as h:
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=1, use_cache=True)

            hidden = extract_tensor(h.layer_outputs.get(args.target_layer_name))
            if hidden is None:
                continue

        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0]
        vision_start_indices = torch.where(input_ids == vision_start_token_id)[0]
        vision_end_indices = torch.where(input_ids == vision_end_token_id)[0]

        if len(vision_start_indices) == 0 or len(vision_end_indices) == 0:
            continue

        img_st = vision_start_indices[0].item()
        img_ed = vision_end_indices[0].item()
        valid_len = attn_mask.sum().item()

        v_feat = hidden[0, img_st + 1 : img_ed, :]
        if valid_len <= img_ed + 1:
            continue
        t_feat = hidden[0, img_ed + 1 : valid_len, :]

        v_feat = v_feat.to(proj_device, dtype=proj_dtype)
        t_feat = t_feat.to(proj_device, dtype=proj_dtype)

        v_mask = torch.ones(1, v_feat.shape[0], device=proj_device, dtype=torch.bool)
        t_mask = torch.ones(1, t_feat.shape[0], device=proj_device, dtype=torch.bool)

        v_proj, t_proj = aux_proj(v_feat.unsqueeze(0), t_feat.unsqueeze(0))

        v_tokens, t_tokens, centers = get_modality_tokens(
            v_proj,
            t_proj,
            v_mask,
            t_mask,
            args.train_method,
            grid_thw,
            Config.num_views,
            Config.gamma,
        )

        # Vision
        if v_tokens.numel() > 0:
            acts_v = (
                sae.v_core.encode(v_tokens)
                if args.sae_type in ("SAE_D", "VL_SAE")
                else sae.core.encode(v_tokens)
            )
            acts_v = acts_v.squeeze(0)
            v_len = acts_v.shape[0]
            num_samples = min(args.sample_per_seq, v_len)
            sampled = random.sample(range(v_len), k=num_samples)
            global_row_id = update_dim_dicts(
                dim_dicts,
                dim_nonzero_count,
                acts_v,
                image_path,
                caption,
                -1,
                global_row_id,
                sampled,
                "vision",
                centers,
            )

        # Text
        if t_tokens.numel() > 0:
            acts_t = (
                sae.t_core.encode(t_tokens)
                if args.sae_type in ("SAE_D", "VL_SAE")
                else sae.core.encode(t_tokens)
            )
            acts_t = acts_t.squeeze(0)
            t_len = acts_t.shape[0]
            num_samples = min(args.sample_per_seq, t_len)
            sampled = random.sample(range(t_len), k=num_samples)
            global_row_id = update_dim_dicts(
                dim_dicts,
                dim_nonzero_count,
                acts_t,
                image_path,
                caption,
                -1,
                global_row_id,
                sampled,
                "text",
                None,
            )

    total_rows = max(global_row_id, 1)
    results = []
    for d_idx in range(dict_size):
        mapping = dim_dicts[d_idx]
        if not mapping:
            continue
        ratio_nz = dim_nonzero_count[d_idx] / total_rows

        items = sorted(
            [
                (v, rid, name, sidx, lbl, modality, caption, cx, cy)
                for (modality, name), (v, rid, sidx, lbl, modality, caption, cx, cy) in mapping.items()
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        max_val = items[0][0] or 1e-8
        step = max_val / 10

        buckets = [[] for _ in range(10)]
        for tpl in items:
            seg = int((max_val - tpl[0]) // step)
            seg = min(seg, 9)
            if len(buckets[seg]) < args.top_k:
                buckets[seg].append(tpl)

        for seg, bucket in enumerate(buckets):
            if not bucket:
                continue
            hi = max_val - seg * step
            lo = hi - step
            grp_name = f"bucket_{seg + 1:02d}"
            interval = f"[{lo:.3g}, {hi:.3g})"
            for rank, (v, rid, name, sidx, lbl, modality, caption, cx, cy) in enumerate(bucket):
                results.append(
                    dict(
                        dim=d_idx,
                        group=grp_name,
                        interval=interval,
                        rank=rank,
                        value=v,
                        row_id=rid,
                        input_name=name,
                        seq_idx=sidx,
                        input_label=lbl,
                        ratio_nonzero=ratio_nz,
                        model_name=args.model_path,
                        dataset_name=args.dataset_file,
                        pretrained_ckpt=args.sae_checkpoint,
                        modality=modality,
                        caption=caption,
                        view_center_x=cx,
                        view_center_y=cy,
                    )
                )

        if args.random_k > 0:
            rnd = random.sample(items, k=min(args.random_k, len(items)))
            for rank, (v, rid, name, sidx, lbl, modality, caption, cx, cy) in enumerate(rnd):
                results.append(
                    dict(
                        dim=d_idx,
                        group="bucket_random",
                        interval="NA",
                        rank=rank,
                        value=v,
                        row_id=rid,
                        input_name=name,
                        seq_idx=sidx,
                        input_label=lbl,
                        ratio_nonzero=ratio_nz,
                        model_name=args.model_path,
                        dataset_name=args.dataset_file,
                        pretrained_ckpt=args.sae_checkpoint,
                        modality=modality,
                        caption=caption,
                        view_center_x=cx,
                        view_center_y=cy,
                    )
                )

    import csv

    cols = [
        "dim",
        "group",
        "interval",
        "rank",
        "value",
        "row_id",
        "input_name",
        "seq_idx",
        "input_label",
        "ratio_nonzero",
        "model_name",
        "dataset_name",
        "pretrained_ckpt",
        "modality",
        "caption",
        "view_center_x",
        "view_center_y",
    ]
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)

    print(f"[feature_csv] Done -> {args.output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
