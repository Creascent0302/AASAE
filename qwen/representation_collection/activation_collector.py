import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
from PIL import Image

# Qwen2.5-VL specific imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from transformers import set_seed

# Reuse your existing hooks
from hooks import InputHook

def activation_allocation(args):
    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    vision_start_token_id = model.config.vision_start_token_id
    vision_end_token_id = model.config.vision_end_token_id

    print(f"Vision Tokens - Start ID: {vision_start_token_id}, End ID: {vision_end_token_id}")

    with open(args.dataset_file, "r") as f:
        datasets = json.load(f)

    feature_dict = {}
    image_features_list = []
    text_features_list = []
    image_files = []
    texts = []

    target_layer_name = args.target_layer_name

    print(f"Targeting layer: {target_layer_name}")

    for idx, line in tqdm(enumerate(datasets), desc="Inference..."):
        image_file = line["key"] + '.jpg'
        image_path = os.path.join(args.image_folder, image_file)
        caption = line["caption"]

        # 2. Construct Prompt using Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": caption},
                ],
            }
        ]

        # 准备输入
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # print(inputs.keys())
        # print(inputs['input_ids'].shape)
        # return
        # 3. Hook & Inference
        # InputHook 捕获的是该层输入的 hidden_states
        with InputHook(model, outputs=[target_layer_name], as_tensor=False) as h:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=1, # 我们只需要获取 prompt 处理阶段的 activation
                    use_cache=True
                )
            
            # 4. Feature Extraction & Slicing
            # h.layer_outputs[target_layer_name] 的 shape 通常是 [batch, seq_len, hidden_dim]
            raw_output = h.layer_outputs[target_layer_name]
            returned_features = None
            if isinstance(raw_output, tuple):
                returned_features = raw_output[0]  # 有些层的 hook 返回是 tuple，第一个元素是我们需要的 tensor
            else:
                returned_features = raw_output

            # 对于 InputHook，捕获的是输入到该层的 hidden_states。
            # Qwen2.5-VL 在进入 LLM 层之前，Input IDs 已经被 Embeddings + Vision Projector 替换扩展了。
            
            # 我们需要找到 input_ids 中 vision_start 和 vision_end 的位置
            # 注意：model.generate 可能会修改 input_ids (比如去除 padding)，
            # 但在这里 InputHook 捕获的是 forward 过程中的 tensor。
            
            # 关键点：我们需要知道此时输入这一层的 input_ids 是什么。
            # 由于 hook 只能拿到 hidden_states，我们需要通过输入的 input_ids 来定位。
            # 这里的 input_ids 是 processor 返回的，在模型内部 forward 时，
            # 非图像 token 的位置保持不变，图像 token 被扩展。
            
            # 更稳健的方法：利用 inputs['input_ids'] 里的特殊 token 定位
            # Qwen2-VL 的处理逻辑： input_ids 里已经是扩展后的 token id 序列了 (包含 vision_start/end)
            input_ids_batch = inputs['input_ids'][0] # batch size = 1
            
            # 寻找 vision start 和 end 的索引
            # torch.where 返回的是 tuple, 取 [0]
            vision_start_indices = torch.where(input_ids_batch == vision_start_token_id)[0]
            vision_end_indices = torch.where(input_ids_batch == vision_end_token_id)[0]

            if len(vision_start_indices) > 0 and len(vision_end_indices) > 0:
                img_st = vision_start_indices[0].item()
                img_ed = vision_end_indices[0].item()
                image_features = returned_features[0, img_st+1:img_ed, :].mean(0)
                # Text Features
                text_features = returned_features[0, img_ed+1:, :].mean(0)
            else:
                print(f"Warning: No vision tokens found in {image_file}")
                image_features = torch.zeros(returned_features.shape[-1]).to(returned_features.device)
                text_features = returned_features[0, :, :].mean(0)
        image_features_list.append( image_features.detach().cpu().float().numpy())
        text_features_list.append(text_features.detach().cpu().float().numpy())
        image_files.append(image_file)
        texts.append(caption)
        
    feature_dict = {
        'image_features': image_features_list,
        'text_features': text_features_list,
        'image_file': image_files,
        'text': texts,
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    # 修改文件名以区分模型
    save_filename = f"qwen2_5_vl_cc3m_{target_layer_name.split('.')[-1]}_embeddings.pt"
    save_path = os.path.join(args.save_path, save_filename)
    torch.save(feature_dict, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径修改为 Qwen 模型路径
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--dataset-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    # Qwen-7B 通常有 28 层 (layers.0 到 layers.27)，所以默认改为 layers.20 以防报错
    parser.add_argument("--target_layer_name", type=str, default="model.layers.20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--save_path', default="./activations", type=str)

    args = parser.parse_args()
    set_seed(args.seed)
    activation_allocation(args)