# extractor.py
import os
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from hooks import InputHook
from config import Config

class FeatureExtractor:
    def __init__(self):
        print(f"[Extractor] Loading VLM from {Config.model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            Config.model_path, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(Config.model_path)
        self.vision_start_token_id = self.model.config.vision_start_token_id
        self.vision_end_token_id = self.model.config.vision_end_token_id

    def extract_and_save_chunk(self, dataset_chunk, chunk_idx):
        chunk_data = [] # 改为保存包含成对字典的列表
        
        print(f"\n[Extractor] Processing Chunk {chunk_idx} ({len(dataset_chunk)} images)...")
        for line in tqdm(dataset_chunk):
            image_file = line["key"] + '.jpg'
            image_path = os.path.join(Config.image_folder, image_file)
            caption = line["caption"]

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": caption},
                ],
            }]

            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_prompt], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self.device)

            with InputHook(self.model, outputs=[Config.target_layer_name], as_tensor=True) as h:
                with torch.inference_mode():
                    _ = self.model.generate(**inputs, max_new_tokens=1, use_cache=True)
                
                returned_features = h.layer_outputs[Config.target_layer_name]
                if isinstance(returned_features, tuple):
                    returned_features = returned_features[0]

                input_ids_batch = inputs['input_ids'][0]
                attn_mask = inputs['attention_mask'][0]
                
                vision_start_indices = torch.where(input_ids_batch == self.vision_start_token_id)[0]
                vision_end_indices = torch.where(input_ids_batch == self.vision_end_token_id)[0]

                if len(vision_start_indices) > 0 and len(vision_end_indices) > 0:
                    img_st = vision_start_indices[0].item()
                    img_ed = vision_end_indices[0].item()
                    valid_len = attn_mask.sum().item()
                    
                    v_feat = returned_features[0, img_st+1:img_ed, :].detach().cpu()
                    
                    if valid_len > img_ed + 1:
                        t_feat = returned_features[0, img_ed+1:valid_len, :].detach().cpu()
                        
                        
                        chunk_data.append({
                            "vision": v_feat,
                            "text": t_feat 
                        })

        if chunk_data:
            save_path = f"{Config.temp_chunk_prefix}{chunk_idx}.pt"
            torch.save(chunk_data, save_path)
            print(f"[Extractor] Saved {len(chunk_data)} aligned pairs to {save_path}")
            return save_path
        else:
            return None