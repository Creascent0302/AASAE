import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ultralytics import SAM
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
QWEN_MODEL_PATH = "../../../../pretrained_models/Qwen2.5-VL-7B-Instruct"
SAM_MODEL_PATH = "../utils/sam2.1_b.pt" 
IMAGE_PATH = "/home/liuzonghao/dataset/Rellis-3D/00000/pylon_camera_node/frame002591-1581624911_849.jpg"


def load_models():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    
    print("正在加载 SAM 2...")
    sam_model = SAM(SAM_MODEL_PATH)
    return model, processor, sam_model

def get_road_bbox_from_qwen(model, processor, image_path):
    """
    使用 Qwen2.5-VL 识别道路并返回边界框
    """
    image = Image.open(image_path)
    w, h = image.size

    prompt_text = (
        "You need to detect the drivable off-road path. "
        f"The size of the image is {w}(width)*{h}(height). "
        "Return the bounding box of the main path in JSON format: "
        "{\"box_2d\": [x_min, y_min, x_max, y_max]}. "
        "The coordinates must be integers within the size of the picture. "
        "Do not output any explanation, only the JSON."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    print(f"Qwen 原始输出: {output_text}")
    
    bboxes = []
    try:
        # 即使模型输出了一些废话，我们尝试找 {} 里的内容
        import re
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            coords = data.get("box_2d") or data.get("bbox")
            
            if coords and len(coords) == 4:
                bboxes.append(coords)
                print(f"解析成功坐标: {coords}")
            else:
                raise ValueError("JSON中未找到有效坐标")
        else:
            # 备用：如果不是 JSON，尝试直接正则匹配 4 个连续数字
            numbers = re.findall(r"\d+", output_text)
            if len(numbers) >= 4:
                # 取前4个看起来像坐标的数字
                ymin, xmin, ymax, xmax = map(int, numbers[:4])
                box = [xmin/1000*w, ymin/1000*h, xmax/1000*w, ymax/1000*h]
                bboxes.append(box)
                print(f"正则暴力解析坐标: {box}")
            else:
                raise ValueError("未找到数字")

    except Exception as e:
        print(f"解析出错 ({e})，使用 Fallback 区域...")
        # Fallback: 假设路在图片下半部分中央
        bboxes.append([w*0.2, h*0.5, w*0.8, h*1.0])
        
    return bboxes

def segment_road_with_sam(sam_model, image_path, bboxes):
    """
    将 Qwen 的 BBox 作为提示输入给 SAM
    """
    results = sam_model.predict(image_path, bboxes=bboxes)
    
    return results[0]

def visualize_results(image_path, result):
    """
    叠加显示结果并保存为文件
    """
    img = cv2.imread(image_path)
    
    if result.masks is not None:
        # 合并所有检测到的 mask
        mask = result.masks.data.cpu().numpy()
        combined_mask = np.any(mask, axis=0).astype(np.uint8) * 255
        
        # 创建绿色透明层
        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 1] = 255
        
        # 叠加
        alpha = 0.5
        mask_indices = combined_mask > 0
        img[mask_indices] = cv2.addWeighted(img[mask_indices], 1-alpha, colored_mask[mask_indices], alpha, 0)
        
        # 画 Box
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    save_path = "../results/result_output.jpg"
    cv2.imwrite(save_path, img)
    print(f"\n✅ 识别完成！可视化结果已保存为: {save_path}")
    print("请查看当前目录下的 result_output.jpg 文件")

def main():
    qwen_model, qwen_processor, sam_model = load_models()
    
    # 2. Qwen 粗定位 (语义理解)
    print("Step 1: Analyzing scene with Qwen2.5-VL...")
    bboxes = get_road_bbox_from_qwen(qwen_model, qwen_processor, IMAGE_PATH)
    print(f"Detected Road BBox: {bboxes}")
    
    # 3. SAM 精细分割 (像素级操作)
    print("Step 2: Segmenting with SAM 2...")
    result = segment_road_with_sam(sam_model, IMAGE_PATH, bboxes)
    
    # 4. 展示
    visualize_results(IMAGE_PATH, result)

if __name__ == "__main__":
    main()
