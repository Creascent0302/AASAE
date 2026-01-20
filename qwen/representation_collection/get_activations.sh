seed=${1:-42}
model_path=${2:-"../../../pretrained_models/Qwen2.5-VL-7B-Instruct"} 
dataset_file=${3:-"../../VL-SAE/CC3M/merged_cc3m.json"}
image_folder=${4:-"../../VL-SAE/CC3M/cc3m_jpg"}
save_path=${5:-"./activations_qwen"}
# Qwen 的层命名通常为 "model.layers.X"，且 7B 模型通常只有 28 层 (0-27)
# 建议提取第 20 层左右，或者根据你的 SAE 需求调整
target_layer=${6:-"model.language_model.layers.20"} 

CUDA_VISIBLE_DEVICES=0 python activation_collector.py \
--model-path ${model_path} \
--dataset-file ${dataset_file} \
--image-folder ${image_folder} \
--seed ${seed} \
--save_path ${save_path} \
--target_layer_name ${target_layer}