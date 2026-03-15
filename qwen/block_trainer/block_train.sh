#!/bin/bash

seed=${1:-42}
model_path=${2:-"../../../pretrained_models/Qwen2.5-VL-7B-Instruct"} 
dataset_file=${3:-"../../VL-SAE/CC3M/merged_cc3m.json"}
image_folder=${4:-"../../VL-SAE/CC3M/cc3m_jpg"}
save_path=${5:-"./checkpoints_sae"} # 修改为存模型权重的路径
target_layer=${6:-"model.language_model.layers.20"} 

# 可以增加控制在线流水线的特有参数
chunk_size=200
topk=64
METHOD="asym" # 可选 'filip' 或 'asym'

CUDA_VISIBLE_DEVICES=0,1,2,3 python orchestrator.py \
--model-path "${model_path}" \
--dataset-file "${dataset_file}" \
--image-folder "${image_folder}" \
--seed "${seed}" \
--save_path "${save_path}" \
--target_layer_name "${target_layer}" \
--chunk_size ${chunk_size} \
--topk ${topk} \
--train_method ${METHOD}