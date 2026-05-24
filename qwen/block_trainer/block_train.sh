#!/bin/bash

seed=${1:-42}
model_path=${2:-"../../../pretrained_models/Qwen2.5-VL-7B-Instruct"} 
# dataset_file=${3:-"/home/liuzonghao/dataset/ShareGPT4V/train_pt_files/sae_train_combined.json"}
dataset_file=${3:-"/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_train_short.json"}

image_folder=${4:-"/home/liuzonghao/AASAE/VL-SAE/CC3M/cc3m_jpg"}
save_path=${5:-"./checkpoints_sae"} # 修改为存模型权重的路径
target_layer=${6:-"model.language_model.layers.20"} 

# 可以增加控制在线流水线的特有参数
chunk_size=200
# METHOD_LIST=("filip" "asym" "sym")
METHOD_LIST=("asym")
for method in "${METHOD_LIST[@]}"; do
    if [ "$method" == "sym" ]; then
        current_topk=256
    else
        current_topk=512
    fi
    echo "=========================================================="
    echo "🚀 开始训练方法: ${method^^} (TopK: ${current_topk}, Chunk: ${chunk_size})"
    echo "=========================================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3 python orchestrator.py \
    --model-path "${model_path}" \
    --dataset-file "${dataset_file}" \
    --image-folder "${image_folder}" \
    --seed "${seed}" \
    --save_path "${save_path}" \
    --target_layer_name "${target_layer}" \
    --chunk_size ${chunk_size} \
    --topk ${current_topk} \
    --train_method ${method}
    2>&1 | tee "train_log_${method}.txt"
    echo "✅ 方法 ${method^^} 训练完成！"
done

