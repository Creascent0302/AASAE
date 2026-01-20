# 1. 训练对齐层
# python train_aux.py \
#     --embeddings_path "../representation_collection/activations_qwen/qwen2_5_vl_cc3m_20_embeddings.pt" \
#     --save_path "./checkpoints_qwen" \
#     --batch_size 2048 \
#     --num_epochs 50

# 2. 训练 SAE
python train.py \
    --embeddings_path "../representation_collection/activations_qwen/qwen2_5_vl_cc3m_20_embeddings.pt" \
    --aux_ae_path "./checkpoints_qwen/qwen_aux_best.pt" \
    --save_path "./checkpoints_qwen" \
    --topk 64 \
    --hidden_ratio 8 \
    --batch_size 512