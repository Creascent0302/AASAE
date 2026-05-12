不压缩的离线训练过程中提取出来的特征向量数据太庞大了，因此采取在线训练的思路进行多轮次的小批次离线训练，每次进行指定数量的数据推理，特征提取，之后进行训练，训练结束后删除提取特征，进行下一批次的特征提取和训练
注意每次提取和训练需要0,1,2,3四张卡，qwen占一张卡做推理，然后训三个SAE分别扔到三张卡上

---

后续可视化扩展建议（Qwen-VL 版本）
1) 复用 extractor 的 token 切分逻辑，将 vision tokens 与 image_grid_thw 映射到 2D 网格。
2) 参照 CaFE 的 CSV 生成：对每个 SAE feature，统计“每张图最大激活的 token”，并保存 input_name + seq_idx + value。
3) attribution 热图：固定 feature 维度 d，用 v_token 向量与 W_enc[:, d] 做打分，反向传播到输入或 vision token，得到热图。
4) 将 seq_idx 对应的 patch 位置叠加到原图，形成可解释样本集。

命令提示（新）
1) 生成 CSV（合并 vision/text）:
	 python feature_csv.py --model-path <qwen> --dataset-file <json> --image-folder <imgs> \
		 --target-layer-name <layer> --sae-checkpoint <ckpt> --sae-type VL_SAE \
		 --train-method filip --output-csv <out.csv> \
		 --aux-proj-path <shared_best_aux_proj.pth>  # 可选
2) 生成可视化（vision 或 text）:
	 python visualize.py --csv-path <out.csv> --model-path <qwen> --target-layer-name <layer> \
		 --sae-checkpoint <ckpt> --sae-type VL_SAE --train-method filip \
		 --modality vision --output-dir <out_dir> \
		 --aux-proj-path <shared_best_aux_proj.pth>  # 可选