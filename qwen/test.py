from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

# 1. 创建一个“迷你”配置，而不是默认的巨型配置
config = Qwen2_5_VLConfig(
    hidden_size=64,        # 默认是 3584 -> 改成 64
    intermediate_size=128, # 默认是 18944 -> 改成 128
    num_hidden_layers=2,   # 默认是 28 -> 改成 2层
    num_attention_heads=4, # 默认是 28 -> 改成 4头
    vocab_size=1000        # 随便给个小词表
)

print("正在初始化迷你模型...")
model = Qwen2_5_VLForConditionalGeneration(config)
print("初始化成功！")
print(model.config)