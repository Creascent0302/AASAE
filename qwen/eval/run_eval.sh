#!/bin/bash

# ==============================================================================
# 多模态 SAE 评估流水线启动脚本
# ==============================================================================

# 默认设置：如果用户不指定，则依次运行所有方法
METHODS="sym,filip,asym"
# METHODS="sym, asym"
# 默认使用的 GPU 编号
CUDA_DEV="0,1,2,3"
ASYM_USE_VIEWS="${ASYM_USE_VIEWS:-0}"

# 帮助信息输出函数
usage() {
    echo "Usage: $0 [-m <methods>] [-d <gpu_ids>] [-h]"
    echo "Options:"
    echo "  -m    指定要运行的方法，用逗号分隔 (可选: sym, filip, asym). 默认: sym,filip,asym"
    echo "  -d    指定使用的 CUDA 设备编号. 默认: 0,1,2,3"
    echo "  -h    显示此帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0 -m asym                  # 仅运行 asym 方法评估"
    echo "  $0 -m filip,sym             # 依次运行 filip 和 sym 方法"
    echo "  $0 -m asym -d 0,1           # 仅运行 asym，且限制只使用 GPU 0 和 1"
    exit 1
}

# 解析命令行传入的参数
while getopts "m:d:h" opt; do
  case $opt in
    m) METHODS="$OPTARG" ;;
    d) CUDA_DEV="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option -$OPTARG" >&2; usage ;;
  esac
done

# 设置环境变量限制 GPU
export CUDA_VISIBLE_DEVICES=$CUDA_DEV

echo "=========================================================="
echo " 🚀 启动 SAE 评估流水线"
echo " 📌 选定方法: $METHODS"
echo " 💻 可用 GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# 运行 Python 评估脚本
# 注意：确保你的 python 脚本名是 evaluate_models.py
python evaluate_models.py \
    --methods "$METHODS" \
    --eval-batch-size 8 \
    --chunk-size 100 \
    --report-file "evaluation_results_all.txt" \
    --score-align 0.6 \
    --score-entail 0.4 \
  #   --topk-sym 256 \
  #   --topk-filip 64 \
  # --topk-asym 64 \

# 捕获退出状态码
if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo " 🎉 评估完成！请查看 evaluation_results_all.txt 获取报告。"
    echo "=========================================================="
else
    echo "=========================================================="
    echo " ❌ 评估过程中发生错误，请检查上方日志。"
    echo "=========================================================="
fi