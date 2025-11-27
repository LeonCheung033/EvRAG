#!/bin/bash
# LLM微调训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# 项目根目录
PROJECT_ROOT="/remote-home/share/liangZhang/EvRAG"
LLAMAFACTORY_PATH="/remote-home/share/liangZhang/EVRAG/LLaMA-Factory-main"

# 配置文件路径
CONFIG_FILE="${PROJECT_ROOT}/config/finetune/qwen3_lora_sft.yaml"

# 进入LLaMA-Factory目录
cd "${LLAMAFACTORY_PATH}" || exit 1

# 执行训练
echo "开始LLM微调训练..."
echo "配置文件: ${CONFIG_FILE}"
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"

llamafactory-cli train "${CONFIG_FILE}"

echo "训练完成！"

