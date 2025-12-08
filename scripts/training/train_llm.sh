#!/bin/bash
# LLM微调训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# LLaMA-Factory路径（相对于项目根目录）
LLAMAFACTORY_PATH="${PROJECT_ROOT}/LLaMA-Factory-main"

# 如果LLaMA-Factory不在项目根目录，可以手动指定
# LLAMAFACTORY_PATH="/path/to/your/LLaMA-Factory-main"

# 配置文件路径
CONFIG_FILE="${PROJECT_ROOT}/config/finetune/qwen3_lora_sft.yaml"

# 检查LLaMA-Factory是否存在
if [ ! -d "${LLAMAFACTORY_PATH}" ]; then
    echo "错误: LLaMA-Factory目录不存在: ${LLAMAFACTORY_PATH}"
    echo "请先克隆LLaMA-Factory: git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory-main"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 进入LLaMA-Factory目录
cd "${LLAMAFACTORY_PATH}" || exit 1

# 执行训练
echo "=========================================="
echo "开始LLM微调训练..."
echo "=========================================="
echo "项目根目录: ${PROJECT_ROOT}"
echo "LLaMA-Factory路径: ${LLAMAFACTORY_PATH}"
echo "配置文件: ${CONFIG_FILE}"
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

llamafactory-cli train "${CONFIG_FILE}"

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练完成！"
    echo "模型保存在: ${PROJECT_ROOT}/models/finetuned/qwen3_lora_sft/"
    echo "=========================================="
else
    echo "=========================================="
    echo "训练失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

