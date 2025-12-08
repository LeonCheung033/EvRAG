#!/bin/bash
# Reranker微调训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# RAG-Retrieval路径（相对于项目根目录）
RAG_RETRIEVAL_PATH="${PROJECT_ROOT}/RAG-Retrieval"

# 如果RAG-Retrieval不在项目根目录，可以手动指定
# RAG_RETRIEVAL_PATH="/path/to/your/RAG-Retrieval"

# 配置文件路径
CONFIG_FILE="${PROJECT_ROOT}/config/finetune/reranker_training.yaml"
ACCELERATE_CONFIG="${RAG_RETRIEVAL_PATH}/rag_retrieval/train/reranker/../../../config/xlmroberta_default_config.yaml"

# 训练脚本目录
TRAIN_DIR="${RAG_RETRIEVAL_PATH}/rag_retrieval/train/reranker"

# 检查RAG-Retrieval是否存在
if [ ! -d "${RAG_RETRIEVAL_PATH}" ]; then
    echo "错误: RAG-Retrieval目录不存在: ${RAG_RETRIEVAL_PATH}"
    echo "请先克隆RAG-Retrieval: git clone https://github.com/NovaSearch-Team/RAG-Retrieval.git RAG-Retrieval"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 检查训练目录是否存在
if [ ! -d "${TRAIN_DIR}" ]; then
    echo "错误: 训练目录不存在: ${TRAIN_DIR}"
    exit 1
fi

# 进入训练目录
cd "${TRAIN_DIR}" || exit 1

# 执行训练
echo "=========================================="
echo "开始Reranker微调训练..."
echo "=========================================="
echo "项目根目录: ${PROJECT_ROOT}"
echo "RAG-Retrieval路径: ${RAG_RETRIEVAL_PATH}"
echo "配置文件: ${CONFIG_FILE}"
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  train_reranker.py \
  --config "${CONFIG_FILE}"

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练完成！"
    echo "模型保存在: ${PROJECT_ROOT}/models/finetuned/bge_reranker/"
    echo "=========================================="
else
    echo "=========================================="
    echo "训练失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

