#!/bin/bash
# Reranker微调训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 项目根目录
PROJECT_ROOT="/remote-home/share/liangZhang/EvRAG"
RAG_RETRIEVAL_PATH="/remote-home/share/liangZhang/EVRAG/RAG-Retrieval"

# 配置文件路径
CONFIG_FILE="${PROJECT_ROOT}/config/finetune/reranker_training.yaml"
ACCELERATE_CONFIG="${RAG_RETRIEVAL_PATH}/rag_retrieval/train/reranker/../../../config/xlmroberta_default_config.yaml"

# 训练脚本目录
TRAIN_DIR="${RAG_RETRIEVAL_PATH}/rag_retrieval/train/reranker"

# 进入训练目录
cd "${TRAIN_DIR}" || exit 1

# 执行训练
echo "开始Reranker微调训练..."
echo "配置文件: ${CONFIG_FILE}"
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  train_reranker.py \
  --config "${CONFIG_FILE}"

echo "训练完成！"

