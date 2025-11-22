#!/bin/bash
# EvRAG 模型下载脚本（使用HuggingFace，直接下载到指定目录）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置模型下载目录（不使用默认cache）
MODELS_DIR="$PROJECT_ROOT/models"

# 激活conda环境
if command -v conda &> /dev/null; then
    if conda env list | grep -q "evrag"; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate evrag
    fi
fi

# 设置环境变量（可选，如果不想使用默认cache）
export HF_HOME="$MODELS_DIR/.hf_cache"  # HuggingFace的元数据cache（很小）
export TRANSFORMERS_CACHE="$MODELS_DIR/.transformers_cache"  # Transformers的cache（很小）

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EvRAG 模型下载脚本（HuggingFace）${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "模型下载目录: ${GREEN}$MODELS_DIR${NC}"

# 创建模型目录
mkdir -p "$MODELS_DIR"

# 检查是否安装了huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}安装huggingface_hub...${NC}"
    pip install huggingface_hub
fi

# 下载函数
download_model() {
    local repo_id=$1
    local local_dir=$2
    local model_name=$3
    
    echo -e "\n${GREEN}下载模型: $model_name${NC}"
    echo -e "Repository: $repo_id"
    echo -e "保存到: $local_dir"
    
    python << EOF
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

repo_id = "$repo_id"
local_dir = "$local_dir"
model_name = "$model_name"

print(f"开始下载 {model_name}...")
try:
    # 使用local_dir参数直接下载到指定目录，不使用cache
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
        resume_download=True,  # 支持断点续传
    )
    print(f"✓ {model_name} 下载完成: {downloaded_path}")
except Exception as e:
    print(f"✗ {model_name} 下载失败: {e}")
    sys.exit(1)
EOF
}

# 下载必需的模型
echo -e "\n${GREEN}开始下载模型...${NC}"

# 1. M3E-small（语义切分服务需要）
download_model \
    "moka-ai/m3e-small" \
    "$MODELS_DIR/m3e-small" \
    "M3E-small"

# 2. BGE-M3（Milvus检索器需要）
download_model \
    "BAAI/bge-m3" \
    "$MODELS_DIR/bge-m3" \
    "BGE-M3"

# 3. BGE-Reranker-v2-m3（重排序器需要）
download_model \
    "BAAI/bge-reranker-v2-m3" \
    "$MODELS_DIR/bge-reranker-v2-m3" \
    "BGE-Reranker-v2-m3"

# 4. Qwen3-8B（LLM模型，用于微调和推理）
download_model \
    "Qwen/Qwen3-8B" \
    "$MODELS_DIR/Qwen3-8B" \
    "Qwen3-8B"

# # 5. Qwen3-Embedding-0.6B（可选）
# echo -e "\n${YELLOW}是否下载 Qwen3-Embedding-0.6B? (y/n)${NC}"
# read -r answer
# if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
#     download_model \
#         "Qwen/Qwen3-Embedding-0.6B" \
#         "$MODELS_DIR/Qwen3-Embedding-0.6B" \
#         "Qwen3-Embedding-0.6B"
# fi

# # 6. Qwen3-Reranker-4B（可选）
# echo -e "\n${YELLOW}是否下载 Qwen3-Reranker-4B? (y/n)${NC}"
# read -r answer
# if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
#     download_model \
#         "Qwen/Qwen3-Reranker-4B" \
#         "$MODELS_DIR/Qwen3-Reranker-4B" \
#         "Qwen3-Reranker-4B"
# fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}模型下载完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "所有模型已下载到: ${GREEN}$MODELS_DIR${NC}"
echo -e "\n模型目录结构:"
ls -lh "$MODELS_DIR" | grep -E "^d" | awk '{print "  " $9}'