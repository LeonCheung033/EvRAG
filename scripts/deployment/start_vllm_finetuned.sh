#!/bin/bash

# vLLM服务启动脚本（仅启动微调模型）
# 用于RAG服务，微调模型使用8001端口

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 默认配置
FINETUNED_MODEL_PATH="${FINETUNED_MODEL_PATH:-$PROJECT_ROOT/models/finetuned/qwen3_lora_sft}"
FINETUNED_BASE_MODEL_PATH="${FINETUNED_BASE_MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-8B}"

FINETUNED_PORT="${FINETUNED_PORT:-8001}"
FINETUNED_GPU_IDS="${FINETUNED_GPU_IDS:-1}"
FINETUNED_MODEL_NAME="${FINETUNED_MODEL_NAME:-qwen3_lora_sft}"

# 并发度配置（可根据显存情况调整）
FINETUNED_MAX_NUM_SEQS="${FINETUNED_MAX_NUM_SEQS:-32}"

# vLLM显存优化配置
FINETUNED_GPU_MEMORY_UTIL="${FINETUNED_GPU_MEMORY_UTIL:-0.80}"
FINETUNED_MAX_MODEL_LEN="${FINETUNED_MAX_MODEL_LEN:-6144}"
FINETUNED_SWAP_SPACE="${FINETUNED_SWAP_SPACE:-4}"

# 日志目录
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"
FINETUNED_LOG="$LOG_DIR/vllm_finetuned.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}启动 vLLM 微调模型服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查vLLM是否已安装
if ! command -v vllm &> /dev/null; then
    echo -e "${RED}错误: vLLM 未安装${NC}"
    echo -e "${YELLOW}请运行: pip install vllm${NC}"
    exit 1
fi

# 检查模型路径
if [ ! -d "$FINETUNED_BASE_MODEL_PATH" ]; then
    echo -e "${RED}错误: 基础模型路径不存在: $FINETUNED_BASE_MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$FINETUNED_MODEL_PATH" ]; then
    echo -e "${RED}错误: 微调模型路径不存在: $FINETUNED_MODEL_PATH${NC}"
    exit 1
fi

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}警告: 端口 $port 已被占用，正在停止...${NC}"
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

check_port $FINETUNED_PORT

# 计算Tensor并行度
FINETUNED_TP_SIZE=$(echo "$FINETUNED_GPU_IDS" | tr ',' '\n' | wc -l)

# 启动微调模型服务（使用LoRA）
echo -e "${GREEN}启动微调模型服务（LoRA）...${NC}"
echo -e "  基础模型路径: ${FINETUNED_BASE_MODEL_PATH}"
echo -e "  LoRA适配器路径: ${FINETUNED_MODEL_PATH}"
echo -e "  端口: ${FINETUNED_PORT}"
echo -e "  GPU: ${FINETUNED_GPU_IDS}"
echo -e "  模型名称: ${FINETUNED_MODEL_NAME}"
echo -e "  最大并发序列数: ${FINETUNED_MAX_NUM_SEQS}"
echo -e "  GPU显存利用率: ${FINETUNED_GPU_MEMORY_UTIL}"
echo -e "  最大模型长度: ${FINETUNED_MAX_MODEL_LEN}"
echo -e "  CPU交换空间: ${FINETUNED_SWAP_SPACE}GiB"
echo ""

# vLLM支持通过--enable-lora参数启用LoRA，并通过--lora-modules指定适配器
CUDA_VISIBLE_DEVICES=$FINETUNED_GPU_IDS nohup vllm serve "$FINETUNED_BASE_MODEL_PATH" \
    --tensor-parallel-size $FINETUNED_TP_SIZE \
    --port $FINETUNED_PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --served-model-name "$FINETUNED_MODEL_NAME" \
    --enable-lora \
    --lora-modules "$FINETUNED_MODEL_NAME=$FINETUNED_MODEL_PATH" \
    --max-num-seqs $FINETUNED_MAX_NUM_SEQS \
    --gpu-memory-utilization $FINETUNED_GPU_MEMORY_UTIL \
    --max-model-len $FINETUNED_MAX_MODEL_LEN \
    --swap-space $FINETUNED_SWAP_SPACE \
    > "$FINETUNED_LOG" 2>&1 &

FINETUNED_PID=$!
echo $FINETUNED_PID > "$LOG_DIR/vllm_finetuned.pid"
echo -e "${GREEN}✓ 微调模型服务已启动（PID: $FINETUNED_PID）${NC}"

# 等待微调服务就绪
echo -e "${YELLOW}等待微调服务就绪...${NC}"
MAX_WAIT=300
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$FINETUNED_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 微调服务已就绪！${NC}"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo -e "${YELLOW}等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)${NC}"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo -e "${RED}错误: 微调服务启动超时${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}微调模型服务启动成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}微调模型服务:${NC}"
echo -e "  地址: http://localhost:$FINETUNED_PORT/v1"
echo -e "  模型名称: $FINETUNED_MODEL_NAME"
echo -e "  进程ID: $FINETUNED_PID"
echo -e "  日志: $FINETUNED_LOG"
echo ""
echo -e "${YELLOW}提示:${NC}"
echo -e "  - 停止服务: ./scripts/stop_vllm_finetuned.sh"
echo -e "  - 查看日志: tail -f $FINETUNED_LOG"
echo ""

