#!/bin/bash

# vLLM服务启动脚本
# 用于启动本地vLLM服务，提供LLM推理能力

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 加载配置文件
CONFIG_FILE="$PROJECT_ROOT/config/config.yaml"

# 从配置文件读取参数（使用Python解析）
get_config_value() {
    python3 -c "
import yaml
import sys
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('$1', '$2'))
"
}

# 读取配置
MODEL_PATH="${MODEL_PATH:-$(get_config_value 'models_dir' 'models')/Qwen3-8B}"
MODEL_NAME="${MODEL_NAME:-$(get_config_value 'local_llm_model_name' 'Qwen3-8B')}"
VLLM_PORT="${VLLM_PORT:-$(get_config_value 'server_port' '8000')}"
VLLM_GPU_IDS="${VLLM_GPU_IDS:-$(get_config_value 'vllm_gpu_ids' '[3,5]')}"

# 解析GPU IDs（从 [3,5] 格式转换为 3,5）
GPU_IDS=$(echo "$VLLM_GPU_IDS" | sed 's/\[//g' | sed 's/\]//g' | sed 's/,//g' | tr ' ' ',')
TENSOR_PARALLEL_SIZE=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# 日志目录
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}启动 vLLM 服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "项目根目录: ${GREEN}$PROJECT_ROOT${NC}"
echo -e "模型路径: ${GREEN}$MODEL_PATH${NC}"
echo -e "模型名称: ${GREEN}$MODEL_NAME${NC}"
echo -e "服务端口: ${GREEN}$VLLM_PORT${NC}"
echo -e "GPU设备: ${GREEN}$GPU_IDS${NC}"
echo -e "Tensor并行度: ${GREEN}$TENSOR_PARALLEL_SIZE${NC}"
echo -e "日志文件: ${GREEN}$VLLM_LOG${NC}"
echo ""

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请检查配置文件中的 models_dir 和模型名称${NC}"
    exit 1
fi

# 检查vLLM是否已安装
if ! command -v vllm &> /dev/null; then
    echo -e "${YELLOW}警告: vLLM 未安装，尝试安装...${NC}"
    pip install vllm
fi

# 检查端口是否被占用
if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}警告: 端口 $VLLM_PORT 已被占用${NC}"
    echo -e "${YELLOW}是否要停止现有服务？(y/n)${NC}"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        echo -e "${YELLOW}正在停止占用端口 $VLLM_PORT 的进程...${NC}"
        lsof -ti:$VLLM_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        echo -e "${RED}退出启动${NC}"
        exit 1
    fi
fi

# 启动vLLM服务
echo -e "${GREEN}正在启动 vLLM 服务...${NC}"
echo -e "${YELLOW}这可能需要几分钟时间加载模型...${NC}"
echo ""

# 使用CUDA_VISIBLE_DEVICES指定GPU，tensor-parallel-size指定并行度
# --served-model-name 指定API中使用的模型名称（简化名称）
# 注意：不添加 --enable-reasoning 参数，禁用思考模式（数据生成任务不需要思考过程）
CUDA_VISIBLE_DEVICES=$GPU_IDS nohup vllm serve "$MODEL_PATH" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port $VLLM_PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --served-model-name "$MODEL_NAME" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > "$PROJECT_ROOT/log/vllm.pid"

echo -e "${GREEN}vLLM 服务已启动（PID: $VLLM_PID）${NC}"
echo -e "日志文件: ${GREEN}$VLLM_LOG${NC}"
echo ""
echo -e "${YELLOW}等待服务就绪...${NC}"

# 等待服务启动（最多等待5分钟）
MAX_WAIT=300
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ vLLM 服务已就绪！${NC}"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo -e "${YELLOW}等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)${NC}"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo -e "${RED}警告: 服务启动超时，请检查日志: $VLLM_LOG${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}vLLM 服务启动成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "服务地址: ${GREEN}http://localhost:$VLLM_PORT/v1${NC}"
echo -e "健康检查: ${GREEN}http://localhost:$VLLM_PORT/health${NC}"
echo -e "API文档: ${GREEN}http://localhost:$VLLM_PORT/docs${NC}"
echo -e "进程ID: ${GREEN}$VLLM_PID${NC}"
echo -e "日志文件: ${GREEN}$VLLM_LOG${NC}"
echo ""
echo -e "${YELLOW}提示:${NC}"
echo -e "  - 查看日志: tail -f $VLLM_LOG"
echo -e "  - 停止服务: ./scripts/stop_vllm.sh 或 kill $VLLM_PID"
echo -e "  - 测试服务: curl http://localhost:$VLLM_PORT/health"
echo ""

