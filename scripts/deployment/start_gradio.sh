#!/bin/bash

# Gradio前端服务启动脚本

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
GRADIO_PORT="${GRADIO_PORT:-8080}"
# RAG服务使用8002端口（8001已被vLLM微调模型占用）
RAG_SERVER_URL="${RAG_SERVER_URL:-http://localhost:8002}"

# 日志目录
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"
GRADIO_LOG="$LOG_DIR/gradio.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}启动 Gradio 前端服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查RAG服务是否运行
echo -e "${YELLOW}检查RAG服务状态...${NC}"
if ! curl -s "$RAG_SERVER_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}警告: RAG服务未运行 ($RAG_SERVER_URL)${NC}"
    echo -e "${YELLOW}请先启动RAG服务:${NC}"
    echo -e "  python -m src.evrag.server.rag_server"
    echo -e "  或使用 uvicorn:"
    echo -e "  uvicorn src.evrag.server.rag_server:app --host 0.0.0.0 --port 8001"
    echo ""
    read -p "是否继续启动Gradio? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ RAG服务运行正常${NC}"
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

check_port $GRADIO_PORT

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 启动Gradio服务
echo -e "${GREEN}启动Gradio服务...${NC}"
echo -e "  端口: ${GRADIO_PORT}"
echo -e "  RAG服务地址: ${RAG_SERVER_URL}"
echo -e "  日志: ${GRADIO_LOG}"
echo ""

# 设置环境变量
export RAG_SERVER_URL="$RAG_SERVER_URL"

# 启动Gradio（后台运行）
nohup python -m src.evrag.server.gradio_app > "$GRADIO_LOG" 2>&1 &
GRADIO_PID=$!
echo $GRADIO_PID > "$LOG_DIR/gradio.pid"

echo -e "${GREEN}✓ Gradio服务已启动（PID: $GRADIO_PID）${NC}"

# 等待服务就绪
echo -e "${YELLOW}等待Gradio服务就绪...${NC}"
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$GRADIO_PORT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Gradio服务已就绪！${NC}"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo -e "${YELLOW}等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)${NC}"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo -e "${RED}错误: Gradio服务启动超时${NC}"
    echo -e "${YELLOW}请查看日志: tail -f $GRADIO_LOG${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Gradio服务启动成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}访问地址:${NC}"
echo -e "  http://localhost:$GRADIO_PORT"
echo ""
echo -e "${GREEN}服务信息:${NC}"
echo -e "  进程ID: $GRADIO_PID"
echo -e "  日志: $GRADIO_LOG"
echo ""
echo -e "${YELLOW}提示:${NC}"
echo -e "  - 停止服务: ./scripts/stop_gradio.sh"
echo -e "  - 查看日志: tail -f $GRADIO_LOG"
echo ""

