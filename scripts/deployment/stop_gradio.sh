#!/bin/bash

# Gradio前端服务停止脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="$PROJECT_ROOT/log"
PID_FILE="$LOG_DIR/gradio.pid"

echo -e "${YELLOW}正在停止Gradio服务...${NC}"

# 从PID文件读取进程ID
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "  找到进程: $PID"
        kill $PID
        sleep 2
        # 如果还在运行，强制杀死
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "  强制停止进程..."
            kill -9 $PID
        fi
        echo -e "${GREEN}✓ Gradio服务已停止${NC}"
    else
        echo -e "${YELLOW}进程 $PID 不存在${NC}"
    fi
    rm -f "$PID_FILE"
else
    echo -e "${YELLOW}PID文件不存在，尝试通过端口查找进程...${NC}"
    GRADIO_PORT="${GRADIO_PORT:-8080}"
    PID=$(lsof -ti:$GRADIO_PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo -e "  找到进程: $PID"
        kill $PID
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID
        fi
        echo -e "${GREEN}✓ Gradio服务已停止${NC}"
    else
        echo -e "${YELLOW}未找到运行中的Gradio服务${NC}"
    fi
fi

echo ""

