#!/bin/bash

# vLLM服务停止脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PID_FILE="$PROJECT_ROOT/log/vllm.pid"

echo -e "${YELLOW}正在停止 vLLM 服务...${NC}"

# 从PID文件读取进程ID
if [ -f "$PID_FILE" ]; then
    VLLM_PID=$(cat "$PID_FILE")
    if ps -p $VLLM_PID > /dev/null 2>&1; then
        echo -e "找到进程 ID: $VLLM_PID"
        kill $VLLM_PID
        sleep 2
        # 如果进程还在，强制杀死
        if ps -p $VLLM_PID > /dev/null 2>&1; then
            echo -e "${YELLOW}强制停止进程...${NC}"
            kill -9 $VLLM_PID
        fi
        echo -e "${GREEN}✓ vLLM 服务已停止${NC}"
        rm -f "$PID_FILE"
    else
        echo -e "${YELLOW}进程 $VLLM_PID 不存在${NC}"
        rm -f "$PID_FILE"
    fi
else
    echo -e "${YELLOW}未找到 PID 文件: $PID_FILE${NC}"
fi

# 也尝试通过端口停止（备用方法）
VLLM_PORT=8000
if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}检测到端口 $VLLM_PORT 仍被占用，尝试停止...${NC}"
    lsof -ti:$VLLM_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}✓ 已停止占用端口 $VLLM_PORT 的进程${NC}"
fi

echo -e "${GREEN}完成${NC}"

