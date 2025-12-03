#!/bin/bash

# 停止vLLM评估服务脚本

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
BASELINE_PID_FILE="$LOG_DIR/vllm_baseline.pid"
FINETUNED_PID_FILE="$LOG_DIR/vllm_finetuned.pid"

BASELINE_PORT="${BASELINE_PORT:-8000}"
FINETUNED_PORT="${FINETUNED_PORT:-8001}"

echo -e "${YELLOW}正在停止 vLLM 评估服务...${NC}"
echo ""

# 停止基线服务
if [ -f "$BASELINE_PID_FILE" ]; then
    BASELINE_PID=$(cat "$BASELINE_PID_FILE")
    if ps -p $BASELINE_PID > /dev/null 2>&1; then
        echo -e "停止基线服务（PID: $BASELINE_PID）..."
        kill $BASELINE_PID 2>/dev/null || true
        sleep 2
        if ps -p $BASELINE_PID > /dev/null 2>&1; then
            kill -9 $BASELINE_PID 2>/dev/null || true
        fi
        echo -e "${GREEN}✓ 基线服务已停止${NC}"
    else
        echo -e "${YELLOW}基线服务进程不存在${NC}"
    fi
    rm -f "$BASELINE_PID_FILE"
else
    echo -e "${YELLOW}未找到基线服务PID文件${NC}"
fi

# 停止微调服务
if [ -f "$FINETUNED_PID_FILE" ]; then
    FINETUNED_PID=$(cat "$FINETUNED_PID_FILE")
    if ps -p $FINETUNED_PID > /dev/null 2>&1; then
        echo -e "停止微调服务（PID: $FINETUNED_PID）..."
        kill $FINETUNED_PID 2>/dev/null || true
        sleep 2
        if ps -p $FINETUNED_PID > /dev/null 2>&1; then
            kill -9 $FINETUNED_PID 2>/dev/null || true
        fi
        echo -e "${GREEN}✓ 微调服务已停止${NC}"
    else
        echo -e "${YELLOW}微调服务进程不存在${NC}"
    fi
    rm -f "$FINETUNED_PID_FILE"
else
    echo -e "${YELLOW}未找到微调服务PID文件${NC}"
fi

# 清理端口占用
if lsof -Pi :$BASELINE_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "清理端口 $BASELINE_PORT..."
    lsof -ti:$BASELINE_PORT | xargs kill -9 2>/dev/null || true
fi

if lsof -Pi :$FINETUNED_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "清理端口 $FINETUNED_PORT..."
    lsof -ti:$FINETUNED_PORT | xargs kill -9 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}✓ 所有服务已停止${NC}"

