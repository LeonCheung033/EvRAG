#!/bin/bash
# EvRAG服务停止脚本

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m's
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EvRAG 服务停止脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 停止语义切分服务
echo -e "\n${GREEN}[1/2] 停止语义切分服务...${NC}"
SEMANTIC_PIDS=$(pgrep -f "semantic_chunk_server" || true)
if [ -z "$SEMANTIC_PIDS" ]; then
    echo -e "${YELLOW}语义切分服务未运行${NC}"
else
    echo -e "${YELLOW}找到语义切分服务进程: $SEMANTIC_PIDS${NC}"
    for pid in $SEMANTIC_PIDS; do
        echo -e "${YELLOW}正在停止进程 $pid...${NC}"
        kill -TERM "$pid" 2>/dev/null || true
    done
    
    # 等待进程优雅退出（最多等待10秒）
    sleep 2
    REMAINING_PIDS=$(pgrep -f "semantic_chunk_server" || true)
    if [ -n "$REMAINING_PIDS" ]; then
        echo -e "${YELLOW}进程仍在运行，强制终止...${NC}"
        for pid in $REMAINING_PIDS; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    
    # 最终检查
    if pgrep -f "semantic_chunk_server" > /dev/null; then
        echo -e "${RED}✗ 语义切分服务停止失败${NC}"
    else
        echo -e "${GREEN}✓ 语义切分服务已停止${NC}"
    fi
fi

# 2. 停止MongoDB
echo -e "\n${GREEN}[2/2] 停止MongoDB...${NC}"
MONGODB_PIDS=$(pgrep -f "mongod.*27017" || true)
if [ -z "$MONGODB_PIDS" ]; then
    echo -e "${YELLOW}MongoDB未运行${NC}"
else
    echo -e "${YELLOW}找到MongoDB进程: $MONGODB_PIDS${NC}"
    
    # 尝试优雅关闭（发送 SIGTERM）
    for pid in $MONGODB_PIDS; do
        echo -e "${YELLOW}正在停止MongoDB进程 $pid...${NC}"
        kill -TERM "$pid" 2>/dev/null || true
    done
    
    # 等待MongoDB优雅关闭（最多等待15秒）
    echo -e "${YELLOW}等待MongoDB优雅关闭...${NC}"
    for i in {1..15}; do
        if ! pgrep -f "mongod.*27017" > /dev/null; then
            break
        fi
        sleep 1
    done
    
    # 如果仍在运行，强制终止
    REMAINING_PIDS=$(pgrep -f "mongod.*27017" || true)
    if [ -n "$REMAINING_PIDS" ]; then
        echo -e "${YELLOW}MongoDB未响应，强制终止...${NC}"
        for pid in $REMAINING_PIDS; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    
    # 最终检查
    if pgrep -f "mongod.*27017" > /dev/null; then
        echo -e "${RED}✗ MongoDB停止失败${NC}"
    else
        echo -e "${GREEN}✓ MongoDB已停止${NC}"
    fi
fi

# 3. 显示最终状态
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}服务停止完成！${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查服务状态
echo -e "\n${GREEN}服务状态检查:${NC}"

# 检查MongoDB
if pgrep -f "mongod.*27017" > /dev/null; then
    echo -e "${RED}✗ MongoDB: 仍在运行${NC}"
else
    echo -e "${GREEN}✓ MongoDB: 已停止${NC}"
fi

# 检查语义切分服务
if pgrep -f "semantic_chunk_server" > /dev/null; then
    echo -e "${RED}✗ 语义切分服务: 仍在运行${NC}"
else
    echo -e "${GREEN}✓ 语义切分服务: 已停止${NC}"
fi

echo ""

