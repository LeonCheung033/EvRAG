#!/bin/bash
# EvRAG服务启动脚本
# 阶段一：环境配置和服务启动

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EvRAG 服务启动脚本${NC}"
echo -e "${GREEN}阶段一：环境配置和服务启动${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}警告: conda未安装，跳过环境激活${NC}"
else
    # 激活conda环境（根据你的环境名称修改）
    if conda env list | grep -q "evrag"; then
        echo -e "${GREEN}激活conda环境: evrag${NC}"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate evrag
    else
        echo -e "${YELLOW}警告: conda环境evrag不存在，跳过环境激活${NC}"
    fi
fi

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com

# 创建必要的目录
mkdir -p "$PROJECT_ROOT/log"
mkdir -p "$PROJECT_ROOT/data/mongodb/data"
mkdir -p "$PROJECT_ROOT/data/mongodb/log"
mkdir -p "$PROJECT_ROOT/data/saved_index"

# 1. 启动MongoDB
echo -e "\n${GREEN}[1/3] 启动MongoDB...${NC}"
MONGODB_BIN="$PROJECT_ROOT/mongodb/mongodb-7.0.20/bin/mongod"
MONGODB_DBPATH="$PROJECT_ROOT/data/mongodb/data"
MONGODB_LOGPATH="$PROJECT_ROOT/data/mongodb/log/mongodb.log"

if [ ! -f "$MONGODB_BIN" ]; then
    echo -e "${RED}错误: MongoDB可执行文件不存在: $MONGODB_BIN${NC}"
    exit 1
fi

# 检查并修复 libcurl 依赖
check_libcurl() {
    # 检查 libcurl.so.4 是否存在
    if ldconfig -p 2>/dev/null | grep -q "libcurl.so.4"; then
        return 0
    fi
    
    # 尝试查找 libcurl 库文件
    LIBCURL_PATH=$(find /usr/lib* -name "libcurl.so*" 2>/dev/null | head -1)
    if [ -n "$LIBCURL_PATH" ]; then
        LIBCURL_DIR=$(dirname "$LIBCURL_PATH")
        # 设置 LD_LIBRARY_PATH
        export LD_LIBRARY_PATH="$LIBCURL_DIR:${LD_LIBRARY_PATH:-}"
        echo -e "${YELLOW}设置 LD_LIBRARY_PATH: $LD_LIBRARY_PATH${NC}"
        return 0
    fi
    
    # 如果找不到，尝试创建符号链接
    LIBCURL_GNUTLS=$(find /usr/lib* -name "libcurl-gnutls.so.4" 2>/dev/null | head -1)
    if [ -n "$LIBCURL_GNUTLS" ]; then
        LIBCURL_DIR=$(dirname "$LIBCURL_GNUTLS")
        LIBCURL_SO4="$LIBCURL_DIR/libcurl.so.4"
        if [ ! -f "$LIBCURL_SO4" ]; then
            echo -e "${YELLOW}创建 libcurl.so.4 符号链接...${NC}"
            ln -sf "$LIBCURL_GNUTLS" "$LIBCURL_SO4" 2>/dev/null || true
            ldconfig 2>/dev/null || true
        fi
        export LD_LIBRARY_PATH="$LIBCURL_DIR:${LD_LIBRARY_PATH:-}"
        return 0
    fi
    
    echo -e "${RED}错误: 未找到 libcurl.so.4，请安装 libcurl4 包${NC}"
    echo -e "${YELLOW}安装命令: apt-get update && apt-get install -y libcurl4${NC}"
    return 1
}

# 检查 libcurl 依赖
if ! check_libcurl; then
    echo -e "${RED}MongoDB 启动失败: 缺少 libcurl.so.4 依赖${NC}"
    exit 1
fi

# 检查MongoDB是否已在运行
if pgrep -f "mongod.*27017" > /dev/null; then
    echo -e "${YELLOW}MongoDB已在运行，跳过启动${NC}"
else
    # 启动 MongoDB，确保使用正确的库路径
    $MONGODB_BIN --port=27017 \
        --dbpath="$MONGODB_DBPATH" \
        --logpath="$MONGODB_LOGPATH" \
        --bind_ip=0.0.0.0 \
        --fork
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ MongoDB已启动${NC}"
        sleep 2
    else
        echo -e "${RED}✗ MongoDB启动失败，请查看日志: $MONGODB_LOGPATH${NC}"
        exit 1
    fi
fi

# 2. 配置Milvus连接（本地文件模式）
echo -e "\n${GREEN}[2/3] 配置Milvus连接（本地文件模式）...${NC}"
MILVUS_DB_PATH="$PROJECT_ROOT/data/saved_index/milvus.db"
MILVUS_DB_DIR=$(dirname "$MILVUS_DB_PATH")

# 确保Milvus数据库目录存在
mkdir -p "$MILVUS_DB_DIR"
echo -e "${GREEN}✓ Milvus数据库目录已创建: $MILVUS_DB_DIR${NC}"

# 测试Milvus连接
echo -e "${YELLOW}测试Milvus连接...${NC}"
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from pymilvus import connections, utility
    
    # 尝试连接Milvus（本地文件模式）
    db_path = Path('$MILVUS_DB_PATH').absolute()
    
    # 检查是否已有连接
    if 'default' not in connections.list_connections():
        connections.connect(
            alias='default',
            uri=str(db_path)
        )
        print('✓ Milvus连接成功')
    else:
        print('✓ Milvus连接已存在')
    
    # 列出所有连接
    conns = connections.list_connections()
    print(f'当前Milvus连接: {conns}')
    
except ImportError as e:
    print(f'✗ 导入pymilvus失败: {e}')
    print('请确保已安装: pip install pymilvus pymilvus-model')
    sys.exit(1)
except Exception as e:
    print(f'✗ Milvus连接失败: {e}')
    print('提示: Milvus使用本地文件模式，首次连接会自动创建数据库文件')
    # 不退出，因为可能是首次使用，数据库文件还未创建
" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Milvus连接配置完成${NC}"
else
    echo -e "${YELLOW}警告: Milvus连接测试失败，但可能不影响后续使用（首次使用时会自动创建）${NC}"
fi

# 3. 启动语义切分服务
echo -e "\n${GREEN}[3/3] 启动语义切分服务...${NC}"
if pgrep -f "semantic_chunk_server" > /dev/null; then
    echo -e "${YELLOW}语义切分服务已在运行，跳过启动${NC}"
else
    nohup python -m src.evrag.server.semantic_chunk_server > "$PROJECT_ROOT/log/semantic_chunk.log" 2>&1 &
    SEMANTIC_PID=$!
    echo -e "${GREEN}✓ 语义切分服务已启动 (PID: $SEMANTIC_PID)${NC}"
    sleep 5
    
    # 检查服务是否正常启动（使用python替代curl）
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:6000/health').read()" 2>/dev/null > /dev/null; then
        echo -e "${GREEN}✓ 语义切分服务健康检查通过${NC}"
    else
        echo -e "${YELLOW}警告: 语义切分服务健康检查失败，请查看日志: $PROJECT_ROOT/log/semantic_chunk.log${NC}"
        echo -e "${YELLOW}提示: 服务可能正在加载模型，请稍后手动检查${NC}"
    fi
fi

# 4. 检查所有服务状态
echo -e "\n${GREEN}[4/4] 检查服务状态...${NC}"

# 检查MongoDB
if pgrep -f "mongod.*27017" > /dev/null; then
    echo -e "${GREEN}✓ MongoDB: 运行中 (端口 27017)${NC}"
else
    echo -e "${RED}✗ MongoDB: 未运行${NC}"
fi

# 检查Milvus连接
if python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from pymilvus import connections
try:
    if 'default' in connections.list_connections():
        print('OK')
    else:
        print('FAIL')
except:
    print('FAIL')
" 2>/dev/null | grep -q "OK"; then
    echo -e "${GREEN}✓ Milvus: 连接正常 (本地文件模式)${NC}"
else
    echo -e "${YELLOW}○ Milvus: 连接未建立（首次使用时会在索引构建时自动连接）${NC}"
fi

# 检查语义切分服务（使用python替代curl）
if python -c "import urllib.request; urllib.request.urlopen('http://localhost:6000/health').read()" 2>/dev/null > /dev/null; then
    echo -e "${GREEN}✓ 语义切分服务: 运行中 (端口 6000)${NC}"
else
    # 如果健康检查失败，但进程存在，可能是模型还在加载
    if pgrep -f "semantic_chunk_server" > /dev/null; then
        echo -e "${YELLOW}○ 语义切分服务: 进程运行中，但健康检查失败（可能正在加载模型）${NC}"
    else
        echo -e "${RED}✗ 语义切分服务: 未运行${NC}"
    fi
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}服务启动完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "日志目录: $PROJECT_ROOT/log"
echo -e "MongoDB日志: $PROJECT_ROOT/data/mongodb/log/mongodb.log"
echo -e "语义切分服务日志: $PROJECT_ROOT/log/semantic_chunk.log"
echo -e "Milvus数据库路径: $PROJECT_ROOT/data/saved_index/milvus.db"
echo -e ""
echo -e "${YELLOW}注意: vLLM服务将在模型微调完成后单独启动${NC}"