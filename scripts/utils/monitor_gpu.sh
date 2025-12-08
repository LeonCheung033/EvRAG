#!/bin/bash
# GPU监控脚本

# 项目根目录
PROJECT_ROOT="/remote-home/share/liangZhang/EvRAG"
LOG_DIR="${PROJECT_ROOT}/logs"
GPU_LOG="${LOG_DIR}/gpu_monitor.log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 监控间隔（秒）
INTERVAL=${INTERVAL:-5}

echo "开始GPU监控..."
echo "日志文件: ${GPU_LOG}"
echo "监控间隔: ${INTERVAL} 秒"
echo "按 Ctrl+C 停止监控"

# 监控循环
while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader | while IFS= read -r line; do
        echo "${timestamp},${line}" >> "${GPU_LOG}"
    done
    sleep "${INTERVAL}"
done

