#!/bin/bash
# 基线模型评估脚本

set -e

# 配置
TEST_DATA="data/qa_pairs/test_qa_pair_handmade.json"  # 测试数据
OUTPUT_BASE_DIR="rag_test_reports/baseline_evaluation"
MAX_WORKERS=6  # 并发数（可根据显存情况调整）
USE_RAGAS=true  # 是否使用RAGas评估

# 检查测试数据是否存在
if [ ! -f "${TEST_DATA}" ]; then
    echo "错误: 测试数据文件不存在: ${TEST_DATA}"
    echo "请确保测试数据文件存在"
    exit 1
fi

# 获取时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE_DIR}_${TIMESTAMP}"

echo "=========================================="
echo "基线模型评估"
echo "=========================================="
echo "测试数据: ${TEST_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "并发数: ${MAX_WORKERS}"
echo "使用RAGas: ${USE_RAGAS}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行基线模型评估
echo "=========================================="
echo "运行基线模型评估"
echo "=========================================="

python main.py evaluate-rag \
    --test-data "${TEST_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --use-baseline \
    --full-test \
    --max-workers ${MAX_WORKERS} \
    $([ "${USE_RAGAS}" = "true" ] && echo "--use-ragas" || echo "--no-use-ragas") \
    2>&1 | tee "${OUTPUT_DIR}/baseline_evaluation.log"

echo ""
echo "✓ 基线模型评估完成"
echo ""

# 分析评估结果
echo "=========================================="
echo "分析评估结果"
echo "=========================================="

if [ -f "scripts/analyze_evaluation_results.py" ]; then
    echo "分析基线模型结果..."
    # 注意：evaluate-rag命令会根据use_baseline参数生成不同的文件名
    # 基线模型的文件名是 baseline_evaluation_results.json 和 baseline_evaluation_summary.json
    SUMMARY_FILE="${OUTPUT_DIR}/baseline_evaluation_summary.json"
    RESULTS_FILE="${OUTPUT_DIR}/baseline_evaluation_results.json"
    
    if [ -f "${SUMMARY_FILE}" ] && [ -f "${RESULTS_FILE}" ]; then
        python scripts/analyze_evaluation_results.py \
            "${SUMMARY_FILE}" \
            "${RESULTS_FILE}" \
            > "${OUTPUT_DIR}/baseline_analysis.txt" 2>&1 || true
    else
        echo "⚠ 警告: 评估结果文件不存在，跳过分析"
        echo "  期望文件: ${SUMMARY_FILE}"
        echo "  期望文件: ${RESULTS_FILE}"
    fi
fi

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="
echo "结果目录: ${OUTPUT_DIR}"
echo "  - 评估结果: ${OUTPUT_DIR}/baseline_evaluation_results.json"
echo "  - 汇总指标: ${OUTPUT_DIR}/baseline_evaluation_summary.json"
echo "  - 评估日志: ${OUTPUT_DIR}/baseline_evaluation.log"
if [ -f "${OUTPUT_DIR}/baseline_analysis.txt" ]; then
    echo "  - 分析报告: ${OUTPUT_DIR}/baseline_analysis.txt"
fi
echo ""

