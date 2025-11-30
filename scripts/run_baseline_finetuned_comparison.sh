#!/bin/bash
# 基线和微调模型对比测试脚本

set -e

# 配置
TEST_DATA="data/qa_pairs/test_qa_pair_handmade.json"  # 全量测试数据（750条）
OUTPUT_BASE_DIR="rag_test_reports/comparison"
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
echo "基线和微调模型对比测试"
echo "=========================================="
echo "测试数据: ${TEST_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "并发数: ${MAX_WORKERS}"
echo "使用RAGas: ${USE_RAGAS}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 1. 运行基线模型评估
echo "=========================================="
echo "步骤1/2: 运行基线模型评估"
echo "=========================================="
BASELINE_OUTPUT="${OUTPUT_DIR}/baseline"
mkdir -p "${BASELINE_OUTPUT}"

python main.py evaluate-rag \
    --test-data "${TEST_DATA}" \
    --output-dir "${BASELINE_OUTPUT}" \
    --use-baseline \
    --full-test \
    --max-workers ${MAX_WORKERS} \
    $([ "${USE_RAGAS}" = "true" ] && echo "--use-ragas" || echo "--no-use-ragas") \
    2>&1 | tee "${OUTPUT_DIR}/baseline_evaluation.log"

echo ""
echo "✓ 基线模型评估完成"
echo ""

# 2. 运行微调模型评估
echo "=========================================="
echo "步骤2/2: 运行微调模型评估"
echo "=========================================="
FINETUNED_OUTPUT="${OUTPUT_DIR}/finetuned"
mkdir -p "${FINETUNED_OUTPUT}"

python main.py evaluate-rag \
    --test-data "${TEST_DATA}" \
    --output-dir "${FINETUNED_OUTPUT}" \
    --full-test \
    --max-workers ${MAX_WORKERS} \
    $([ "${USE_RAGAS}" = "true" ] && echo "--use-ragas" || echo "--no-use-ragas") \
    2>&1 | tee "${OUTPUT_DIR}/finetuned_evaluation.log"

echo ""
echo "✓ 微调模型评估完成"
echo ""

# 3. 生成对比报告
echo "=========================================="
echo "步骤3/3: 生成对比报告"
echo "=========================================="

python - <<EOF
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
baseline_summary = json.load(open(output_dir / "baseline" / "baseline_evaluation_summary.json"))
finetuned_summary = json.load(open(output_dir / "finetuned" / "finetuned_evaluation_summary.json"))

# 提取关键指标
baseline_comprehensive = baseline_summary.get("comprehensive_accuracy", {}).get("score", 0.0)
finetuned_comprehensive = finetuned_summary.get("comprehensive_accuracy", {}).get("score", 0.0)

baseline_semantic = baseline_summary.get("semantic_keyword_score", {}).get("mean", 0.0)
finetuned_semantic = finetuned_summary.get("semantic_keyword_score", {}).get("mean", 0.0)

baseline_ragas = baseline_summary.get("ragas_scores", {}).get("average", 0.0) if baseline_summary.get("ragas_scores") else 0.0
finetuned_ragas = finetuned_summary.get("ragas_scores", {}).get("average", 0.0) if finetuned_summary.get("ragas_scores") else 0.0

# 计算提升
improvement = ((finetuned_comprehensive - baseline_comprehensive) / baseline_comprehensive * 100) if baseline_comprehensive > 0 else 0.0

# 生成对比报告
comparison = {
    "baseline": {
        "comprehensive_accuracy": baseline_comprehensive,
        "semantic_keyword_score": baseline_semantic,
        "ragas_score": baseline_ragas,
    },
    "finetuned": {
        "comprehensive_accuracy": finetuned_comprehensive,
        "semantic_keyword_score": finetuned_semantic,
        "ragas_score": finetuned_ragas,
    },
    "improvement": {
        "comprehensive_accuracy_improvement": improvement,
        "absolute_improvement": finetuned_comprehensive - baseline_comprehensive,
    }
}

# 保存对比报告
with open(output_dir / "comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

# 打印对比结果
print("\n" + "="*50)
print("对比结果")
print("="*50)
print(f"基线模型 - 综合准确率: {baseline_comprehensive:.4f} ({baseline_comprehensive*100:.2f}%)")
print(f"微调模型 - 综合准确率: {finetuned_comprehensive:.4f} ({finetuned_comprehensive*100:.2f}%)")
print(f"提升: {improvement:.2f}%")
print(f"绝对提升: {finetuned_comprehensive - baseline_comprehensive:.4f}")
print("="*50)
EOF

echo ""
echo "✓ 对比报告已生成: ${OUTPUT_DIR}/comparison_results.json"
echo ""

# 4. 分析评估结果
echo "=========================================="
echo "步骤4/4: 分析评估结果"
echo "=========================================="

echo "分析基线模型结果..."
python scripts/analyze_evaluation_results.py \
    "${OUTPUT_DIR}/baseline/baseline_evaluation_summary.json" \
    "${OUTPUT_DIR}/baseline/baseline_evaluation_results.json" \
    > "${OUTPUT_DIR}/baseline_analysis.txt" 2>&1

echo "分析微调模型结果..."
python scripts/analyze_evaluation_results.py \
    "${OUTPUT_DIR}/finetuned/finetuned_evaluation_summary.json" \
    "${OUTPUT_DIR}/finetuned/finetuned_evaluation_results.json" \
    > "${OUTPUT_DIR}/finetuned_analysis.txt" 2>&1

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "结果目录: ${OUTPUT_DIR}"
echo "  - 基线模型结果: ${OUTPUT_DIR}/baseline/"
echo "  - 微调模型结果: ${OUTPUT_DIR}/finetuned/"
echo "  - 对比报告: ${OUTPUT_DIR}/comparison_results.json"
echo "  - 基线分析: ${OUTPUT_DIR}/baseline_analysis.txt"
echo "  - 微调分析: ${OUTPUT_DIR}/finetuned_analysis.txt"
echo ""

