# Qwen基线系统评估脚本使用说明

## 脚本位置
`scripts/run_qwen_baseline_evaluation.py`

## 功能说明
该脚本用于运行Qwen基线系统（Qwen3-32B + Qwen3-Embedding-8B）的RAG评估。
- 从原始PDF文件加载并分块（模拟真实chatbot的粗糙处理方式）
- 使用FAISS向量检索（无BM25，无重排序）
- 使用LangChain的RAG链生成答案

## 基本用法

### 1. 快速测试（默认抽样30条）
```bash
cd /remote-home/share/liangZhang/EvRAG
python scripts/run_qwen_baseline_evaluation.py
```

### 2. 使用指定的测试文件（全量测试）
```bash
python scripts/run_qwen_baseline_evaluation.py \
    --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \
    --full-test
```

### 3. 抽样测试（指定样本数）
```bash
python scripts/run_qwen_baseline_evaluation.py \
    --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \
    --sample-size 10
```

### 4. 自定义PDF路径和分块参数
```bash
python scripts/run_qwen_baseline_evaluation.py \
    --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \
    --pdf-path data/Tesla_Manual.pdf \
    --chunk-size 500 \
    --chunk-overlap 50 \
    --topk 5
```

### 5. 自定义输出目录
```bash
python scripts/run_qwen_baseline_evaluation.py \
    --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \
    --output-dir rag_test_reports/my_test \
    --full-test
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test-data` | str | `data/qa_pairs/test_qa_pair_handmade_verify01.json` | 测试数据文件路径 |
| `--output-dir` | str | None | 输出目录（None时自动生成带时间戳的目录） |
| `--pdf-path` | str | None | PDF文件路径（None时从config.yaml读取） |
| `--topk` | int | 10 | 检索数量 |
| `--chunk-size` | int | 500 | 文本分块大小 |
| `--chunk-overlap` | int | 50 | 文本分块重叠大小 |
| `--sample-size` | int | None | 抽样测试样本数（None表示全量，仅在--full-test未指定时生效） |
| `--full-test` | flag | False | 是否执行全量测试 |
| `--no-ragas` | flag | False | 不使用RAGas评估 |
| `--max-workers` | int | None | 最大并发工作线程数 |
| `--llm-model` | str | None | LLM模型名称（默认：Qwen/Qwen3-32B） |
| `--embedding-model` | str | None | Embedding模型名称（默认：Qwen/Qwen3-Embedding-8B） |
| `--temperature` | float | 0.1 | LLM温度参数 |
| `--max-tokens` | int | 512 | LLM最大生成token数 |

## 输出文件

评估完成后，会在输出目录生成以下文件：

1. **qwen_baseline_evaluation_results.json**: 详细的评估结果（每个样本的评估数据）
2. **qwen_baseline_evaluation_results_summary.json**: 汇总指标（平均值、标准差等）

## 评估指标

脚本会计算以下指标：

1. **语义相似度+关键词加权得分**: 结合语义相似度和关键词匹配的综合得分
2. **生成质量指标**: BLEU、ROUGE-1、ROUGE-2、ROUGE-L
3. **RAGas得分**（可选）: Context Recall、Context Precision
4. **综合准确率**: 语义得分和RAGas得分的加权平均
5. **响应时间**: 检索时间、生成时间、总时间

## 注意事项

1. **首次运行**：如果FAISS索引不存在，脚本会自动从PDF构建索引（可能需要一些时间）
2. **配置文件**：确保`config/config.yaml`中配置了`siliconflow_api_key`和`pdf_path`
3. **测试数据格式**：测试数据应为JSON数组，每个元素包含：
   - `unique_id`: 唯一标识符
   - `question`: 问题
   - `answer`: 标准答案
   - `keywords`: 关键词列表（可选）

## 示例输出

```
============================================================
Qwen基线系统评估
============================================================

📂 加载测试数据: data/qa_pairs/test_qa_pair_handmade_verify01.json
   总测试样本数: 258
   全量测试样本数: 258

📁 输出目录: rag_test_reports/qwen_baseline_20241201_120000

🔧 初始化Qwen基线评估器...
   ✓ Qwen基线评估器初始化成功

🚀 执行批量评估...
   评估进度: 100%|████████████| 258/258 [05:23<00:00,  1.25s/it]

   ✓ 评估完成
     有效结果数: 258

📊 计算汇总指标...

============================================================
评估结果汇总
============================================================
总样本数: 258

语义相似度+关键词加权得分:
  平均分: 0.7234
  标准差: 0.1523

生成质量指标:
  BLEU: 0.4521
  ROUGE-1: 0.6234
  ROUGE-2: 0.5123
  ROUGE-L: 0.5891

RAGas得分:
  Context Recall: 0.7123
  Context Precision: 0.6543
  平均分: 0.6912

综合准确率:
  得分: 0.7101
  语义得分: 0.7234
  RAGas得分: 0.6912

响应时间:
  平均检索时间: 0.45秒
  平均生成时间: 1.23秒
  平均总时间: 1.68秒
============================================================

💾 保存评估结果...
   ✓ 结果已保存到: rag_test_reports/qwen_baseline_20241201_120000/qwen_baseline_evaluation_results.json

============================================================
✓ 评估完成！
============================================================
结果文件: rag_test_reports/qwen_baseline_20241201_120000
```

