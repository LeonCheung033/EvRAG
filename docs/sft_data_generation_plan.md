<!-- c4822b3f-1ee8-44f5-8a16-f46f66d8c0fc 1b618c62-5ed3-4cdd-a708-ee0a720d86da -->
# 微调数据生成计划（Phase 3）

## 一、文件生成流程说明（基于原项目分析）

### 核心文件生成流程

1. **train_data.json** → 从 `train_qa_pair.json` 生成RAG检索数据（需要实现）

   - 使用RAG系统（BM25 + Milvus + Reranker）检索上下文
   - 使用LLM生成响应（带引用标记）
   - 格式：每行一个JSON，`{"query": "...", "context": [...], "response": "...", "merged_docs": [...]}`

2. **summary_data/train.json** 和 **summary_data/test.json** → 从 `train_data.json` 生成SFT训练数据（需要实现）

   - 解析LLM响应，提取答案和引用
   - 格式化instruction和output
   - 格式：JSON数组，`{"query": "...", "context": "...", "instruction": "...", "input": "", "output": "..."}`

3. **rerank_data/train.json**、**rerank_data/dev.json**、**rerank_data/test.json** → 从 `train_data.json` 生成Reranker训练数据（需要实现）

   - 生成正样本（label=2）、中等样本（label=1）、负样本（label=0）
   - 格式：每行一个JSON，`{"query": "...", "content": "...", "label": 0/1/2}`

## 二、实现方案

### 1. 创建微调数据生成模块

**新项目文件**: `src/evrag/gen_qa/sft_data_generator.py`

**主要功能**:

- 从 `train_qa_pair.json` 加载QA对
- 使用RAG系统检索上下文（BM25 + Milvus + Reranker）
- 使用LLM生成响应（带引用标记）
- 生成 `train_data.json`
- 从 `train_data.json` 生成summary_data和rerank_data

### 2. 原项目代码参考路径

**核心实现文件**: `/remote-home/share/liangZhang/EVRAG/generate_sft_data.py`

这是原项目中微调数据生成的主文件，包含完整的实现逻辑。新项目实现时应参考此文件。

### 3. 实现步骤（含原项目代码路径）

#### 步骤1: 生成train_data.json（RAG检索+LLM生成）

**原项目代码位置**:

- RAG检索流程：`/remote-home/share/liangZhang/EVRAG/generate_sft_data.py:48-69`
- Prompt模板：`/remote-home/share/liangZhang/EVRAG/generate_sft_data.py:26-36`

**实现要点**:

- 从 `train_qa_pair.json` 读取QA对
- 对每个问题执行RAG检索：
  - BM25检索（topk=5）
  - Milvus检索（topk=10）
  - 合并文档（merge_docs）
  - Reranker重排序（topk=5）
- 使用LLM生成响应（带引用标记格式：`答案【引用编号1,引用编号2】`）
- 保存到 `train_data.json`（JSONL格式）

**参考代码片段**:

```python
# 原项目 generate_sft_data.py:48-66
query = item["question"].strip()
bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
merged_docs = merge_docs(bm25_docs, milvus_docs)
ranked_docs = qwen3_reranker.rank(query, merged_docs, topk=5)
context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
response = request_chat(query, context)
context = [q.page_content for q in ranked_docs]
all_docs = [q.page_content for q in merged_docs]
info = {"query": query, "context": context, "response": response, "merged_docs": all_docs}
```

**新项目实现**:

- 使用 `BM25Retriever`、`MilvusRetriever`、`BGEReranker`
- 使用 `ChatClient` 生成响应
- 使用 `merge_docs` 和 `post_processing` 工具函数

#### 步骤2: 生成summary_data（SFT训练数据）

**原项目代码位置**:

- 数据解析和格式化：`/remote-home/share/liangZhang/EVRAG/generate_sft_data.py:86-117`

**实现要点**:

- 从 `train_data.json` 读取数据
- 解析LLM响应，提取答案和引用：
  - 提取引用标记：`【1, 2, 3】`
  - 去除引用标记，得到纯答案
  - 格式化答案：`答案【引用编号】` 或 `无答案`
- 构建instruction（使用LLM_CHAT_PROMPT模板）
- 切分训练集和测试集（TEST_RATE=0.08）
- 保存到 `summary_data/train.json` 和 `summary_data/test.json`

**参考代码片段**:

```python
# 原项目 generate_sft_data.py:89-117
all_cites = re.findall("[【](.*?)[】]", response)
cites = sorted(list(set(cites)))
format_answer = answer + f"【{cites}】" if cites else "无答案"
instruction = LLM_CHAT_PROMPT.format(query=query, context=context)
item = {
    "query": query,
    "context": context,
    "instruction": instruction,
    "input": "",
    "output": format_answer
}
```

#### 步骤3: 生成rerank_data（Reranker训练数据）

**原项目代码位置**:

- Reranker数据生成：`/remote-home/share/liangZhang/EVRAG/generate_sft_data.py:118-142`

**实现要点**:

- 从 `train_data.json` 读取数据
- 为每个QA对生成Reranker样本：
  - **正样本（label=2）**：`context[0]`（最相关的文档）
  - **中等样本（label=1）**：从 `context[-2:]` 随机选择一个
  - **负样本（label=0）**：从 `merged_docs` 中不在 `context` 的文档随机选择，或如果答案为"无答案"则从所有 `merged_docs` 随机选择
- 测试集：如果答案为"无答案"，不生成rerank数据；否则生成包含正样本、中等样本和负样本的列表
- 从训练集中切分dev集（RERANK_DEV_SIZE=1000）
- 保存到 `rerank_data/train.json`、`rerank_data/dev.json`、`rerank_data/test.json`

**参考代码片段**:

```python
# 原项目 generate_sft_data.py:132-142
if format_answer != "无答案":
    positive = info["context"][0]
    middle = random.choice(info["context"][-2:])
    rerank_train.append({"query": query, "content": positive, "label": 2})
    rerank_train.append({"query": query, "content": middle, "label": 1})
    if neg_docs:
        negative = random.choice(neg_docs)
        rerank_train.append({"query": query, "content": negative, "label": 0})
```

## 三、实现细节

### 1. 数据格式

#### train_data.json（JSONL格式）

```json
{"query": "问题", "context": ["文档1", "文档2", ...], "response": "答案【1,2】", "merged_docs": ["所有文档", ...]}
```

#### summary_data/train.json（JSON数组格式）

```json
[
  {
    "query": "问题",
    "context": "1.文档1\n2.文档2\n...",
    "instruction": "### 信息\n...\n### 任务\n...",
    "input": "",
    "output": "答案【1,2】"
  }
]
```

#### rerank_data/train.json（JSONL格式）

```json
{"query": "问题", "content": "文档内容", "label": 2}
{"query": "问题", "content": "文档内容", "label": 1}
{"query": "问题", "content": "文档内容", "label": 0}
```

### 2. 配置参数

- `BM25_TOPK`: BM25检索数量（默认：5）
- `MILVUS_TOPK`: Milvus检索数量（默认：10）
- `RERANKER_TOPK`: Reranker重排序数量（默认：5）
- `MAX_INPUT_SIZE`: 最大输入长度（默认：4096）
- `TEST_RATE`: 测试集比例（默认：0.08）
- `RERANK_DEV_SIZE`: Reranker开发集大小（默认：1000）

### 3. 错误处理

- RAG检索失败重试机制
- LLM生成失败重试机制
- 数据格式验证
- 文件读写错误处理

### 4. 性能优化

- 并发处理（使用ThreadPoolExecutor）
- 支持断点续传（checkpoint机制）
- 进度条显示（tqdm）

## 四、文件结构

```
src/evrag/gen_qa/
├── sft_data_generator.py    # 新建，微调数据生成主模块
└── ...

data/
├── qa_pairs/
│   └── train_qa_pair.json   # 输入：训练集QA对
├── qa_pairs/
│   └── train_data.json      # 中间：RAG检索数据
├── summary_data/
│   ├── train.json           # 输出：SFT训练集
│   └── test.json            # 输出：SFT测试集
└── rerank_data/
    ├── train.json           # 输出：Reranker训练集
    ├── dev.json             # 输出：Reranker开发集
    └── test.json             # 输出：Reranker测试集
```

## 五、在main.py中添加命令

**命令**: `generate-sft-data`

**参数**:

- `--train-qa-path`: 训练集QA对文件路径（默认：`data/qa_pairs/train_qa_pair.json`）
- `--output-dir`: 输出目录（默认：`data`）
- `--step`: 执行步骤（`train_data`, `summary`, `rerank`, `all`）
- `--bm25-topk`: BM25检索数量（默认：5）
- `--milvus-topk`: Milvus检索数量（默认：10）
- `--reranker-topk`: Reranker重排序数量（默认：5）
- `--test-rate`: 测试集比例（默认：0.08）
- `--rerank-dev-size`: Reranker开发集大小（默认：1000）
- `--max-workers`: 最大并发工作线程数（默认：10）

## 六、数据分析和可视化

### 1. 数据统计

- SFT数据：训练集/测试集数量、平均context长度、平均output长度
- Reranker数据：训练集/开发集/测试集数量、标签分布（0/1/2比例）

### 2. 可视化图表

- SFT数据样本分布图（训练集vs测试集）
- Reranker数据标签分布图（正/负/中等样本比例）
- 数据生成进度图

## 七、测试和验证

1. 验证train_data.json格式正确性
2. 验证summary_data格式正确性
3. 验证rerank_data格式正确性
4. 验证数据平衡性（正负样本比例）
5. 验证数据完整性（无缺失字段）

## 八、参考文档

- 原项目实现：`/remote-home/share/liangZhang/EVRAG/generate_sft_data.py`
- QA数据处理文档：`docs/PHASE2_QA_DATA_PROCESSING.md`

### To-dos

- [ ] 创建SFTDataGenerator类，实现train_data.json生成功能
- [ ] 实现RAG检索流程（BM25 + Milvus + Reranker）
- [ ] 实现LLM响应生成（带引用标记）
- [ ] 实现summary_data生成（从train_data.json解析和格式化）
- [ ] 实现rerank_data生成（正/负/中等样本）
- [ ] 在main.py中添加generate-sft-data命令
- [ ] 添加数据统计和可视化功能
- [ ] 添加数据格式验证和完整性检查