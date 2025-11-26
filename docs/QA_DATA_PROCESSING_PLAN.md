# QA数据处理和训练测试集生成计划

## 一、文件生成流程说明（基于原项目分析）

### 核心文件生成流程

1. **qa_pair.json** → 从 `clean_docs.pkl` 生成原始QA对（已实现）

2. **expand_qa_pair.json** → 从 `qa_pair.json` 中的问题生成同义问题（需要实现）
   - 格式：每行一个JSON，`{"unique_id": "问题文本", "raw_resp": "改写后的5个问题（换行分隔）"}`

3. **train_qa_pair.json** 和 **test_qa_pair.json** → 从扩充后的QA对切分生成（需要实现）
   - 90% 训练集，10% 测试集
   - 测试集包含 `keywords` 字段

4. **test_keywords_pair.json** → 从测试集的唯一答案提取关键词（需要实现）
   - 格式：每行一个JSON，`{"unique_id": "答案文本", "raw_resp": "关键词（逗号分隔）"}`
   - 注意：原项目代码有bug（变量名写错），新项目需要修正

5. **train_data.json** → 从 `train_qa_pair.json` 生成微调数据（后续阶段实现，不在本阶段）
   - 使用RAG系统（BM25 + Milvus + Reranker）检索上下文
   - 生成包含 `query`, `context`, `response`, `merged_docs` 的数据
   - 用于生成微调训练数据（summary_data 和 rerank_data）

6. **test_qa_pair_verify.json** → 从 `test_qa_pair.json` 筛选验证集（可选，用于RAG评估）
   - 通常是从完整测试集中筛选一部分用于验证
   - 用于RAG系统性能评估

7. **test_qa_pair_pred.json** → 对 `test_qa_pair_verify.json` 进行RAG预测（后续阶段实现，不在本阶段）
   - 包含预测答案、引用页面、相关图片等
   - 用于评估RAG系统的性能（使用语义相似度+关键词加权得分）

## 二、实现方案

### 1. 创建QA数据处理模块

**新项目文件**: `src/evrag/gen_qa/qa_processor.py`

**主要功能**:
- 加载已生成的 `qa_pair.json`
- QA质量打分（使用Deepseek）
- 过滤低质量QA对
- 问题改写（生成同义问题）
- 答案匹配和扩充
- 训练/测试集切分
- 测试集关键词提取
- 负样本添加
- 输出最终训练和测试集

### 2. 原项目代码参考路径

**核心实现文件**: `/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py`

这是原项目中QA数据处理的主文件，包含完整的实现逻辑。新项目实现时应参考此文件。

### 3. 实现步骤（含原项目代码路径）

#### 步骤1: QA质量打分和过滤

**原项目代码位置**: 
- Prompt模板：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py:106-130` (QA_QUALITY_PROMPT_TPL)
- 新项目已有实现：`src/evrag/gen_qa/generator.py:444-482` (score_qa_quality方法)

**实现要点**:
- 使用 `QAGenerator.score_qa_quality()` 方法对每个QA对打分
- 过滤阈值：**3分**（保留3分及以上的QA对）
- 同时过滤包含"无法准确"或"未提及"的答案（与原项目一致）
- 原项目代码：`run.py:253` 过滤逻辑

**参考代码片段**:
```python
# 原项目 run.py:253
if "无法准确" in answer or "未提及" in answer:
    continue
```

#### 步骤2: 问题改写（Query Rewriting）

**原项目代码位置**: 
- Prompt模板：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py:57-73` (GENERALIZE_PROMPT_TPL)
- 问题提取和改写：`run.py:213-223` (从qa_pair.json提取问题，生成expand_qa_pair.json)
- 核心函数：`run.py:171-203` (gen_qa函数，支持expand模式)

**实现要点**:
- 从过滤后的QA对中提取所有问题
- 使用 `GENERALIZE_PROMPT_TPL` 为每个问题生成5个同义问题
- 保存到 `expand_qa_pair.json`（格式：每行一个JSON，包含 `unique_id` 和 `raw_resp`）
- 原项目使用 `gen_qa` 函数的 `expand=True` 模式

**参考代码片段**:
```python
# 原项目 run.py:213-223
question_docs = []
fd = open(QA_PATH, "r")
idx = 0
for line in fd:
    info = json.loads(line)
    resp = json.loads(info["raw_resp"])
    for qa in resp:
        question_docs.append(Document(page_content=qa["question"], metadata={"unique_id": str(idx)}))
        idx += 1
expand_qa_dict = gen_qa(question_docs, GENERALIZE_PROMPT_TPL, OUTPUT_PATH, expand=True)
```

#### 步骤3: 答案匹配和扩充

**原项目代码位置**: 
- 问题改写结果解析：`run.py:237-243` (解析expand_qa_pair.json)
- QA对扩充：`run.py:246-266` (合并原始问题和改写问题，匹配答案)

**实现要点**:
- 将原始问题和改写问题合并
- 每个改写问题匹配原始答案
- 生成扩充后的QA对列表
- 使用MD5生成unique_id

**参考代码片段**:
```python
# 原项目 run.py:237-266
expand_qa_pairs = {}
for unique_id, info in expand_qa_dict.items():
    question = info["unique_id"]
    expand_questions = info["raw_resp"]
    expand_questions = expand_questions.split("\n")
    expand_questions = [re.sub(r'^\d[.. ]', '', item).strip() for item in expand_questions]
    expand_qa_pairs[question] = expand_questions

for unique_id, info in qa_dict.items():
    resp = json.loads(info["raw_resp"])
    for qa in resp:
        question = qa["question"].strip()
        answer = qa["answer"].strip()
        expand_questions = [question] + expand_qa_pairs[question]
        for query in expand_questions:
            unique_id = hashlib.md5(query.encode('utf-8')).hexdigest()
            item = {"unique_id": unique_id, "question": query, "answer": answer}
            # 随机分配到训练集或测试集
```

#### 步骤4: 训练/测试集切分

**原项目代码位置**: 
- 切分逻辑：`run.py:263-266` (随机切分，90%训练集，10%测试集)
- 随机种子设置：`run.py:23` (random.seed(42))

**实现要点**:
- 随机切分比例：90% 训练集，10% 测试集
- 使用固定随机种子（42）确保可复现

**参考代码片段**:
```python
# 原项目 run.py:263-266
if random.random() < 0.9:
    train_qa_pairs.append(item)
else:
    test_qa_pairs.append(item)
```

#### 步骤5: 测试集关键词提取

**原项目代码位置**: 
- Prompt模板：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py:75-104` (KEYWORDS_PROMPT_TPL)
- 唯一答案提取：`run.py:271-274` (提取测试集中所有唯一的答案)
- 关键词生成：`run.py:277` (使用gen_qa函数生成关键词)
- 关键词映射：`run.py:280-288` (建立答案到关键词的映射，**注意bug**)

**实现要点**:
- 提取测试集中所有**唯一的答案**
- 使用 `KEYWORDS_PROMPT_TPL` 为每个答案提取关键词
- 建立答案到关键词的映射
- 将关键词添加到测试集的每个QA对中（`keywords` 字段）
- **重要**：修正原项目的bug（第283行变量名写错 `kewyords` 应该是 `keywords`）

**参考代码片段**:
```python
# 原项目 run.py:271-288
unique_test_answers = set([item["answer"] for item in test_qa_pairs])
test_answer_docs = []
for idx, answer in enumerate(unique_test_answers):
    test_answer_docs.append(Document(page_content=answer, metadata={"unique_id": str(idx)}))

keywords_dict = gen_qa(test_answer_docs, KEYWORDS_PROMPT_TPL, TEST_KEYWORDS_PATH, expand=True)

keywords_mapping = {}
for unique_id, info in keywords_dict.items():
    keywords = info["raw_resp"].split(",")
    # 原项目bug：变量名写错，应该是 keywords 而不是 kewyords
    kewyords = [item for item in keywords if item not in ["无", "Model 3"]]  # BUG!
    keywords_mapping[info["unique_id"]] = kewyords  # BUG!

for info in test_qa_pairs:
    keywords = keywords_mapping[info["answer"]]  # 这里使用答案作为key
    info["keywords"] = keywords
```

**注意**：原项目代码第283-284行有bug，变量名写错了。新项目实现时需要修正。

#### 步骤6: 负样本添加

**原项目代码位置**: 
- 负样本读取：`run.py:291-293` (从raw_general_chats.txt读取)
- 负样本分配：`run.py:295-311` (95%训练集，5%测试集)

**实现要点**:
- 从 `raw_general_chats.txt` 读取通用对话数据
- 95% 加入训练集，5% 加入测试集
- 答案设置为"无答案"
- 测试集的负样本 `keywords` 字段为空数组

**参考代码片段**:
```python
# 原项目 run.py:291-311
chats_data = open(CHATS_PATH).readlines()
chats_data = [item.strip() for item in chats_data]

random.seed(42)
for line in chats_data:
    if random.random() < 0.95:
        train_qa_pairs.append({
            "unique_id": hashlib.md5(line.encode('utf-8')).hexdigest(),
            "question": line,
            "answer": "无答案"
        })
    else:
        test_qa_pairs.append({
            "unique_id": hashlib.md5(line.encode('utf-8')).hexdigest(),
            "question": line,
            "answer": "无答案",
            "keywords": []
        })
```

#### 步骤7: 输出最终数据

**原项目代码位置**: 
- 训练集输出：`run.py:313-317` (写入train_qa_pair.json)
- 测试集输出：`run.py:319-323` (写入test_qa_pair.json)

**实现要点**:
- 训练集：`train_qa_pair.json`（JSON数组格式）
- 测试集：`test_qa_pair.json`（JSON数组格式，包含 `keywords` 字段）
- 关键词映射：`test_keywords_pair.json`（每行一个JSON，在步骤5中已生成）

**参考代码片段**:
```python
# 原项目 run.py:313-323
random.seed(42)
with open(TRAIN_PATH, "w") as fd:
    random.shuffle(train_qa_pairs)
    fd.write(json.dumps(train_qa_pairs, ensure_ascii=False, indent=2))
    print("训练集已写入:", TRAIN_PATH, len(train_qa_pairs))

random.seed(42)
with open(TEST_PATH, "w") as fd:
    random.shuffle(test_qa_pairs)
    fd.write(json.dumps(test_qa_pairs, ensure_ascii=False, indent=2))
    print("测试集已写入:", TEST_PATH, len(test_qa_pairs))
```

### 4. 后续阶段文件生成参考（不在本阶段实现）

#### train_data.json 生成

**原项目代码位置**: `/remote-home/share/liangZhang/EVRAG/generate_sft_data.py:48-69`

**功能说明**:
- 从 `train_qa_pair.json` 读取QA对
- 使用RAG系统（BM25 + Milvus + Reranker）检索上下文
- 生成包含 `query`, `context`, `response`, `merged_docs` 的数据
- 用于后续生成微调训练数据

**参考代码片段**:
```python
# 原项目 generate_sft_data.py:48-66
fd = open("data/qa_pairs/train_qa_pair.json")
test_qa_pairs = json.load(fd)
output_handler = open("data/qa_pairs/train_data.json", "w")
for item in tqdm(test_qa_pairs):
    query = item["question"].strip()
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    ranked_docs = qwen3_reranker.rank(query, merged_docs, topk=5)
    context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
    response = request_chat(query, context)
    answer = post_processing(response, ranked_docs)
    context = [q.page_content for q in ranked_docs]
    all_docs = [q.page_content for q in merged_docs]
    info = {"query": query, "context": context, "response": response, "merged_docs": all_docs}
    output_handler.write(json.dumps(info, ensure_ascii=False) + '\n')
```

#### test_qa_pair_verify.json 和 test_qa_pair_pred.json 生成

**原项目代码位置**: `/remote-home/share/liangZhang/EVRAG/final_score.py:133-159`

**功能说明**:
- `test_qa_pair_verify.json`: 从 `test_qa_pair.json` 筛选的验证集（手动筛选或按比例筛选）
- `test_qa_pair_pred.json`: 对验证集进行RAG预测，包含预测答案、引用页面等
- 用于评估RAG系统性能

**参考代码片段**:
```python
# 原项目 final_score.py:133-159
fd = open("data/qa_pairs/test_qa_pair_verify.json")
test_qa_pairs = json.load(fd)
result = []
for item in test_qa_pairs:
    query = item["question"].strip()
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=BM25_RETRIEVE_SIZE)
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=MILVUS_RETRIEVE_SIZE)
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=RERANK_SIZE)
    context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
    response = request_chat(query, context)
    answer = post_processing(response, ranked_docs)
    item["pred"] = answer  # 包含answer, cite_pages, related_images等
    item["context"] = context
    result.append(item)

with open("data/qa_pairs/test_qa_pair_pred.json", "w") as fw:
    fw.write(json.dumps(result, ensure_ascii=False, indent=4))
```

### 3. 数据格式

#### 输入格式 (`qa_pair.json`)
```json
{"unique_id": "...", "raw_resp": "[{\"question\": \"...\", \"answer\": \"...\"}, ...]"}
```

#### 中间格式 (`expand_qa_pair.json`)
```json
{"unique_id": "问题文本", "raw_resp": "1. 改写问题1\n2. 改写问题2\n..."}
```

#### 输出格式 (`train_qa_pair.json`)
```json
[
  {
    "unique_id": "...",
    "question": "...",
    "answer": "..."
  },
  ...
]
```

#### 输出格式 (`test_qa_pair.json`)
```json
[
  {
    "unique_id": "...",
    "question": "...",
    "answer": "...",
    "keywords": ["关键词1", "关键词2", ...]
  },
  ...
]
```

#### 输出格式 (`test_keywords_pair.json`)
```json
{"unique_id": "答案文本", "raw_resp": "关键词1,关键词2,关键词3"}
```

### 4. 在 main.py 中添加命令

**命令**: `process-qa`

**参数**:
- `--qa-pair-path`: QA对文件路径（默认：`data/qa_pairs/qa_pair.json`）
- `--output-dir`: 输出目录（默认：`data/qa_pairs`）
- `--negative-samples-path`: 负样本文件路径（默认：`data/ut/raw_general_chats.txt`）
- `--quality-threshold`: 质量打分阈值（默认：3）
- `--train-ratio`: 训练集比例（默认：0.9）
- `--use-deepseek`: 是否使用Deepseek进行质量打分（默认：True）
- `--skip-quality-scoring`: 跳过质量打分步骤（如果已打分）
- `--skip-question-rewriting`: 跳过问题改写步骤（如果已改写）

### 5. 负样本文件获取建议

**文件路径**: `data/ut/raw_general_chats.txt`

**文件格式**: 每行一个对话句子，UTF-8编码

**获取方式**（在文档中说明）:
1. **从公开对话数据集提取**：从通用对话数据集（如中文对话数据集）中提取与汽车领域无关的对话句子
2. **从用户日志提取**：从实际用户对话日志中提取无关问题
3. **手动收集**：收集一些常见的通用对话问题（如"你好"、"今天天气怎么样"等）
4. **使用现有文件**：如果已有 `raw_general_chats.txt`，直接使用

**示例内容**（参考原项目）:
```
介绍一下北京的著名景点
导航去第一个
成都的市花是什么花
谢娜老公是谁
放首她的歌
...
```

## 三、实现细节

### 1. QA质量打分

**原项目参考**: 
- LLM调用函数：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py:145-169` (chat函数)
- 新项目已有实现：`src/evrag/gen_qa/generator.py:444-482` (score_qa_quality方法)

**实现要点**:
- 使用 `QAGenerator.score_qa_quality()` 方法
- 并发处理，使用 `ThreadPoolExecutor`（参考原项目 `run.py:174`）
- 支持断点续传（保存checkpoint，参考原项目 `run.py:172-203` 的checkpoint逻辑）

**原项目并发处理参考**:
```python
# 原项目 run.py:174-203
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {...}
    for unique_id in tqdm(futures):
        future = futures[unique_id]
        result = future.result()
        # 处理结果并保存checkpoint
```

### 2. 问题改写

**原项目参考**: 
- 核心函数：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py:171-203` (gen_qa函数)
- 结果解析：`run.py:237-243` (解析改写结果，去除序号)

**实现要点**:
- 使用 `QAGenerator.generalize_questions()` 方法（新项目需要实现，可参考原项目gen_qa函数）
- 解析改写结果（去除序号，清理格式）
- 保存中间结果到 `expand_qa_pair.json`

**原项目结果解析参考**:
```python
# 原项目 run.py:237-243
expand_qa_pairs = {}
for unique_id, info in expand_qa_dict.items():
    question = info["unique_id"]
    expand_questions = info["raw_resp"]
    expand_questions = expand_questions.split("\n")
    # 去除序号：1., 2., 3. 等格式
    expand_questions = [re.sub(r'^\d[.. ]', '', item).strip() for item in expand_questions]
    expand_qa_pairs[question] = expand_questions
```

### 3. 关键词提取映射逻辑

**原项目参考**: 
- 唯一答案提取：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py:271-274`
- 关键词映射：`run.py:280-288` (**注意bug**)

**实现要点**:
- 提取测试集中所有**唯一的答案**（使用set去重）
- 为每个唯一答案提取关键词
- 建立答案到关键词的映射（使用答案文本作为key）
- 将关键词添加到测试集的每个QA对中
- **修正原项目bug**：变量名 `kewyords` 应该是 `keywords`

**原项目bug位置**:
```python
# 原项目 run.py:283-284 (有bug)
kewyords = [item for item in keywords if item not in ["无", "Model 3"]]  # BUG: 变量名写错
keywords_mapping[info["unique_id"]] = kewyords  # BUG: 使用了错误的变量名
```

**修正后的代码**:
```python
# 新项目应该这样写
keywords = [item for item in keywords if item not in ["无", "Model 3"]]
keywords_mapping[info["unique_id"]] = keywords
```

### 4. 错误处理

**原项目参考**: 
- LLM调用重试：`/remote-home/share/liangZhang/EVRAG/src/gen_qa/run.py:145-169` (chat函数的max_retry机制)
- 文件锁：`run.py:173, 194-202` (使用threading.Lock保证文件写入安全)

**实现要点**:
- LLM调用失败重试机制（参考原项目chat函数的max_retry参数）
- JSON解析错误处理（参考原项目gen_qa函数中的异常处理）
- 文件读写错误处理（参考原项目使用文件锁保证并发安全）

**原项目错误处理参考**:
```python
# 原项目 run.py:145-169 (LLM重试机制)
def chat(prompt, max_retry=3, debug=False, temperature=0.85, top_p=0.95):
    def do_chat(prompt):
        # LLM调用
        return completion.choices[0].message.content
    
    x = do_chat(prompt)
    while max_retry > 0:
        try:
            return do_chat(prompt)
        except Exception as e:
            max_retry -= 1
            sleep_seconds = random.randint(1, 4)
            if debug:
                print(f"{str(e)}, remain retry: {max_retry}, sleeping {sleep_seconds}s")
            time.sleep(sleep_seconds)
    return None

# 原项目 run.py:194-202 (文件锁)
file_lock.acquire()
try:
    with open(qa_ckpt_filename, 'a') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
except Exception as e:
    print(e)
finally:
    file_lock.release()
```

## 四、文件结构

### 新项目文件结构

```
src/evrag/gen_qa/
├── generator.py          # 已有，包含QA生成、问题改写、关键词提取、质量打分
│                         # 参考原项目：EVRAG/src/gen_qa/run.py (gen_qa函数)
├── qa_processor.py      # 新建，QA数据处理主模块
│                         # 参考原项目：EVRAG/src/gen_qa/run.py (主流程)
└── ...

main.py                  # 添加 process-qa 命令

data/qa_pairs/
├── qa_pair.json         # 输入：已生成的QA对
├── expand_qa_pair.json  # 中间：问题改写结果
├── train_qa_pair.json   # 输出：训练集
├── test_qa_pair.json    # 输出：测试集
└── test_keywords_pair.json  # 输出：测试集关键词映射

data/ut/
└── raw_general_chats.txt  # 负样本数据
```

### 原项目对应文件结构

```
EVRAG/src/gen_qa/
└── run.py               # 核心实现文件，包含所有处理逻辑

EVRAG/data/qa_pairs/
├── qa_pair.json         # 输入
├── expand_qa_pair.json  # 中间结果
├── train_qa_pair.json   # 输出
├── test_qa_pair.json    # 输出
└── test_keywords_pair.json  # 输出

EVRAG/data/ut/
└── raw_general_chats.txt  # 负样本数据
```

## 五、配置项

在 `config.py` 中添加：
- `qa_quality_threshold`: QA质量打分阈值（默认：3）
- `train_test_split_ratio`: 训练/测试集切分比例（默认：0.9）
- `negative_samples_train_ratio`: 负样本训练集比例（默认：0.95）
- `raw_general_chats_path`: 负样本文件路径

## 六、测试和验证

1. 验证QA质量打分功能
2. 验证问题改写功能
3. 验证训练/测试集切分比例
4. 验证关键词提取和匹配（修正原项目bug）
5. 验证负样本添加
6. 验证输出文件格式

## 七、关键函数和类参考

### 原项目核心函数

1. **gen_qa函数** (`run.py:171-203`)
   - 功能：通用的QA生成函数，支持QA生成、问题改写、关键词提取
   - 参数：`splitted_docs`, `prompt_tmpl`, `qa_ckpt_filename`, `expand=False`
   - 特点：支持并发处理、断点续传、文件锁

2. **chat函数** (`run.py:145-169`)
   - 功能：LLM调用封装，包含重试机制
   - 参数：`prompt`, `max_retry=3`, `debug=False`, `temperature=0.85`, `top_p=0.95`
   - 特点：自动重试、随机延迟

3. **build_qa_prompt函数** (`run.py:140-142`)
   - 功能：构建prompt，替换模板中的占位符
   - 参数：`prompt_tmpl`, `text`

### 新项目已有实现

1. **QAGenerator类** (`src/evrag/gen_qa/generator.py`)
   - `score_qa_quality()`: QA质量打分（已实现）
   - `extract_keywords()`: 关键词提取（已实现）
   - `generate_qa_pairs()`: QA对生成（已实现）
   - 需要实现：`generalize_questions()` 方法（问题改写）

### 新项目需要实现

1. **QAProcessor类** (`src/evrag/gen_qa/qa_processor.py`)
   - 整合所有处理步骤
   - 参考原项目 `run.py` 的主流程（`if __name__ == "__main__"` 部分）

## 八、Git Flow 工作流规划

### 1. 分支策略

#### 当前分支状态
- **基础分支**: `develop`
- **功能分支**: `feature/data-preparation`（如果已存在）或创建新分支

#### 分支创建方案

**方案A：在现有feature/data-preparation分支上工作**（推荐）
```bash
# 切换到feature/data-preparation分支
git checkout feature/data-preparation
git pull origin feature/data-preparation

# 如果分支不存在，从develop创建
git checkout -b feature/data-preparation develop
```

**方案B：创建子功能分支**（如果feature/data-preparation已合并到develop）
```bash
# 从develop创建QA处理子功能分支
git checkout develop
git pull origin develop
git checkout -b feature/qa-processing develop
```

### 2. 提交策略

#### 2.1 提交拆分原则

按照功能模块和实现步骤进行原子性提交，每个提交应该：
- 完成一个独立的功能点
- 包含相关的测试代码
- 能够独立编译和运行
- 提交信息清晰明确

#### 2.2 建议的提交顺序

**提交1：创建QA处理器模块框架**
```bash
git add src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): add QAProcessor class framework

- Add QAProcessor class with basic structure
- Add method stubs for all processing steps
- Reference: EVRAG/src/gen_qa/run.py"
```

**提交2：实现QA质量打分功能**
```bash
git add src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): implement QA quality scoring

- Add quality scoring using QAGenerator.score_qa_quality()
- Add filtering logic for low-quality QA pairs
- Filter answers containing '无法准确' or '未提及'
- Reference: EVRAG/src/gen_qa/run.py:253"
```

**提交3：实现问题改写功能**
```bash
git add src/evrag/gen_qa/generator.py src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): implement question rewriting

- Add generalize_questions() method to QAGenerator
- Implement question expansion using GENERALIZE_PROMPT_TPL
- Parse and clean rewritten questions
- Save to expand_qa_pair.json
- Reference: EVRAG/src/gen_qa/run.py:213-243"
```

**提交4：实现答案匹配和扩充**
```bash
git add src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): implement answer matching and expansion

- Merge original and rewritten questions
- Match answers to expanded questions
- Generate expanded QA pairs with MD5 unique_id
- Reference: EVRAG/src/gen_qa/run.py:246-266"
```

**提交5：实现训练/测试集切分**
```bash
git add src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): implement train/test split

- Split QA pairs into train (90%) and test (10%) sets
- Use fixed random seed (42) for reproducibility
- Reference: EVRAG/src/gen_qa/run.py:263-266"
```

**提交6：实现关键词提取**
```bash
git add src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): implement keyword extraction for test set

- Extract unique answers from test set
- Generate keywords using KEYWORDS_PROMPT_TPL
- Build answer-to-keywords mapping
- Fix bug: correct variable name from 'kewyords' to 'keywords'
- Add keywords to test QA pairs
- Reference: EVRAG/src/gen_qa/run.py:271-288"
```

**提交7：实现负样本添加**
```bash
git add src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): add negative samples from raw_general_chats.txt

- Load negative samples from raw_general_chats.txt
- Distribute 95% to train set, 5% to test set
- Set answer to '无答案' for negative samples
- Set empty keywords array for test negative samples
- Reference: EVRAG/src/gen_qa/run.py:291-311"
```

**提交8：实现最终数据输出**
```bash
git add src/evrag/gen_qa/qa_processor.py
git commit -m "feat(gen_qa): implement final data output

- Write train_qa_pair.json (JSON array format)
- Write test_qa_pair.json (JSON array with keywords)
- Write test_keywords_pair.json (line-by-line JSON)
- Shuffle data with fixed random seed
- Reference: EVRAG/src/gen_qa/run.py:313-323"
```

**提交9：添加main.py命令**
```bash
git add main.py
git commit -m "feat(cli): add process-qa command to main.py

- Add 'process-qa' command with all required parameters
- Support skip flags for quality scoring and question rewriting
- Add progress tracking and error handling
- Reference: EVRAG/src/gen_qa/run.py main flow"
```

**提交10：添加单元测试**
```bash
git add tests/unit/test_qa_processor.py
git commit -m "test(gen_qa): add unit tests for QAProcessor

- Test QA quality scoring
- Test question rewriting
- Test train/test split
- Test keyword extraction
- Test negative sample addition
- Test data output formats"
```

**提交11：更新配置**
```bash
git add config/config.yaml src/evrag/config.py
git commit -m "chore(config): add QA processing configuration

- Add qa_quality_threshold (default: 3)
- Add train_test_split_ratio (default: 0.9)
- Add negative_samples_train_ratio (default: 0.95)
- Add raw_general_chats_path configuration"
```

**提交12：更新文档**
```bash
git add docs/PHASE2_DATA_PREPARATION.md
git commit -m "docs(phase2): update QA processing documentation

- Add QA data processing workflow
- Add file generation flow explanation
- Add reference to old project code paths
- Add negative sample file acquisition methods
- Add usage examples"
```

### 3. 提交信息格式

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

#### 格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type类型
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `test`: 测试相关
- `chore`: 配置或工具变动
- `refactor`: 代码重构

#### Scope范围
- `gen_qa`: QA生成模块
- `cli`: 命令行接口
- `config`: 配置相关
- `phase2`: 阶段二相关

#### 示例
```bash
# 新功能
git commit -m "feat(gen_qa): implement QA quality scoring"

# 修复bug
git commit -m "fix(gen_qa): correct variable name from kewyords to keywords"

# 文档更新
git commit -m "docs(phase2): add QA processing workflow documentation"

# 测试
git commit -m "test(gen_qa): add unit tests for QAProcessor"
```

### 4. 代码审查检查清单

在提交PR前，确保：

- [ ] 代码遵循PEP 8规范
- [ ] 所有函数都有类型提示和文档字符串
- [ ] 单元测试覆盖率 > 80%
- [ ] 所有测试通过
- [ ] 代码通过ruff和mypy检查
- [ ] 提交信息符合Conventional Commits规范
- [ ] 已更新相关文档
- [ ] 已添加原项目代码路径参考（如适用）

### 5. 合并流程

#### 5.1 完成功能开发后

```bash
# 1. 确保所有更改已提交
git status

# 2. 运行测试套件
pytest tests/ -v
pytest tests/unit/test_qa_processor.py -v

# 3. 运行代码质量检查
bash scripts/check_code_quality.sh

# 4. 确保代码格式化
ruff format src/ tests/

# 5. 确保通过类型检查
mypy src/evrag/gen_qa/qa_processor.py
```

#### 5.2 合并到develop分支

**如果使用git flow**:
```bash
# 完成功能分支
git flow feature finish qa-processing

# 或手动合并
git checkout develop
git pull origin develop
git merge --no-ff feature/qa-processing
git push origin develop
```

**如果直接在feature/data-preparation分支上工作**:
```bash
# 确保develop是最新的
git checkout develop
git pull origin develop

# 合并feature/data-preparation到develop
git merge --no-ff feature/data-preparation
git push origin develop
```

#### 5.3 创建Pull Request（如果使用GitHub/GitLab）

PR标题格式：
```
feat(gen_qa): implement QA data processing pipeline
```

PR描述模板：
```markdown
## 功能描述
实现完整的QA数据处理流程，包括质量打分、问题改写、训练/测试集切分、关键词提取和负样本添加。

## 实现内容
- [x] QA质量打分和过滤
- [x] 问题改写（Query Rewriting）
- [x] 答案匹配和扩充
- [x] 训练/测试集切分
- [x] 测试集关键词提取
- [x] 负样本添加
- [x] 最终数据输出
- [x] main.py命令集成
- [x] 单元测试
- [x] 文档更新

## 测试
- [x] 单元测试通过
- [x] 代码质量检查通过
- [x] 类型检查通过

## 参考
- 原项目代码路径：EVRAG/src/gen_qa/run.py
- 相关文档：docs/QA_DATA_PROCESSING_PLAN.md
```

### 6. 测试要求

#### 6.1 单元测试

创建 `tests/unit/test_qa_processor.py`，测试：
- QA质量打分功能
- 问题改写功能
- 训练/测试集切分
- 关键词提取和映射
- 负样本添加
- 数据输出格式

#### 6.2 集成测试

创建 `tests/integration/test_qa_processing_pipeline.py`，测试：
- 完整的QA处理流程
- 端到端数据生成
- 文件格式验证

#### 6.3 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/test_qa_processor.py -v

# 运行集成测试
pytest tests/integration/test_qa_processing_pipeline.py -v

# 生成覆盖率报告
pytest tests/ --cov=src/evrag/gen_qa --cov-report=html
```

### 7. 回滚策略

如果合并后发现问题：

```bash
# 1. 创建hotfix分支
git checkout develop
git checkout -b hotfix/fix-qa-processing-issue

# 2. 修复问题
# ... 进行修复 ...

# 3. 提交修复
git commit -m "fix(gen_qa): fix issue description"

# 4. 合并回develop
git checkout develop
git merge --no-ff hotfix/fix-qa-processing-issue
git push origin develop

# 5. 删除hotfix分支
git branch -d hotfix/fix-qa-processing-issue
```

### 8. 版本标签（可选）

如果这是一个重要的里程碑，可以打标签：

```bash
# 创建标签
git tag -a v0.2.0-qa-processing -m "feat: QA data processing pipeline"

# 推送标签
git push origin v0.2.0-qa-processing
```

## 九、文档更新

更新 `docs/PHASE2_DATA_PREPARATION.md`，添加：
- QA数据处理流程说明
- 文件生成流程说明（包括后续阶段的文件）
- 原项目代码路径参考
- 负样本文件获取方法
- 使用示例

