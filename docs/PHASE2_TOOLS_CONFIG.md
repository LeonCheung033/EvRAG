# 阶段二：工具配置和系统优化

## 概述

本文档说明阶段二中涉及的所有工具配置和系统优化，包括GPU设备分配、LLM服务配置和性能监控等。

## 目录

1. [GPU设备分配](#gpu设备分配)
2. [LLM服务配置](#llm服务配置)
3. [性能监控](#性能监控)

## GPU设备分配

### 当前GPU状态

系统有 **8张 NVIDIA RTX A6000 GPU**，每张显存 **49GB**。

### GPU使用情况

| GPU ID | 显存使用 | 状态 | 分配用途 |
|--------|---------|------|---------|
| GPU 0  | 370MB   | 空闲 | **语义切分服务** |
| GPU 1  | 5MB     | 空闲 | **Milvus检索器** |
| GPU 2  | 5MB     | 空闲 | **Reranker** |
| GPU 3  | 5MB     | 空闲 | **vLLM服务（3卡并行）** |
| GPU 4  | 28GB    | 占用 | 其他任务（暂不分配） |
| GPU 5  | 5MB     | 空闲 | **vLLM服务（3卡并行）** |
| GPU 6  | 5MB     | 空闲 | **vLLM服务（3卡并行）** |
| GPU 7  | 28GB    | 占用 | 其他任务（暂不分配） |

### GPU分配方案

#### 1. 语义切分服务（Semantic Chunk Service）

- **GPU**: GPU 0
- **模型**: M3E-small
- **显存需求**: ~1GB
- **配置项**: `semantic_chunk_gpu_id: 0`
- **服务端口**: 6000

#### 2. Milvus检索器（Milvus Retriever）

- **GPU**: GPU 1
- **模型**: BGE-M3
- **显存需求**: ~2-3GB
- **配置项**: `milvus_retriever_gpu_id: 1`
- **说明**: 用于生成文档和查询的embedding向量

#### 3. Reranker（重排序器）

- **GPU**: GPU 2
- **模型**: BGE-Reranker-v2-m3
- **显存需求**: ~2-3GB
- **配置项**: `reranker_gpu_id: 2`
- **说明**: 对检索结果进行重排序

#### 4. vLLM服务（LLM推理服务）

- **GPU**: GPU 3, 5, 6（3卡tensor parallelism）
- **模型**: Qwen3-8B（微调后）
- **显存需求**: 每卡约15-20GB（3卡并行）
- **配置项**: `vllm_gpu_ids: [3, 5, 6]`
- **服务端口**: 8000
- **说明**: 使用tensor parallelism将模型切分到3张GPU上

### 配置方法

在 `config/config.yaml` 中配置：

```yaml
# GPU设备分配配置
semantic_chunk_gpu_id: 0      # 语义切分服务使用的GPU ID
milvus_retriever_gpu_id: 1    # Milvus检索器使用的GPU ID
reranker_gpu_id: 2            # Reranker使用的GPU ID
vllm_gpu_ids: [3, 5, 6]       # vLLM服务使用的GPU ID列表（tensor parallelism）
```

### 代码实现

#### 语义切分服务

**文件**: `src/evrag/server/semantic_chunk_server.py`

- 已支持通过配置指定GPU设备ID
- 启动时会显示使用的设备：`✓ 模型部署在设备: cuda:0`

#### Reranker

**文件**: `src/evrag/reranker/bge_reranker.py`

- 已支持通过配置指定GPU设备ID
- 使用 `model.to(f"cuda:{device_id})` 加载到指定GPU
- 启动时会显示使用的设备

#### Milvus检索器

**文件**: `src/evrag/retriever/milvus_retriever.py`

- 已支持通过配置指定GPU设备ID
- 使用 `torch.cuda.set_device()` 设置默认GPU设备

#### 配置类

**文件**: `src/evrag/config.py`

- 添加了GPU设备ID配置字段
- 支持从YAML配置文件读取

### 使用方法

#### 启动语义切分服务

```bash
cd /remote-home/share/liangZhang/EvRAG
python -m src.evrag.server.semantic_chunk_server
```

**输出示例**:
```
✓ 语义切分模型已加载: /remote-home/share/liangZhang/EvRAG/models/m3e-small
✓ 模型部署在设备: cuda:0
```

#### 使用Reranker

```python
from src.evrag.reranker.bge_reranker import BGEReranker

# 自动使用配置中的GPU 2
reranker = BGEReranker()
```

#### 使用Milvus检索器

```python
from src.evrag.retriever.milvus_retriever import MilvusRetriever

# 自动使用配置中的GPU 1
retriever = MilvusRetriever(docs=documents)
```

#### 启动vLLM服务

```bash
# 使用配置中的GPU 3,5,6
CUDA_VISIBLE_DEVICES=3,5,6 vllm serve /path/to/model \
    --tensor-parallel-size 3 \
    --port 8000
```

### 验证GPU分配

启动服务后，可以使用以下命令验证GPU使用情况：

```bash
# 实时监控GPU使用
nvidia-smi -l 1

# 查看特定GPU
nvidia-smi | grep -A 2 "GPU 0"  # 语义切分服务
nvidia-smi | grep -A 2 "GPU 1"  # Milvus检索器
nvidia-smi | grep -A 2 "GPU 2"  # Reranker
nvidia-smi | grep -E "GPU [356]"  # vLLM服务
```

### 修改GPU分配

如果需要修改GPU分配，只需编辑 `config/config.yaml`：

```yaml
# 例如，将语义切分服务改为使用GPU 1
semantic_chunk_gpu_id: 1

# 将vLLM改为使用GPU 0,1,2（3卡并行）
vllm_gpu_ids: [0, 1, 2]
```

**注意**: 修改配置后需要重启相应的服务。

### 显存使用估算

| 服务 | 模型 | 显存使用（估算） |
|------|------|----------------|
| 语义切分 | M3E-small | ~1GB |
| Milvus检索 | BGE-M3 | ~2-3GB |
| Reranker | BGE-Reranker-v2-m3 | ~2-3GB |
| vLLM（单卡） | Qwen3-8B | ~15-20GB |
| vLLM（3卡并行） | Qwen3-8B | 每卡~5-7GB |

### GPU故障排查

#### 问题1: GPU显存不足

**症状**: 模型加载失败，提示OOM（Out of Memory）

**解决方案**:
- 检查是否有其他进程占用GPU
- 减小batch size
- 使用更小的模型或量化模型

#### 问题2: GPU设备ID不存在

**症状**: `RuntimeError: CUDA error: invalid device ordinal`

**解决方案**:
- 检查 `nvidia-smi` 确认可用GPU数量
- 修改 `config.yaml` 中的GPU ID配置

#### 问题3: 服务使用了错误的GPU

**症状**: 服务启动后，模型加载在错误的GPU上

**解决方案**:
- 检查配置文件中的GPU ID设置
- 重启服务
- 查看服务启动日志确认使用的设备

## LLM服务配置

### 概述

EvRAG系统支持多个LLM服务商，采用可插拔设计，可以在配置文件中灵活配置不同的服务商。

### 支持的服务商

1. **本地vLLM服务** - 用于问答、推理等（微调后的Qwen3-8B模型）
2. **豆包API** - 用于文档清洗等任务
3. **Deepseek API** - 可选，用于其他任务（如QA生成）

### 配置文件设置

在 `config/config.yaml` 中配置各个服务商：

```yaml
# 本地vLLM服务配置（用于问答、推理等）
local_llm_model_name: "qwen3_lora_sft_int4"  # 本地微调后的模型名称
local_llm_base_url: "http://localhost:8000/v1"  # 本地vLLM服务地址
local_llm_api_key: "EMPTY"  # 本地服务通常不需要API key

# 豆包API配置（用于文档清洗等）
doubao_api_key: "your_doubao_api_key_here"  # 豆包API密钥
doubao_base_url: "https://ark.cn-beijing.volces.com/api/v3"  # 豆包API地址
doubao_model_name: "doubao-seed-1-6-lite-251015"  # 豆包推理接入点ID

# Deepseek API配置（可选，用于QA生成等）
deepseek_api_key: "your_deepseek_api_key_here"  # Deepseek API密钥
deepseek_base_url: "https://api.deepseek.com"  # Deepseek API地址
deepseek_model_name: "deepseek-chat"  # Deepseek模型名称
```

### 使用方式

#### 1. 文档清洗（使用豆包API）

文档清洗自动使用豆包API，无需额外配置：

```python
from src.evrag.client import OpenAIClient, CleanClient

# 自动使用配置中的豆包API
llm_client = OpenAIClient(service="doubao")
clean_client = CleanClient(llm_client)
clean_docs = clean_client.clean_documents(raw_docs)
```

#### 2. 问答/推理（使用本地vLLM）

问答功能使用本地vLLM服务：

```python
from src.evrag.client import LocalLLMClient, ChatClient

# 自动使用配置中的本地vLLM
llm_client = LocalLLMClient()
chat_client = ChatClient(llm_client)
answer = chat_client.chat(query="问题", context="上下文")
```

#### 3. QA生成（使用Deepseek API）

QA生成使用Deepseek API：

```python
from src.evrag.client import OpenAIClient
from src.evrag.gen_qa import QAGenerator

# 使用Deepseek API
llm_client = OpenAIClient(service="deepseek")
generator = QAGenerator(llm_client)
qa_dict = generator.generate_qa_from_documents(documents)
```

#### 4. 自定义服务

如果需要使用其他OpenAI兼容的API服务：

```python
from src.evrag.client import OpenAIClient

# 自定义服务
llm_client = OpenAIClient(
    service="custom",
    api_key="your_api_key",
    base_url="https://api.example.com/v1",
    model="your-model-name"
)
```

### 代码中的使用

#### main.py 中的使用

- **文档清洗**：自动使用豆包API
  ```python
  llm_client = OpenAIClient(service="doubao")
  ```

- **问答推理**：使用本地vLLM
  ```python
  llm_client = LocalLLMClient()
  ```

- **QA生成**：使用Deepseek API
  ```python
  llm_client = OpenAIClient(service="deepseek")
  ```

### 配置优先级

#### OpenAIClient

1. 传入参数（`api_key`, `base_url`, `model`）
2. 配置文件中的对应服务商配置（`doubao_*`, `deepseek_*`）

#### LocalLLMClient

1. 传入参数（`api_key`, `base_url`, `model`）
2. 配置文件中的 `local_llm_*` 配置
3. 兼容旧配置 `llm_*`（向后兼容）

### 注意事项

1. **API密钥安全**：
   - 建议将API密钥存储在配置文件中
   - 不要将包含API密钥的配置文件提交到Git仓库
   - 可以使用环境变量，但需要在代码中明确处理

2. **服务可用性**：
   - 确保对应的服务已启动（本地vLLM）或可访问（远程API）
   - 文档清洗需要豆包API可用
   - 问答功能需要本地vLLM服务运行
   - QA生成需要Deepseek API可用

3. **模型名称**：
   - 豆包：使用推理接入点ID（如 `doubao-seed-1-6-lite-251015`）
   - Deepseek：使用模型名称（如 `deepseek-chat`）
   - 本地vLLM：使用微调后的模型名称（如 `qwen3_lora_sft_int4`）

### LLM服务故障排查

#### 问题1：豆包API调用失败

**症状**：文档清洗时提示API key错误

**解决方案**：
1. 检查 `doubao_api_key` 是否在配置文件中正确设置
2. 确认API密钥有效且未过期
3. 检查 `doubao_base_url` 是否正确

#### 问题2：本地vLLM连接失败

**症状**：问答时提示连接错误

**解决方案**：
1. 确认vLLM服务已启动：`curl http://localhost:8000/health`
2. 检查 `local_llm_base_url` 配置是否正确
3. 确认模型名称 `local_llm_model_name` 与vLLM服务中的模型名称一致

#### 问题3：Deepseek API调用失败

**症状**：QA生成时提示API key错误

**解决方案**：
1. 检查 `deepseek_api_key` 是否在配置文件中正确设置
2. 确认API密钥有效且未过期
3. 检查 `deepseek_base_url` 是否正确

#### 问题4：服务商选择错误

**症状**：使用 `OpenAIClient(service="xxx")` 时报错

**解决方案**：
- 确保 `service` 参数为：`"doubao"`, `"deepseek"`, 或 `"custom"`
- 如果使用 `"custom"`，必须提供所有参数（`api_key`, `base_url`, `model`）

## 性能监控

### 概述

EvRAG系统在 `prepare-data` 和 `build-index` 两个命令中集成了性能监控功能，自动记录执行时间、CPU、内存、GPU等资源使用情况。

### 监控内容

#### 时间统计
- 总执行时间
- 每个步骤的耗时（PDF解析、文档清洗、文档切分、索引构建等）

#### 资源使用
- **CPU使用率**：进程CPU使用率、系统CPU使用率
- **内存使用**：进程内存（RSS、VMS）、系统内存使用率
- **GPU使用**（如果可用）：
  - GPU利用率
  - 显存使用情况
  - GPU温度

### 输出位置

性能报告保存在 `logs/performance/` 目录下，文件名格式：
- `prepare_data_YYYYMMDD_HHMMSS.json`
- `build_index_YYYYMMDD_HHMMSS.json`
- `gen_qa_YYYYMMDD_HHMMSS.json`

### 报告格式

#### JSON报告结构

```json
{
  "summary": {
    "task_name": "prepare_data",
    "start_time": "2025-11-24T12:00:00",
    "end_time": "2025-11-24T12:30:00",
    "total_duration_seconds": 1800.0,
    "total_duration_formatted": "30分0.00秒",
    "steps": [
      {
        "name": "PDF解析",
        "duration_seconds": 120.5,
        "duration_formatted": "2分0.50秒"
      },
      ...
    ],
    "step_summary": {
      "PDF解析": {
        "duration_seconds": 120.5,
        "duration_formatted": "2分0.50秒"
      },
      ...
    },
    "resource_usage": {
      "avg_cpu_percent": 45.2,
      "avg_memory_mb": 2048.5,
      "max_memory_mb": 3072.0,
      "gpu": {
        "gpu_0": {
          "name": "NVIDIA RTX A6000",
          "avg_utilization_percent": 85.3,
          "max_utilization_percent": 98.5,
          "avg_memory_used_mb": 10240.0,
          "max_memory_used_mb": 12288.0
        },
        ...
      }
    }
  },
  "detailed_samples": [
    {
      "timestamp": "2025-11-24T12:00:00",
      "label": "start",
      "elapsed_seconds": 0.0,
      "cpu_percent": 10.5,
      "memory_rss_mb": 512.0,
      ...
    },
    ...
  ]
}
```

### 使用示例

#### 执行命令

```bash
# 数据准备（自动记录性能）
python main.py prepare-data --config config/config.yaml

# 索引构建（自动记录性能）
python main.py build-index --config config/config.yaml

# QA生成（自动记录性能）
python main.py gen-qa --config config/config.yaml
```

#### 查看报告

执行完成后，会在控制台输出性能摘要，同时保存详细的JSON报告：

```
============================================================
性能监控报告: prepare_data
============================================================
总耗时: 30分0.00秒

步骤耗时:
  - PDF解析: 2分0.50秒
  - 文档清洗: 25分30.20秒
  - 文档切分: 2分29.30秒

资源使用:
  - 平均CPU使用率: 45.20%
  - 平均内存使用: 2048.50 MB
  - 峰值内存使用: 3072.00 MB

GPU使用:
  - NVIDIA RTX A6000 (gpu_0):
    - 平均利用率: 85.30%
    - 峰值利用率: 98.50%
    - 平均显存使用: 10240.00 MB
    - 峰值显存使用: 12288.00 MB
============================================================
```

### 依赖要求

#### 必需依赖
- `psutil` - 用于监控CPU和内存（如果未安装，会跳过CPU/内存监控）

#### 可选依赖
- `pynvml` - 用于监控GPU（如果未安装，会跳过GPU监控）

#### 安装依赖

```bash
# 安装psutil（推荐）
pip install psutil

# 安装pynvml（可选，用于GPU监控）
pip install nvidia-ml-py
```

### 数据分析

#### 分析性能报告

可以使用Python脚本分析性能报告：

```python
import json
from pathlib import Path

# 加载报告
report_path = Path("logs/performance/prepare_data_20251124_120000.json")
with open(report_path) as f:
    report = json.load(f)

summary = report["summary"]

# 分析总耗时
print(f"总耗时: {summary['total_duration_formatted']}")

# 分析各步骤耗时
for step_name, step_info in summary["step_summary"].items():
    print(f"{step_name}: {step_info['duration_formatted']}")

# 分析资源使用
resource = summary["resource_usage"]
print(f"平均CPU: {resource['avg_cpu_percent']:.2f}%")
print(f"峰值内存: {resource['max_memory_mb']:.2f} MB")

# 分析GPU使用
if "gpu" in resource:
    for gpu_id, gpu_info in resource["gpu"].items():
        print(f"{gpu_info['name']} ({gpu_id}):")
        print(f"  平均利用率: {gpu_info['avg_utilization_percent']:.2f}%")
        print(f"  平均显存使用: {gpu_info['avg_memory_used_mb']:.2f} MB")
```

#### 对比多次运行

可以收集多次运行的报告，进行对比分析：

```python
import json
from pathlib import Path
import pandas as pd

# 收集所有报告
reports_dir = Path("logs/performance")
reports = []

for report_file in reports_dir.glob("prepare_data_*.json"):
    with open(report_file) as f:
        report = json.load(f)
        summary = report["summary"]
        reports.append({
            "timestamp": summary["start_time"],
            "total_duration": summary["total_duration_seconds"],
            "avg_cpu": summary["resource_usage"]["avg_cpu_percent"],
            "max_memory_mb": summary["resource_usage"]["max_memory_mb"],
        })

# 转换为DataFrame进行分析
df = pd.DataFrame(reports)
print(df.describe())
```

### 优化建议

根据性能报告，可以：

1. **识别瓶颈**：查看哪个步骤耗时最长
2. **资源优化**：根据CPU/内存使用情况调整并发数
3. **GPU优化**：根据GPU利用率调整batch size或模型配置
4. **对比改进**：对比优化前后的性能报告，验证改进效果

### 注意事项

1. **性能开销**：监控本身会带来少量性能开销（通常<1%）
2. **采样频率**：资源采样在每个步骤开始和结束时进行
3. **GPU监控**：需要NVIDIA驱动和pynvml库支持
4. **报告大小**：详细样本数据可能较大，如果不需要可以只查看summary

## 完整配置示例

### config.yaml 完整配置

```yaml
# GPU设备分配配置
semantic_chunk_gpu_id: 0      # 语义切分服务使用的GPU ID
milvus_retriever_gpu_id: 1    # Milvus检索器使用的GPU ID
reranker_gpu_id: 2            # Reranker使用的GPU ID
vllm_gpu_ids: [3, 5, 6]       # vLLM服务使用的GPU ID列表（tensor parallelism）

# 本地vLLM服务（问答）
local_llm_model_name: "qwen3_lora_sft_int4"
local_llm_base_url: "http://localhost:8000/v1"
local_llm_api_key: "EMPTY"

# 豆包API（文档清洗）
doubao_api_key: "sk-xxxxxxxxxxxxx"
doubao_base_url: "https://ark.cn-beijing.volces.com/api/v3"
doubao_model_name: "doubao-seed-1-6-lite-251015"

# Deepseek API（QA生成）
deepseek_api_key: "sk-xxxxxxxxxxxxx"
deepseek_base_url: "https://api.deepseek.com"
deepseek_model_name: "deepseek-chat"
```

## 相关文档

- [阶段二数据准备开发文档](./PHASE2_DATA_PREPARATION.md) - 完整的数据准备和索引构建开发文档
- [阶段二数据修复总结](./PHASE2_DATA_FIXES_SUMMARY.md) - 本阶段所有修复和问题分析

