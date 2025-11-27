<!-- c4822b3f-1ee8-44f5-8a16-f46f66d8c0fc fe7bdac2-55d1-4b2e-9e86-8a8f64710223 -->
# 模型微调阶段详细实现计划（Phase 4）

## 一、分支规划和提交计划

### 1.1 Git分支操作

```bash
# 从develop分支创建feature分支
git checkout develop
git pull origin develop
git flow feature start model-finetuning

# 开发完成后合并
git flow feature finish model-finetuning
git push origin develop
```

### 1.2 提交计划

- **提交1**: 添加LLM微调配置和脚本
- **提交2**: 添加Reranker微调配置和脚本
- **提交3**: 实现训练监控和可视化模块
- **提交4**: 实现模型评估和对比功能
- **提交5**: 添加命令行接口和文档

## 二、LLM微调实现（使用LLaMA-Factory）

### 2.1 原项目代码参考路径

**核心配置文件**: `/remote-home/share/liangZhang/EVRAG/LLaMA-Factory-main/examples/train_lora/qwen3_lora_sft.yaml`

**训练脚本**: `/remote-home/share/liangZhang/EVRAG/LLaMA-Factory-main/train.sh`

**训练工作流**: `/remote-home/share/liangZhang/EVRAG/LLaMA-Factory-main/src/llamafactory/train/sft/workflow.py:40-127`

### 2.2 实现步骤

#### 步骤1: 创建LLM微调配置模块

**新项目文件**: `src/evrag/finetune/llm_finetuner.py`

**主要功能**:

- 配置LLaMA-Factory训练参数
- 准备SFT训练数据（转换格式）
- 执行训练命令
- 监控训练过程
- 保存训练日志和checkpoint

**参考代码片段**:

```python
# 原项目 qwen3_lora_sft.yaml:1-40
model_name_or_path: /root/autodl-tmp/RAG/models/Qwen3-8B/
stage: sft
finetuning_type: lora
lora_rank: 8
dataset: faq_summary
template: qwen3
cutoff_len: 4500
output_dir: saves/qwen3-8b/lora/sft
```

**新项目实现**:

- 使用LLaMA-Factory的CLI接口：`llamafactory-cli train <config_file>`
- 配置文件路径：`config/finetune/qwen3_lora_sft.yaml`
- 数据路径：`data/summary_data/train.json` 和 `data/summary_data/test.json`
- 输出路径：`models/finetuned/qwen3_lora_sft/`

#### 步骤2: 数据格式转换

**原项目代码位置**:

- LLaMA-Factory数据格式要求：参考 `LLaMA-Factory-main/src/llamafactory/data/dataset.py`

**实现要点**:

- 将 `summary_data/train.json` 转换为LLaMA-Factory所需格式
- 创建数据集配置文件：`config/datasets/faq_summary.yaml`
- 数据格式：`{"instruction": "...", "input": "", "output": "..."}`

**新项目实现**:

- 创建 `src/evrag/finetune/data_converter.py`
- 实现 `convert_sft_data_to_llamafactory` 方法
- 生成数据集配置YAML文件

#### 步骤3: 训练配置管理

**新项目文件**: `config/finetune/qwen3_lora_sft.yaml`

**配置参数**:

```yaml
model_name_or_path: models/Qwen3-8B/
stage: sft
finetuning_type: lora
lora_rank: 8
lora_target: all
dataset: faq_summary
template: qwen3
cutoff_len: 4500
output_dir: models/finetuned/qwen3_lora_sft/
logging_steps: 10
save_steps: 100
plot_loss: true
report_to: tensorboard
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

#### 步骤4: 训练执行脚本

**新项目文件**: `scripts/train_llm.sh`

**脚本内容**:

```bash
#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 执行训练
llamafactory-cli train config/finetune/qwen3_lora_sft.yaml
```

## 三、Reranker微调实现（使用RAG-Retrieval）

### 3.1 原项目代码参考路径

**核心配置文件**: `/remote-home/share/liangZhang/EVRAG/RAG-Retrieval/rag_retrieval/train/reranker/config/training_bert.yaml`

**训练脚本**: `/remote-home/share/liangZhang/EVRAG/RAG-Retrieval/rag_retrieval/train/reranker/train_reranker.py:104-279`

**训练启动脚本**: `/remote-home/share/liangZhang/EVRAG/RAG-Retrieval/rag_retrieval/train/reranker/train.sh`

### 3.2 实现步骤

#### 步骤1: 创建Reranker微调配置模块

**新项目文件**: `src/evrag/finetune/reranker_finetuner.py`

**主要功能**:

- 配置RAG-Retrieval训练参数
- 准备Reranker训练数据
- 执行训练命令
- 监控训练过程

**参考代码片段**:

```python
# 原项目 training_bert.yaml:1-41
model_name_or_path: "/root/autodl-tmp/RAG/models/BAAI/bge-reranker-v2-m3"
model_type: "bert_encoder"
train_dataset: "/root/autodl-tmp/RAG/data/rerank_data/train.json"
val_dataset: "/root/autodl-tmp/RAG/data/rerank_data/dev.json"
loss_type: "pointwise_bce"
output_dir: "./output/bert"
epochs: 1
lr: 5e-5
batch_size: 8
```

**新项目实现**:

- 配置文件路径：`config/finetune/reranker_training.yaml`
- 数据路径：`data/rerank_data/train.json`、`dev.json`、`test.json`
- 输出路径：`models/finetuned/bge_reranker/`

#### 步骤2: 训练配置管理

**新项目文件**: `config/finetune/reranker_training.yaml`

**配置参数**:

```yaml
model_name_or_path: models/BAAI/bge-reranker-v2-m3
model_type: bert_encoder
num_labels: 1
query_format: "{}"
document_format: "{}"
train_dataset: data/rerank_data/train.json
train_dataset_type: pointwise
max_label: 2
min_label: 0
max_len: 4096
val_dataset: data/rerank_data/dev.json
val_dataset_type: pointwise
loss_type: pointwise_bce
output_dir: models/finetuned/bge_reranker/
epochs: 1
lr: 5e-5
batch_size: 8
gradient_accumulation_steps: 2
mixed_precision: fp16
log_with: tensorboard
```

#### 步骤3: 训练执行脚本

**新项目文件**: `scripts/train_reranker.sh`

**脚本内容**:

```bash
#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 进入RAG-Retrieval目录
cd RAG-Retrieval/rag_retrieval/train/reranker

# 执行训练
CUDA_VISIBLE_DEVICES="0" accelerate launch \
  --config_file ../../../config/xlmroberta_default_config.yaml \
  train_reranker.py \
  --config ../../../../config/finetune/reranker_training.yaml
```

## 四、训练监控和可视化

### 4.1 原项目代码参考路径

**LLaMA-Factory训练监控**: `/remote-home/share/liangZhang/EVRAG/LLaMA-Factory-main/src/llamafactory/train/sft/workflow.py:106-115`

**TensorBoard日志**: LLaMA-Factory和RAG-Retrieval都支持TensorBoard日志记录

### 4.2 实现步骤

#### 步骤1: 创建训练监控模块

**新项目文件**: `src/evrag/finetune/training_monitor.py`

**主要功能**:

- 实时监控训练loss、学习率等指标
- 从TensorBoard日志中提取数据
- 生成训练过程可视化图表
- 监控GPU利用率和内存使用

**实现要点**:

- 使用 `tensorboard` 库读取TensorBoard事件文件
- 使用 `nvidia-smi` 或 `gpustat` 监控GPU
- 使用 `matplotlib` 和 `seaborn` 生成图表

#### 步骤2: 图表生成功能

**新项目文件**: `src/evrag/finetune/visualization.py`

**需要生成的图表**:

1. **训练loss曲线** (`plot_training_loss`)

   - 训练loss和验证loss对比图
   - X轴：训练步数/epoch
   - Y轴：loss值
   - 使用matplotlib绘制

2. **学习率曲线** (`plot_learning_rate`)

   - 学习率变化图
   - X轴：训练步数
   - Y轴：学习率值

3. **评估指标曲线** (`plot_evaluation_metrics`)

   - 准确率、F1等指标变化
   - 多指标对比图

4. **GPU利用率图** (`plot_gpu_utilization`)

   - 训练过程中的GPU使用情况
   - 从nvidia-smi日志中提取数据
   - 堆叠面积图显示多卡GPU利用率

5. **内存使用图** (`plot_memory_usage`)

   - 显存占用变化
   - 时间序列图

6. **训练速度图** (`plot_training_speed`)

   - tokens/秒、samples/秒
   - 柱状图或折线图

7. **收敛分析图** (`plot_convergence_analysis`)

   - loss收敛速度分析
   - 平滑曲线对比

8. **训练时间线** (`plot_training_timeline`)

   - 各阶段耗时分析
   - 甘特图或堆叠柱状图

**参考代码片段**:

```python
# 原项目 workflow.py:106-115
if trainer.is_world_process_zero() and finetuning_args.plot_loss:
    keys = ["loss"]
    if isinstance(dataset_module.get("eval_dataset"), dict):
        keys += sum(
            [[f"eval_{key}_loss", f"eval_{key}_accuracy"] 
             for key in dataset_module["eval_dataset"].keys()], []
        )
    else:
        keys += ["eval_loss", "eval_accuracy"]
    plot_loss(training_args.output_dir, keys=keys)
```

#### 步骤3: GPU监控脚本

**新项目文件**: `scripts/monitor_gpu.sh`

**脚本内容**:

```bash
#!/bin/bash
# 监控GPU使用情况，保存到日志文件
while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader >> logs/gpu_monitor.log
    sleep 5
done
```

## 五、模型评估

### 5.1 原项目代码参考路径

**评估脚本**: `/remote-home/share/liangZhang/EVRAG/final_score.py`（RAG系统评估）

### 5.2 实现步骤

#### 步骤1: 创建模型评估模块

**新项目文件**: `src/evrag/finetune/model_evaluator.py`

**主要功能**:

- 在验证集上评估微调后的模型
- 对比微调前后的性能
- 生成评估报告

**实现要点**:

- LLM评估：使用测试集进行生成质量评估（BLEU、ROUGE、语义相似度）
- Reranker评估：使用测试集进行排序质量评估（NDCG、MRR、准确率）

#### 步骤2: 性能对比分析

**新项目文件**: `src/evrag/finetune/performance_comparison.py`

**对比指标**:

- **LLM性能对比**:
  - 生成质量：BLEU、ROUGE-L、语义相似度
  - 引用准确率：答案中引用标记的准确性
  - 响应时间：平均响应时间

- **Reranker性能对比**:
  - 排序准确率：Top-K准确率
  - NDCG@K：归一化折损累积增益
  - MRR：平均倒数排名

#### 步骤3: 评估报告生成

**新项目文件**: `src/evrag/finetune/evaluation_report.py`

**报告内容**:

- 评估指标汇总表
- 性能对比图表
- 错误案例分析
- 改进建议

## 六、命令行接口

### 6.1 在main.py中添加命令

**命令1**: `finetune-llm`

**参数**:

- `--config`: 训练配置文件路径（默认：`config/finetune/qwen3_lora_sft.yaml`）
- `--resume`: 是否从checkpoint恢复训练
- `--monitor`: 是否启用训练监控
- `--output-dir`: 模型输出目录

**命令2**: `finetune-reranker`

**参数**:

- `--config`: 训练配置文件路径（默认：`config/finetune/reranker_training.yaml`）
- `--resume`: 是否从checkpoint恢复训练
- `--monitor`: 是否启用训练监控
- `--output-dir`: 模型输出目录

**命令3**: `evaluate-model`

**参数**:

- `--model-type`: 模型类型（`llm` 或 `reranker`）
- `--model-path`: 模型路径
- `--test-data`: 测试数据路径
- `--output-report`: 评估报告输出路径

**命令4**: `plot-training-metrics`

**参数**:

- `--log-dir`: TensorBoard日志目录
- `--output-dir`: 图表输出目录
- `--metrics`: 要绘制的指标（默认：all）

## 七、文件结构

```
src/evrag/finetune/
├── __init__.py
├── llm_finetuner.py          # LLM微调模块
├── reranker_finetuner.py     # Reranker微调模块
├── data_converter.py          # 数据格式转换
├── training_monitor.py         # 训练监控
├── visualization.py           # 可视化图表生成
├── model_evaluator.py          # 模型评估
├── performance_comparison.py    # 性能对比
└── evaluation_report.py         # 评估报告生成

config/finetune/
├── qwen3_lora_sft.yaml        # LLM微调配置
├── reranker_training.yaml     # Reranker微调配置
└── datasets/
    └── faq_summary.yaml        # 数据集配置

scripts/
├── train_llm.sh               # LLM训练脚本
├── train_reranker.sh          # Reranker训练脚本
└── monitor_gpu.sh             # GPU监控脚本

models/finetuned/
├── qwen3_lora_sft/            # LLM微调模型输出
│   ├── checkpoint-*/
│   ├── adapter_model/
│   └── training_state.json
└── bge_reranker/              # Reranker微调模型输出
    ├── checkpoint-*/
    └── pytorch_model.bin

reports/
├── training/                  # 训练报告
│   ├── llm_training_report.md
│   ├── reranker_training_report.md
│   └── plots/
│       ├── training_loss.png
│       ├── learning_rate.png
│       ├── gpu_utilization.png
│       └── ...
└── evaluation/                 # 评估报告
    ├── llm_evaluation_report.md
    ├── reranker_evaluation_report.md
    └── performance_comparison.png
```

## 八、测试和验证

### 8.1 训练测试

- **配置验证**：验证训练配置文件正确性
- **数据验证**：验证训练数据格式和完整性
- **训练启动**：验证训练能够正常启动
- **Checkpoint保存**：验证checkpoint能够正常保存和加载

### 8.2 监控测试

- **指标记录**：验证训练指标能够正常记录到TensorBoard
- **GPU监控**：验证GPU监控脚本能够正常运行
- **图表生成**：验证所有图表能够正常生成

### 8.3 评估测试

- **模型加载**：验证微调后的模型能够正常加载
- **评估执行**：验证评估脚本能够正常运行
- **报告生成**：验证评估报告能够正常生成

## 九、依赖和工具

### 9.1 必需依赖

- `llamafactory`：LLM微调框架
- `accelerate`：分布式训练加速
- `tensorboard`：训练日志记录和可视化
- `matplotlib`、`seaborn`：图表生成
- `nvidia-ml-py`：GPU监控
- `transformers`：模型加载和评估

### 9.2 可选依赖

- `wandb`：实验跟踪（可选）
- `gpustat`：GPU状态监控（可选）

## 十、参考文档

- LLaMA-Factory文档：`LLaMA-Factory-main/README.md`
- RAG-Retrieval文档：`RAG-Retrieval/rag_retrieval/train/reranker/README.md`
- 原项目训练配置：`EVRAG/LLaMA-Factory-main/examples/train_lora/qwen3_lora_sft.yaml`
- 原项目Reranker配置：`EVRAG/RAG-Retrieval/rag_retrieval/train/reranker/config/training_bert.yaml`

### To-dos

- [ ] 创建微调模块目录结构和基础文件（__init__.py等）
- [ ] 实现LLM微调模块（llm_finetuner.py），包括配置管理、数据转换、训练执行
- [ ] 实现Reranker微调模块（reranker_finetuner.py），包括配置管理、训练执行
- [ ] 实现数据格式转换模块（data_converter.py），将SFT数据转换为LLaMA-Factory格式
- [ ] 实现训练监控模块（training_monitor.py），实时监控训练指标和GPU使用
- [ ] 实现可视化模块（visualization.py），生成训练loss曲线、学习率曲线、GPU利用率等图表
- [ ] 实现模型评估模块（model_evaluator.py），在验证集上评估微调后的模型
- [ ] 实现性能对比模块（performance_comparison.py），对比微调前后模型性能
- [ ] 实现评估报告生成模块（evaluation_report.py），生成详细的评估报告
- [ ] 创建训练配置文件（qwen3_lora_sft.yaml、reranker_training.yaml、faq_summary.yaml）
- [ ] 创建训练脚本（train_llm.sh、train_reranker.sh、monitor_gpu.sh）
- [ ] 在main.py中添加命令行接口（finetune-llm、finetune-reranker、evaluate-model、plot-training-metrics）
- [ ] 测试完整的训练流程（配置验证、训练启动、checkpoint保存）
- [ ] 测试训练监控和可视化功能（指标记录、图表生成）
- [ ] 测试模型评估功能（模型加载、评估执行、报告生成）