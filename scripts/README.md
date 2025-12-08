# Scripts 目录说明

本目录包含项目运行所需的各种脚本，按功能分类组织。

## 📁 目录结构

```
scripts/
├── deployment/          # 部署和服务管理脚本
├── evaluation/          # 评估和测试脚本
├── training/            # 模型训练脚本
├── utils/               # 工具和辅助脚本
└── setup/               # 环境设置脚本
```

---

## 🚀 Deployment（部署脚本）

用于启动和停止各种服务。

### 启动脚本

- **`start_gradio.sh`** - 启动Gradio前端服务（端口8080）
- **`start_vllm_finetuned.sh`** - 启动微调后的vLLM服务（端口8001）
- **`start_vllm_for_evaluation.sh`** - 启动用于评估的vLLM服务
- **`start_mongodb_semantic_services.sh`** - 启动MongoDB和语义服务

### 停止脚本

- **`stop_gradio.sh`** - 停止Gradio前端服务
- **`stop_vllm_finetuned.sh`** - 停止微调后的vLLM服务
- **`stop_vllm_for_evaluation.sh`** - 停止评估用的vLLM服务
- **`stop_mongodb_semantic_services.sh`** - 停止MongoDB和语义服务

### 使用示例

```bash
# 启动完整RAG系统（按顺序执行）
./scripts/deployment/start_vllm_finetuned.sh      # 1. 启动LLM服务
python -m src.evrag.server.rag_server              # 2. 启动RAG服务（端口8002）
./scripts/deployment/start_gradio.sh               # 3. 启动前端界面

# 停止服务
./scripts/deployment/stop_gradio.sh
./scripts/deployment/stop_vllm_finetuned.sh
```

---

## 📊 Evaluation（评估脚本）

用于评估RAG系统性能和模型效果。

### 评估脚本

- **`run_baseline_evaluation.sh`** - 运行基线模型评估
- **`run_finetuned_evaluation.sh`** - 运行微调模型评估
- **`run_baseline_finetuned_comparison.sh`** - 对比基线和微调模型
- **`run_qwen_baseline_evaluation.py`** - Qwen基线模型评估（Python脚本）

### 分析脚本

- **`evaluate_llm_comparison.py`** - LLM模型对比评估
- **`evaluate_reranker_comparison.py`** - Reranker模型对比评估
- **`analyze_evaluation_results.py`** - 分析评估结果

### 文档

- **`README_qwen_baseline.md`** - Qwen基线评估说明文档

### 使用示例

```bash
# 运行RAG系统评估
./scripts/evaluation/run_finetuned_evaluation.sh

# 对比基线和微调模型
./scripts/evaluation/run_baseline_finetuned_comparison.sh

# 分析评估结果
python scripts/evaluation/analyze_evaluation_results.py
```

---

## 🎓 Training（训练脚本）

用于模型微调和训练。

### 训练脚本

- **`train_llm.sh`** - LLM微调训练脚本
- **`train_reranker.sh`** - Reranker微调训练脚本

### 使用示例

```bash
# 训练LLM
./scripts/training/train_llm.sh

# 训练Reranker
./scripts/training/train_reranker.sh
```

**注意**: 训练前需要准备好训练数据和配置文件。详细说明请参考 `dev_docs/` 目录下的相关文档。

---

## 🛠️ Utils（工具脚本）

辅助工具和测试脚本。

### 工具脚本

- **`check_code_quality.sh`** - 代码质量检查（linting、格式化等）
- **`monitor_gpu.sh`** - GPU使用情况监控
- **`test_image_loading.py`** - 测试图片加载功能

### 使用示例

```bash
# 代码质量检查
./scripts/utils/check_code_quality.sh

# 监控GPU
./scripts/utils/monitor_gpu.sh

# 测试图片加载
python scripts/utils/test_image_loading.py
```

---

## ⚙️ Setup（设置脚本）

环境设置和依赖管理。

### 设置脚本

- **`download_models.sh`** - 下载所需模型文件
- **`manage_dependencies.sh`** - 管理项目依赖

### 使用示例

```bash
# 下载模型
./scripts/setup/download_models.sh

# 管理依赖
./scripts/setup/manage_dependencies.sh
```

---

## 📝 注意事项

1. **权限**: 确保脚本有执行权限：`chmod +x scripts/**/*.sh`
2. **环境**: 所有脚本应在项目根目录执行，或使用绝对路径
3. **依赖**: 运行前确保已安装所有依赖（`pip install -r requirements.txt`）
4. **配置**: 某些脚本需要配置文件，请参考 `config/` 目录

---

## 🔗 相关文档

- 主README: `README.md`
- 开发文档: `dev_docs/`
- 配置文件: `config/`

