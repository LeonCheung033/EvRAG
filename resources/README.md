# Resources 资源目录

本目录包含项目对外公开的资源文件，包括图片、示例数据等。

## 📁 目录结构

```
resources/
├── images/          # 项目图片资源
│   ├── sys_architecture.png              # 系统架构图
│   ├── data_pipeline.png                 # 数据处理Pipeline图
│   ├── qa_statistics.png                 # QA数据统计图
│   ├── llm_training_loss.png             # LLM训练Loss曲线
│   ├── reranker_training_loss.png        # Reranker训练Loss曲线
│   ├── performance_comparison_table.png  # 性能对比表格
│   ├── ablation_study_table.png          # 消融实验表格
│   ├── component_contribution.png        # 组件贡献度分析图
│   └── metrics_comparison_bar.png        # 指标对比柱状图
└── README.md         # 本文件
```

## 🖼️ 图片资源说明

### 系统架构相关

- **sys_architecture.png**: EvRAG系统整体架构图，展示用户界面层、RAG服务层、检索层、生成层和重排序层的完整架构

### 数据处理相关

- **data_pipeline.png**: 完整的数据处理Pipeline，从PDF解析到训练数据的完整流程
- **qa_statistics.png**: QA数据生成统计图，展示训练数据的分布和质量

### 训练过程相关

- **llm_training_loss.png**: LLM模型训练Loss曲线，展示训练和评估Loss的变化
- **reranker_training_loss.png**: Reranker模型训练Loss曲线，展示训练和验证Loss的变化

### 实验结果相关

- **performance_comparison_table.png**: 性能对比表格，对比我们的RAG系统与基线系统的各项指标
- **ablation_study_table.png**: 消融实验表格，展示各组件对系统性能的贡献
- **component_contribution.png**: 组件贡献度分析图，展示Reranker微调、LLM微调和协同效应的独立贡献
- **metrics_comparison_bar.png**: 指标对比柱状图，展示微调前后各项指标的对比

## 📊 使用说明

这些图片资源主要用于：
1. **README.md**: 在项目README中展示系统架构、工作流程和实验结果
2. **学术报告**: 在论文或报告中展示系统设计和实验结果
3. **项目演示**: 在演示文稿中展示项目亮点

## 📝 图片生成

图片来源于项目训练和评估过程，主要生成位置：
- 训练Loss曲线：TensorBoard日志提取
- 性能对比表格：评估脚本生成
- 数据统计图：数据分析脚本生成

如需重新生成图片，请参考：
- `scripts/evaluation/` - 评估脚本
- `src/evrag/finetune/visualization.py` - 可视化工具
- `src/evrag/utils/test_and_visualize.py` - 测试和可视化工具

