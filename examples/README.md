# 示例数据目录

本目录包含EvRAG项目运行过程中会出现的完整目录结构示例，包括`data/`和`models/`目录。

## 目录结构

```
examples/
├── data/                    # 数据目录示例
│   ├── dataset_info.json   # 数据集信息配置
│   ├── failed/             # 失败的数据文件
│   ├── finetune/           # 微调训练数据
│   ├── logs/               # 日志和统计信息
│   ├── qa_pairs/           # QA对数据
│   ├── rerank_data/        # Reranker训练数据
│   ├── summary_data/       # LLM训练数据（摘要格式）
│   ├── ut/                 # 单元测试数据
│   └── visualizations/     # 可视化结果
└── models/                 # 模型目录示例
    ├── Qwen3-8B/           # 基础LLM模型
    ├── m3e-small/          # 语义切分嵌入模型
    ├── bge-m3/             # 向量检索嵌入模型
    ├── bge-reranker-v2-m3/ # 基线Reranker模型
    └── finetuned/          # 微调后的模型
        ├── qwen3_lora_sft/ # LLM微调模型（LoRA）
        └── bge_reranker/   # Reranker微调模型
```

## 使用说明

这些示例文件展示了项目运行过程中会生成的数据和模型文件结构。实际使用时：

1. **数据文件**：运行相应的命令会自动生成到`data/`目录
2. **模型文件**：需要从HuggingFace下载或通过训练生成到`models/`目录

## 详细说明

- **data目录说明**：请参考 `examples/data/README.md`
- **models目录说明**：请参考 `examples/models/README.md`

## 注意事项

- 示例文件只包含少量数据，用于展示数据格式和目录结构
- 实际运行时会生成更多数据文件
- 某些文件（如`.pkl`、`.db`、`.safetensors`）由于格式特殊或文件较大，未包含在示例中
- 图片文件保存在`saved_images/`目录，未包含在示例中

