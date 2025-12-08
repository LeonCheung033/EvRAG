"""
RAGas框架评估模块

使用RAGas框架进行LLM评估，计算ContextRecall和ContextPrecision。
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper

from ..config import get_settings


class RAGasEvaluator:
    """
    RAGas评估器

    使用RAGas框架进行LLM评估，计算ContextRecall和ContextPrecision。
    评估LLM使用SiliconFlow API调用DeepSeek-V3模型。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_workers: int = 8,  # 降低默认并发数，避免429错误
    ):
        """
        初始化RAGas评估器

        Args:
            api_key: API密钥（如果为None，从环境变量或配置文件读取）
            base_url: API基础URL（如果为None，从环境变量或配置文件读取）
            model_name: 模型名称（如果为None，从环境变量或配置文件读取）
            max_workers: 最大并发工作线程数（默认8，避免速率限制）
        """
        settings = get_settings()

        # 优先使用参数，其次使用环境变量，最后使用配置文件
        # 注意：RAGas评估使用SiliconFlow API调用DeepSeek-V3模型
        self.api_key = (
            api_key
            or os.environ.get("SILICONFLOW_API_KEY")
            or settings.siliconflow_api_key
        )
        self.base_url = (
            base_url
            or os.environ.get("SILICONFLOW_BASE_URL")
            or settings.siliconflow_base_url
        )
        # 默认使用DeepSeek-V3模型（通过SiliconFlow）
        self.model_name = model_name or "deepseek-ai/DeepSeek-V3"

        if not self.api_key:
            raise ValueError(
                "RAGas评估需要API密钥，请设置api_key参数或SILICONFLOW_API_KEY环境变量"
            )
        if not self.base_url:
            raise ValueError(
                "RAGas评估需要API基础URL，请设置base_url参数或SILICONFLOW_BASE_URL环境变量"
            )
        if not self.model_name:
            raise ValueError(
                "RAGas评估需要模型名称，请设置model_name参数或RAGAS_MODEL_NAME环境变量"
            )

        # 保存并发配置
        self.max_workers = max_workers

        # 初始化Langchain ChatOpenAI客户端（用于RAGas）
        # 设置temperature=0以确保输出更稳定，减少格式错误
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.0,  # 设置为0，确保输出更稳定
            max_retries=5,  # 增加重试次数
            timeout=300,  # 增加超时时间
        )

        # 包装为RAGas的LLM
        self.evaluator_llm = LangchainLLMWrapper(self.llm)

        print("✓ RAGas评估器已初始化")
        print(f"  模型: {self.model_name}")
        print(f"  API地址: {self.base_url}")

    def evaluate_from_rag_results(
        self,
        rag_results: List[Dict[str, Any]],
        sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        从RAG评估结果中提取数据并进行RAGas评估

        Args:
            rag_results: RAG评估结果列表，每个元素包含：
                - question: 问题
                - ground_truth_answer: 标准答案
                - generated_answer: 生成答案
                - context: 检索到的上下文（字符串）
            sample_size: 抽样测试样本数（None表示全量测试）

        Returns:
            包含RAGas评估指标的字典：
                - context_recall: 上下文召回率
                - context_precision: 上下文精确率
        """
        # 抽样测试
        if sample_size and sample_size < len(rag_results):
            import random

            rag_results = random.sample(rag_results, sample_size)
            print(f"RAGas评估：从{len(rag_results)}条中抽取{sample_size}条")

        # 构建RAGas评估数据集
        dataset = []
        for result in rag_results:
            query = result.get("question", "")
            reference = result.get("ground_truth_answer", "")
            response = result.get("generated_answer", "")
            context = result.get("context", "")

            # context可能是字符串，需要拆分为文档列表
            # 格式：【1】文档1内容\n【2】文档2内容...
            if isinstance(context, str):
                # 按照【数字】标记拆分文档
                import re

                # 匹配【数字】开头的模式
                pattern = r"【\d+】"
                parts = re.split(pattern, context)
                # 过滤掉空字符串，保留文档内容
                context_list = [part.strip() for part in parts if part.strip()]
                # 如果拆分失败（没有【数字】标记），则使用整个字符串
                if not context_list:
                    context_list = [context]
            elif isinstance(context, list):
                context_list = context
            else:
                context_list = [str(context)]

            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": context_list,
                    "response": response,
                    "reference": reference,
                }
            )

        if not dataset:
            return {
                "context_recall": 0.0,
                "context_precision": 0.0,
            }

        # 执行RAGas评估
        print(f"开始RAGas评估，共{len(dataset)}条样本...")
        # 调试：打印第一条样本的格式
        if len(dataset) > 0:
            sample = dataset[0]
            print("调试信息 - 第一条样本格式:")
            print(
                f"  user_input类型: {type(sample.get('user_input'))}, 长度: {len(sample.get('user_input', ''))}"
            )
            print(
                f"  retrieved_contexts类型: {type(sample.get('retrieved_contexts'))}, 数量: {len(sample.get('retrieved_contexts', []))}"
            )
            if sample.get("retrieved_contexts"):
                print(
                    f"  第一个context长度: {len(sample.get('retrieved_contexts')[0])}"
                )
            print(
                f"  response类型: {type(sample.get('response'))}, 长度: {len(sample.get('response', ''))}"
            )
            print(
                f"  reference类型: {type(sample.get('reference'))}, 长度: {len(sample.get('reference', ''))}"
            )

        try:
            evaluation_dataset = EvaluationDataset.from_list(dataset)

            # 配置并发参数（适用于云端API）
            # 降低并发数，添加重试和退避机制，避免429错误
            run_config = RunConfig(
                max_workers=self.max_workers,  # 默认8，避免速率限制
                timeout=300,  # 增加超时时间（秒）
                max_retries=5,  # 增加重试次数
                max_wait=10,  # 重试最大等待时间（秒），用于处理429错误
            )

            # 使用ContextRecall和ContextPrecision两个指标（不启用AnswerRelevancy）
            ragas_result = evaluate(
                dataset=evaluation_dataset,
                metrics=[
                    LLMContextRecall(),
                    LLMContextPrecisionWithReference(),
                ],
                llm=self.evaluator_llm,
                run_config=run_config,
            )

            # 提取评估结果
            # ragas_result是一个RAGas评估结果对象，包含DataFrame
            result_df = ragas_result.to_pandas()

            # 调试：打印DataFrame的列名和前几行数据
            print(f"调试信息 - DataFrame列名: {list(result_df.columns)}")
            print(f"调试信息 - DataFrame形状: {result_df.shape}")

            # 计算平均值（处理NaN值）

            context_recall = 0.0
            context_precision = 0.0

            # RAGas返回的列名可能是 'context_recall' 或 'llm_context_recall'
            recall_col = None
            for col in ["context_recall", "llm_context_recall"]:
                if col in result_df.columns:
                    recall_col = col
                    break

            if recall_col:
                # 只计算有效值（非NaN）的平均值
                valid_recall = result_df[recall_col].dropna()
                if len(valid_recall) > 0:
                    context_recall = float(valid_recall.mean())
                else:
                    context_recall = 0.0
                    print(f"警告: {recall_col}列所有值都是NaN")

            # RAGas返回的列名是 'llm_context_precision_with_reference'
            precision_col = None
            for col in ["llm_context_precision_with_reference", "context_precision"]:
                if col in result_df.columns:
                    precision_col = col
                    break

            if precision_col:
                # 只计算有效值（非NaN）的平均值
                valid_precision = result_df[precision_col].dropna()
                if len(valid_precision) > 0:
                    context_precision = float(valid_precision.mean())
                    print(
                        f"✓ {precision_col}计算成功: {len(valid_precision)}/{len(result_df)}个有效值"
                    )
                else:
                    context_precision = 0.0
                    print(f"⚠ 警告: {precision_col}列所有值都是NaN")
                    # 详细调试信息
                    print(f"  总样本数: {len(result_df)}")
                    print(f"  NaN数量: {result_df[precision_col].isna().sum()}")
                    print(f"  列的前5个值: {result_df[precision_col].head().tolist()}")
                    # 检查是否有错误信息
                    if "error" in result_df.columns:
                        error_count = result_df["error"].notna().sum()
                        if error_count > 0:
                            print(f"  有{error_count}个样本包含错误信息")
                            print(
                                f"  错误示例: {result_df[result_df['error'].notna()]['error'].iloc[0]}"
                            )
            else:
                print("⚠ 警告: 未找到context_precision相关列")
                print(f"  可用列: {list(result_df.columns)}")
                print(
                    f"  尝试查找包含'precision'的列: {[col for col in result_df.columns if 'precision' in col.lower()]}"
                )
                # 尝试使用其他可能的列名
                for alt_col in [
                    "precision",
                    "context_precision_score",
                    "llm_context_precision",
                ]:
                    if alt_col in result_df.columns:
                        print(f"  找到替代列: {alt_col}")
                        valid_precision = result_df[alt_col].dropna()
                        if len(valid_precision) > 0:
                            context_precision = float(valid_precision.mean())
                            print(f"  使用{alt_col}计算: {context_precision:.4f}")
                            break

            print("✓ RAGas评估完成")
            print(f"  ContextRecall: {context_recall:.4f}")
            print(f"  ContextPrecision: {context_precision:.4f}")

            return {
                "context_recall": context_recall,
                "context_precision": context_precision,
            }

        except Exception as e:
            print(f"❌ RAGas评估失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "context_recall": 0.0,
                "context_precision": 0.0,
                "error": str(e),
            }

    def evaluate_from_dataset(
        self,
        dataset: List[Dict[str, Any]],
        sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        直接从数据集进行RAGas评估

        Args:
            dataset: 评估数据集，每个元素包含：
                - user_input: 问题
                - retrieved_contexts: 检索到的上下文列表
                - response: 生成的答案
                - reference: 标准答案
            sample_size: 抽样测试样本数（None表示全量测试）

        Returns:
            包含RAGas评估指标的字典
        """
        # 抽样测试
        if sample_size and sample_size < len(dataset):
            import random

            dataset = random.sample(dataset, sample_size)
            print(f"RAGas评估：从{len(dataset)}条中抽取{sample_size}条")

        if not dataset:
            return {
                "context_recall": 0.0,
                "context_precision": 0.0,
            }

        # 执行RAGas评估
        print(f"开始RAGas评估，共{len(dataset)}条样本...")
        try:
            evaluation_dataset = EvaluationDataset.from_list(dataset)

            # 使用ContextRecall和ContextPrecision两个指标
            ragas_result = evaluate(
                dataset=evaluation_dataset,
                metrics=[
                    LLMContextRecall(),
                    LLMContextPrecisionWithReference(),
                ],
                llm=self.evaluator_llm,
            )

            # 提取评估结果
            result_df = ragas_result.to_pandas()

            # 计算平均值（处理NaN值）
            import numpy as np

            context_recall = 0.0
            context_precision = 0.0

            if "context_recall" in result_df.columns:
                recall_mean = result_df["context_recall"].mean()
                context_recall = (
                    float(recall_mean) if not np.isnan(recall_mean) else 0.0
                )

            if "context_precision" in result_df.columns:
                precision_mean = result_df["context_precision"].mean()
                context_precision = (
                    float(precision_mean) if not np.isnan(precision_mean) else 0.0
                )

            print("✓ RAGas评估完成")
            print(f"  ContextRecall: {context_recall:.4f}")
            print(f"  ContextPrecision: {context_precision:.4f}")

            return {
                "context_recall": context_recall,
                "context_precision": context_precision,
            }

        except Exception as e:
            print(f"❌ RAGas评估失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "context_recall": 0.0,
                "context_precision": 0.0,
                "error": str(e),
            }
