"""
Qwen基线系统评估模块

实现Qwen3-32B + Qwen3-Embedding-8B基线系统的评估。
使用SiliconFlow API调用Qwen模型，仅使用向量检索（无BM25，无重排序）。
基于LangChain的RAG链实现。
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_core.documents import Document
import numpy as np

# LangChain组件
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# PDF加载库
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    try:
        from langchain_community.document_loaders import PyPDFLoader

        PYMUPDF_AVAILABLE = False
    except ImportError:
        pass

from ..config import get_settings
from .metrics import (
    calculate_semantic_keyword_score,
    calculate_generation_quality,
    calculate_response_time,
)
from .ragas_evaluator import RAGasEvaluator


class QwenBaselineFAISSRetriever:
    """
    Qwen基线系统的FAISS检索器

    使用Qwen3-Embedding-8B + FAISS进行向量检索（仅向量检索，无BM25混合）。
    基于LangChain的OpenAIEmbeddings和FAISS实现。
    从原始PDF文件加载并分块，模拟真实chatbot的粗糙处理方式。
    """

    def __init__(
        self,
        pdf_path: Optional[Path] = None,
        faiss_db_path: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ):
        """
        初始化Qwen基线FAISS检索器

        Args:
            pdf_path: PDF文件路径（如果为None，从配置读取）
            faiss_db_path: FAISS数据库保存路径（如果为None，从配置读取）
            chunk_size: 文本分块大小（默认500）
            chunk_overlap: 文本分块重叠大小（默认50）
            api_key: SiliconFlow API密钥（如果为None，从配置读取）
            base_url: SiliconFlow API基础URL（如果为None，从配置读取）
            embedding_model_name: Embedding模型名称（默认Qwen/Qwen3-Embedding-8B）
        """
        settings = get_settings()

        # SiliconFlow配置
        self.api_key = api_key or settings.siliconflow_api_key
        self.base_url = (
            base_url or settings.siliconflow_base_url or "https://api.siliconflow.cn/v1"
        )
        self.embedding_model_name = embedding_model_name or "Qwen/Qwen3-Embedding-8B"

        if not self.api_key:
            raise ValueError(
                "SiliconFlow API key is required. "
                "Please set siliconflow_api_key in config.yaml."
            )

        # 创建OpenAI Embeddings（使用SiliconFlow API）
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            check_embedding_ctx_length=False,  # 避免上下文长度检查
        )

        # PDF文件路径
        if pdf_path is None:
            pdf_path = settings.pdf_path
        if pdf_path is None:
            raise ValueError(
                "PDF文件路径未指定。请提供pdf_path参数或在config.yaml中设置pdf_path。"
            )
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise ValueError(f"PDF文件不存在: {self.pdf_path}")

        # 分块参数
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # FAISS数据库路径
        if faiss_db_path is None:
            faiss_db_path = settings.faiss_qwen_db_path or Path(
                "data/saved_index/faiss_qwen.db"
            )
        self.faiss_db_path = Path(faiss_db_path)

        # 尝试加载现有的FAISS索引
        faiss_index_file = self.faiss_db_path / "index.faiss"
        faiss_pkl_file = self.faiss_db_path / "index.pkl"

        if faiss_index_file.exists() and faiss_pkl_file.exists():
            try:
                print(f"加载现有FAISS索引: {self.faiss_db_path}")
                self.vectorstore = FAISS.load_local(
                    str(self.faiss_db_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("✓ FAISS索引加载成功")
            except Exception as e:
                print(f"⚠ 加载FAISS索引失败: {e}")
                print("  将从PDF重新构建索引...")
                self._build_index_from_pdf()
        else:
            # 如果索引不存在，从PDF构建
            print("FAISS索引不存在，将从PDF文件构建索引...")
            self._build_index_from_pdf()

        print("✓ Qwen基线FAISS检索器已初始化")
        print(f"  PDF路径: {self.pdf_path}")
        print(f"  索引路径: {self.faiss_db_path}")
        print(f"  Embedding模型: {self.embedding_model_name}")
        print(f"  分块大小: {self.chunk_size}, 重叠: {self.chunk_overlap}")

    def _build_index_from_pdf(self):
        """从PDF文件加载文档、分块并构建FAISS索引"""
        print(f"从PDF文件加载文档: {self.pdf_path}")

        # 1. 加载PDF文档（使用PyMuPDF）
        if PYMUPDF_AVAILABLE:
            # 使用PyMuPDF加载PDF
            doc = fitz.open(str(self.pdf_path))
            docs = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # 只添加非空页面
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(self.pdf_path),
                                "page": page_num + 1,
                            },
                        )
                    )
            doc.close()
            print(f"  ✓ 使用PyMuPDF加载了 {len(docs)} 页PDF文档")
        else:
            # 回退到PyPDFLoader
            try:
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(str(self.pdf_path))
                docs = loader.load()
                print(f"  ✓ 使用PyPDFLoader加载了 {len(docs)} 页PDF文档")
            except Exception as e:
                raise ValueError(
                    f"无法加载PDF文件。请安装PyMuPDF: pip install pymupdf\n"
                    f"或安装cryptography: pip install cryptography>=3.1\n"
                    f"错误详情: {e}"
                )

        if not docs:
            raise ValueError("PDF文档为空，无法构建FAISS索引")

        # 2. 分块（Chunking）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(docs)
        print(f"  ✓ 分块完成，共 {len(chunks)} 个文档块")

        if not chunks:
            raise ValueError("PDF文档分块后为空，无法构建FAISS索引")

        # 3. 构建FAISS索引
        print("构建FAISS索引（这可能需要一些时间）...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # 保存索引
        self.faiss_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.faiss_db_path))
        print(f"✓ FAISS索引已保存到: {self.faiss_db_path}")

    def retrieve_topk(
        self,
        query: str,
        topk: int = 10,
    ) -> List[Document]:
        """
        检索Top-K文档

        Args:
            query: 查询问题
            topk: 返回的文档数量

        Returns:
            文档列表
        """
        # 使用FAISS进行相似度搜索
        results = self.vectorstore.similarity_search_with_score(query, k=topk)

        # 转换为Document对象（保留score信息）
        docs = []
        for doc, score in results:
            docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "score": float(score),
                    },
                )
            )

        return docs


class QwenBaselineEvaluator:
    """
    Qwen基线系统评估器

    使用Qwen3-32B + Qwen3-Embedding-8B进行端到端RAG评估。
    仅使用向量检索（无BM25，无重排序），直接生成答案（无后处理）。
    基于LangChain的RAG链实现。
    """

    def __init__(
        self,
        topk: int = 10,
        llm_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        pdf_path: Optional[Path] = None,
        faiss_db_path: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        """
        初始化Qwen基线评估器

        Args:
            topk: 检索数量（topk）
            llm_model_name: LLM模型名称（默认Qwen/Qwen3-32B）
            embedding_model_name: Embedding模型名称（默认Qwen/Qwen3-Embedding-8B）
            max_workers: 最大并发工作线程数
            pdf_path: PDF文件路径（如果为None，从配置读取）
            faiss_db_path: FAISS数据库路径
            chunk_size: 文本分块大小（默认500）
            chunk_overlap: 文本分块重叠大小（默认50）
            api_key: SiliconFlow API密钥（如果为None，从配置读取）
            base_url: SiliconFlow API基础URL（如果为None，从配置读取）
            temperature: LLM温度参数
            max_tokens: LLM最大生成token数
        """
        settings = get_settings()

        self.topk = topk
        self.max_workers = max_workers or 2

        # SiliconFlow配置
        self.api_key = api_key or settings.siliconflow_api_key
        self.base_url = (
            base_url or settings.siliconflow_base_url or "https://api.siliconflow.cn/v1"
        )
        self.llm_model_name = llm_model_name or "Qwen/Qwen3-32B"
        self.embedding_model_name = embedding_model_name or "Qwen/Qwen3-Embedding-8B"

        if not self.api_key:
            raise ValueError(
                "SiliconFlow API key is required. "
                "Please set siliconflow_api_key in config.yaml."
            )

        # 初始化FAISS检索器（从PDF加载并分块）
        self.retriever = QwenBaselineFAISSRetriever(
            pdf_path=pdf_path or settings.pdf_path,
            faiss_db_path=faiss_db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model_name=self.embedding_model_name,
        )

        # 初始化LLM（使用ChatOpenAI，通过SiliconFlow API）
        self.llm = ChatOpenAI(
            model=self.llm_model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 构建Prompt模板（简单的基线prompt，无后处理要求）
        self.prompt_template = ChatPromptTemplate.from_template(
            """你是一个专业助手，请严格根据以下提供的参考资料回答问题。

如果资料中没有相关信息，请回答"根据现有资料无法回答该问题"。

参考资料：

{context}

问题：{question}

请直接输出答案，不要输出任何思考过程或解释性文字。如果无法从信息中得到答案，请说"无答案"。
"""
        )

        # 格式化文档的函数
        def format_docs(docs):
            return "\n\n".join(
                [f"【{idx + 1}】{doc.page_content}" for idx, doc in enumerate(docs)]
            )

        # 构建RAG链
        # 注意：这里不使用LangChain的retriever，而是手动调用retrieve_topk
        # 以便更好地控制检索过程和记录时间
        self.format_docs = format_docs

        # 初始化语义相似度模型（用于评估）
        from text2vec import SentenceModel

        self.semantic_model = SentenceModel()

        self.evaluation_results: List[Dict[str, Any]] = []

        print("✓ Qwen基线评估器初始化成功")
        print(f"  LLM模型: {self.llm_model_name}")
        print(f"  Embedding模型: {self.embedding_model_name}")
        print(f"  API地址: {self.base_url}")

    def _generate_answer(self, question: str, context: str) -> str:
        """
        使用LangChain RAG链生成答案

        Args:
            question: 问题
            context: 检索到的上下文

        Returns:
            生成的答案
        """
        # 构建消息
        messages = self.prompt_template.format_messages(
            context=context, question=question
        )

        # 调用LLM
        response = self.llm.invoke(messages)

        # 提取答案内容
        if hasattr(response, "content"):
            return response.content.strip()
        else:
            return str(response).strip()

    def evaluate_single(
        self,
        question: str,
        ground_truth_answer: str,
        keywords: List[str],
        unique_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估单个样本

        Args:
            question: 问题
            ground_truth_answer: 标准答案
            keywords: 关键词列表
            unique_id: 唯一标识符

        Returns:
            包含评估结果的字典
        """
        timings = {}
        start_time = time.time()

        # 1. 向量检索
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve_topk(
            question,
            topk=self.topk,
        )
        timings["retrieval_time"] = time.time() - retrieval_start

        # 2. 构建上下文（无重排序）
        context = self.format_docs(retrieved_docs)

        # 3. LLM生成（无后处理）
        generation_start = time.time()
        generated_answer = self._generate_answer(question, context)
        timings["generation_time"] = time.time() - generation_start

        # 计算总时间
        timings["total_time"] = time.time() - start_time

        # 4. 计算评估指标
        # 语义相似度+关键词加权评分
        semantic_keyword_score = calculate_semantic_keyword_score(
            pred=generated_answer,
            gold=ground_truth_answer,
            keywords=keywords,
            semantic_model=self.semantic_model,
        )

        # 生成质量指标
        generation_quality = calculate_generation_quality(
            pred=generated_answer,
            gold=ground_truth_answer,
            use_rouge=True,
        )

        # 响应时间指标
        response_time_metrics = calculate_response_time(timings)

        # 构建评估结果
        result = {
            "unique_id": unique_id,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "generated_answer": generated_answer,
            "keywords": keywords,
            "context": context,
            "retrieved_docs_count": len(retrieved_docs),
            "metrics": {
                "semantic_keyword_score": semantic_keyword_score,
                "generation_quality": generation_quality,
                "response_time": response_time_metrics,
            },
            "timings": timings,
        }

        return result

    def evaluate_batch(
        self,
        test_data: List[Dict[str, Any]],
        sample_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量评估

        Args:
            test_data: 测试数据列表
            sample_size: 抽样测试样本数（None表示全量测试）
            show_progress: 是否显示进度条

        Returns:
            评估结果列表
        """
        # 抽样测试
        if sample_size and sample_size < len(test_data):
            import random

            random.seed(42)
            test_data = random.sample(test_data, sample_size)
            print(f"抽样测试：从{len(test_data)}条中抽取{sample_size}条")

        results = []

        # 使用线程池进行并发评估
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(
                    self.evaluate_single,
                    item["question"],
                    item["answer"],
                    item.get("keywords", []),
                    item.get("unique_id"),
                ): item
                for item in test_data
            }

            # 收集结果
            if show_progress:
                futures = tqdm(
                    as_completed(future_to_item), total=len(test_data), desc="评估进度"
                )
            else:
                futures = as_completed(future_to_item)

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"评估失败: {item.get('unique_id', 'unknown')}, 错误: {e}")
                    results.append(
                        {
                            "unique_id": item.get("unique_id"),
                            "question": item.get("question", ""),
                            "error": str(e),
                        }
                    )

        self.evaluation_results = results
        return results

    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        计算汇总指标

        Returns:
            包含各项指标平均值的字典
        """
        if not self.evaluation_results:
            return {}

        # 过滤掉有错误的结果
        valid_results = [r for r in self.evaluation_results if "error" not in r]

        if not valid_results:
            return {"error": "没有有效的评估结果"}

        # 提取各项指标
        semantic_keyword_scores = []
        generation_qualities = []
        response_times = []
        no_answer_stats = {"total": 0, "correct": 0, "incorrect": 0}

        for result in valid_results:
            metrics = result.get("metrics", {})

            # 语义相似度+关键词加权得分
            semantic_keyword = metrics.get("semantic_keyword_score", {})
            if semantic_keyword:
                semantic_keyword_scores.append(semantic_keyword.get("final_score", 0.0))

            # 生成质量
            generation_quality = metrics.get("generation_quality", {})
            if generation_quality:
                generation_qualities.append(generation_quality)

            # 响应时间
            response_time = metrics.get("response_time", {})
            if response_time:
                response_times.append(response_time)

            # 无答案样本统计
            gold = result.get("ground_truth_answer", "")
            pred = result.get("generated_answer", "")
            if gold == "无答案":
                no_answer_stats["total"] += 1
                if pred == "无答案":
                    no_answer_stats["correct"] += 1
                else:
                    no_answer_stats["incorrect"] += 1

        # 计算平均值
        summary = {
            "total_samples": len(valid_results),
            "semantic_keyword_score": {
                "mean": float(np.mean(semantic_keyword_scores))
                if semantic_keyword_scores
                else 0.0,
                "std": float(np.std(semantic_keyword_scores))
                if semantic_keyword_scores
                else 0.0,
            },
            "generation_quality": {
                "bleu": float(
                    np.mean([g.get("bleu", 0.0) for g in generation_qualities])
                )
                if generation_qualities
                else 0.0,
                "rouge_1": float(
                    np.mean([g.get("rouge_1", 0.0) for g in generation_qualities])
                )
                if generation_qualities
                else 0.0,
                "rouge_2": float(
                    np.mean([g.get("rouge_2", 0.0) for g in generation_qualities])
                )
                if generation_qualities
                else 0.0,
                "rouge_l": float(
                    np.mean([g.get("rouge_l", 0.0) for g in generation_qualities])
                )
                if generation_qualities
                else 0.0,
            },
            "response_time": {
                "mean_retrieval_time": float(
                    np.mean([r.get("retrieval_time", 0.0) for r in response_times])
                )
                if response_times
                else 0.0,
                "mean_generation_time": float(
                    np.mean([r.get("generation_time", 0.0) for r in response_times])
                )
                if response_times
                else 0.0,
                "mean_total_time": float(
                    np.mean([r.get("total_time", 0.0) for r in response_times])
                )
                if response_times
                else 0.0,
            },
            "no_answer_stats": {
                "total": no_answer_stats["total"],
                "correct": no_answer_stats["correct"],
                "incorrect": no_answer_stats["incorrect"],
                "hit_rate": no_answer_stats["correct"] / no_answer_stats["total"]
                if no_answer_stats["total"] > 0
                else 0.0,
                "false_positive_rate": no_answer_stats["incorrect"]
                / no_answer_stats["total"]
                if no_answer_stats["total"] > 0
                else 0.0,
            },
        }

        return summary

    def calculate_comprehensive_accuracy(
        self,
        use_ragas: bool = True,
        semantic_weight: float = 0.7,
        ragas_weight: float = 0.3,
        ragas_sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        计算综合准确率

        Args:
            use_ragas: 是否使用RAGas评估
            semantic_weight: 语义相似度+关键词加权得分的权重（默认0.7）
            ragas_weight: RAGas得分的权重（默认0.3）
            ragas_sample_size: RAGas评估抽样样本数（None表示全量）

        Returns:
            包含综合准确率的字典
        """
        summary = self.calculate_summary_metrics()

        if "error" in summary:
            return summary

        # 获取语义相似度+关键词加权得分
        avg_semantic_keyword_score = summary.get("semantic_keyword_score", {}).get(
            "mean", 0.0
        )

        # 计算RAGas得分
        avg_ragas_score = 0.0
        ragas_results = None

        if use_ragas:
            try:
                # 使用较低的并发数（避免429速率限制）
                ragas_evaluator = RAGasEvaluator(max_workers=4)
                ragas_results = ragas_evaluator.evaluate_from_rag_results(
                    self.evaluation_results,
                    sample_size=ragas_sample_size,
                )

                context_recall = ragas_results.get("context_recall", 0.0)
                context_precision = ragas_results.get("context_precision", 0.0)

                # 计算RAGas平均得分（与rag_evaluator.py保持一致）
                # 使用权重：ContextRecall 0.7, ContextPrecision 0.3
                avg_ragas_score = 0.7 * context_recall + 0.3 * context_precision

                summary["ragas_scores"] = {
                    "context_recall": context_recall,
                    "context_precision": context_precision,
                    "average": avg_ragas_score,
                }
            except Exception as e:
                print(f"⚠ RAGas评估失败: {e}")
                print("  将仅使用语义相似度+关键词加权得分计算综合准确率")
                use_ragas = False

        # 计算综合准确率
        if use_ragas and avg_ragas_score > 0:
            comprehensive_accuracy = (
                semantic_weight * avg_semantic_keyword_score
                + ragas_weight * avg_ragas_score
            )
        else:
            # 如果RAGas评估不可用，仅使用语义相似度+关键词加权得分
            comprehensive_accuracy = avg_semantic_keyword_score

        summary["comprehensive_accuracy"] = {
            "score": comprehensive_accuracy,
            "semantic_keyword_score": avg_semantic_keyword_score,
            "ragas_score": avg_ragas_score if use_ragas else None,
            "weights": {
                "semantic_weight": semantic_weight,
                "ragas_weight": ragas_weight,
            },
            "use_ragas": use_ragas,
        }

        return summary

    def save_results(self, output_path: Path, summary: Optional[Dict[str, Any]] = None):
        """
        保存评估结果

        Args:
            output_path: 输出文件路径
            summary: 可选的汇总指标（如果提供，将使用此汇总而不是重新计算）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存详细结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)

        # 保存汇总指标（如果提供了summary，使用它；否则重新计算）
        if summary is None:
            summary = self.calculate_summary_metrics()
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"评估结果已保存到: {output_path}")
        print(f"汇总指标已保存到: {summary_path}")
