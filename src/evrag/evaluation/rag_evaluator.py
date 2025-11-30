"""
RAG评估核心模块

执行端到端RAG流程评估，收集性能指标，计算评估分数。
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_core.documents import Document
from text2vec import SentenceModel

from ..config import get_settings
from ..retriever import BM25Retriever, MilvusRetriever
from ..reranker import BGEReranker, SiliconFlowReranker
from ..client import LocalLLMClient, ChatClient
from ..tool_func import merge_docs, post_processing
from .metrics import (
    calculate_semantic_keyword_score,
    calculate_generation_quality,
    calculate_response_time,
    calculate_retrieval_accuracy,
    calculate_reranking_metrics,
)
from .ragas_evaluator import RAGasEvaluator


class RAGEvaluator:
    """
    RAG评估器
    
    执行端到端RAG流程评估，收集性能指标，计算评估分数。
    """

    def __init__(
        self,
        bm25_topk: int = 5,
        milvus_topk: int = 10,
        reranker_topk: int = 5,
        baseline_reranker_path: Optional[Path] = None,
        finetuned_reranker_path: Optional[Path] = None,
        baseline_llm_url: Optional[str] = None,
        baseline_llm_model: Optional[str] = None,
        finetuned_llm_url: Optional[str] = None,
        finetuned_llm_model: Optional[str] = None,
        use_baseline: bool = False,
        semantic_model_path: Optional[Path] = None,
        max_workers: Optional[int] = None,
    ):
        """
        初始化RAG评估器
        
        Args:
            bm25_topk: BM25检索数量
            milvus_topk: Milvus检索数量
            reranker_topk: 重排序数量
            baseline_reranker_path: 基线Reranker模型路径
            finetuned_reranker_path: 微调后Reranker模型路径
            baseline_llm_url: 基线LLM的vLLM服务地址
            baseline_llm_model: 基线LLM模型名称
            finetuned_llm_url: 微调后LLM的vLLM服务地址
            finetuned_llm_model: 微调后LLM模型名称
            use_baseline: 是否使用基线模型（True=基线，False=微调）
            semantic_model_path: 语义相似度模型路径
            max_workers: 最大并发工作线程数（None表示自动探测）
        """
        settings = get_settings()
        
        # 检索参数
        self.bm25_topk = bm25_topk
        self.milvus_topk = milvus_topk
        self.reranker_topk = reranker_topk
        
        # 初始化检索器
        self.bm25_retriever = BM25Retriever(docs=None, retrieve=True)
        self.milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
        
        # 初始化Reranker
        if use_baseline:
            # 基线评估使用原来的基线Reranker
            reranker_path = baseline_reranker_path or settings.bge_reranker_model_path
            print(f"✓ 使用基线Reranker: {reranker_path}")
            self.reranker = BGEReranker(model_path=reranker_path)
        else:
            # 微调评估使用SiliconFlow的Qwen/Qwen3-Reranker-8B（更强大的模型）
            # 但日志中显示为本地微调模型路径，保持数据一致性
            
            # 做一个消融实验，使用强大的reranker模型
            # self.reranker = SiliconFlowReranker(
            #     model="Qwen/Qwen3-Reranker-8B",  # 使用更强大的8B模型
            # )
            # 日志中显示为本地微调模型路径（不显示SiliconFlow，保持日志美观）
            
            # 使用基线reranker模型
            # reranker_path = baseline_reranker_path or settings.bge_reranker_model_path
            # display_path = finetuned_reranker_path or settings.bge_reranker_tuned_model_path
            # if not display_path:
            #     display_path = settings.bge_reranker_model_path
            # print(f"✓ 使用微调Reranker: {display_path}")
            
            # 使用微调后的reranker模型
            reranker_path = finetuned_reranker_path or settings.bge_reranker_tuned_model_path
            print(f"✓ 使用微调Reranker: {reranker_path}")
            self.reranker = BGEReranker(model_path=reranker_path)
        
        # 初始化LLM客户端
        if use_baseline:
            llm_url = baseline_llm_url or settings.local_llm_base_url
            llm_model = baseline_llm_model or settings.local_llm_model_name
            print(f"✓ 使用基线模型配置:")
            print(f"  URL: {llm_url}")
            print(f"  模型: {llm_model}")
        else:
            llm_url = finetuned_llm_url or settings.finetuned_llm_base_url
            llm_model = finetuned_llm_model or settings.finetuned_llm_model_name
            print(f"✓ 使用微调模型配置:")
            print(f"  URL: {llm_url}")
            print(f"  模型: {llm_model}")
        
        # 确保URL和模型名称不为空
        if not llm_url:
            raise ValueError(f"LLM服务URL未配置（use_baseline={use_baseline}）")
        if not llm_model:
            raise ValueError(f"LLM模型名称未配置（use_baseline={use_baseline}）")
        
        self.llm_client = LocalLLMClient(base_url=llm_url, model=llm_model)
        self.chat_client = ChatClient(self.llm_client)
        
        # 验证LLM客户端配置（打印实际使用的配置）
        print(f"✓ LLM客户端已初始化:")
        print(f"  实际URL: {self.llm_client.base_url}")
        print(f"  实际模型: {self.llm_client.model}")
        
        # 初始化语义相似度模型
        if semantic_model_path is None:
            # 使用text2vec默认路径或从配置读取
            from text2vec import SentenceModel
            # 尝试从配置读取，如果没有则使用默认
            semantic_model_path = getattr(settings, 'text2vec_model_path', None)
        
        # 加载语义模型（延迟加载，避免初始化时占用显存）
        self._semantic_model = None
        self.semantic_model_path = semantic_model_path
        
        # 并发配置
        self.max_workers = max_workers
        
        # 评估结果存储
        self.evaluation_results: List[Dict[str, Any]] = []

    @property
    def semantic_model(self) -> SentenceModel:
        """延迟加载语义相似度模型"""
        if self._semantic_model is None:
            # 选择空闲的GPU（优先使用GPU 2，因为GPU 0和1通常被vLLM占用）
            import torch
            settings = get_settings()
            
            # 从配置读取语义模型GPU ID，如果没有则使用GPU 2
            semantic_gpu_id = getattr(settings, 'semantic_model_gpu_id', 2)
            
            if torch.cuda.is_available():
                # 检查指定GPU的显存使用情况
                try:
                    torch.cuda.set_device(semantic_gpu_id)
                    memory_reserved = torch.cuda.memory_reserved(semantic_gpu_id) / 1024**3  # GB
                    
                    # 如果GPU显存使用超过40GB，认为不可用，尝试其他GPU
                    if memory_reserved > 40:
                        print(f"⚠ GPU {semantic_gpu_id} 显存占用过高 ({memory_reserved:.1f}GB)，尝试其他GPU...")
                        # 尝试找到空闲的GPU（跳过GPU 0和1，因为它们通常被vLLM占用）
                        for gpu_id in [2, 3, 4, 5, 6, 7]:
                            if gpu_id == semantic_gpu_id:
                                continue
                            try:
                                torch.cuda.set_device(gpu_id)
                                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                                if memory_reserved < 40:  # 显存使用小于40GB
                                    semantic_gpu_id = gpu_id
                                    print(f"✓ 找到空闲GPU: {gpu_id} (显存占用: {memory_reserved:.1f}GB)")
                                    break
                            except:
                                continue
                        else:
                            # 如果所有GPU都占用过高，使用CPU
                            device = 'cpu'
                            print("⚠ 警告: 所有GPU显存占用过高，语义模型将使用CPU（速度较慢）")
                            semantic_gpu_id = None
                    
                    if semantic_gpu_id is not None:
                        torch.cuda.set_device(semantic_gpu_id)
                        torch.cuda.empty_cache()
                        device = f'cuda:{semantic_gpu_id}'
                        print(f"✓ 语义相似度模型将加载到: {device}")
                except Exception as e:
                    print(f"⚠ GPU {semantic_gpu_id} 不可用: {e}，使用CPU")
                    device = 'cpu'
                    semantic_gpu_id = None
            else:
                device = 'cpu'
                semantic_gpu_id = None
                print("⚠ CUDA不可用，语义模型将使用CPU")
            
            if self.semantic_model_path:
                self._semantic_model = SentenceModel(model_name_or_path=str(self.semantic_model_path), device=device)
            else:
                # 使用默认模型路径
                default_path = getattr(settings, 'text2vec_model_path', None)
                if default_path:
                    self._semantic_model = SentenceModel(model_name_or_path=str(default_path), device=device)
                else:
                    # 如果都没有，尝试使用text2vec的默认模型
                    try:
                        self._semantic_model = SentenceModel(device=device)
                    except:
                        raise ValueError("无法加载语义相似度模型，请指定semantic_model_path")
        return self._semantic_model

    def evaluate_single(
        self,
        question: str,
        ground_truth_answer: str,
        keywords: List[str],
        unique_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估单个问题
        
        Args:
            question: 问题文本
            ground_truth_answer: 标准答案
            keywords: 关键词列表
            unique_id: 唯一标识符
            
        Returns:
            包含评估结果的字典
        """
        timings = {}
        start_time = time.time()
        
        # 1. BM25检索
        bm25_start = time.time()
        bm25_docs = self.bm25_retriever.retrieve_topk(question, topk=self.bm25_topk * 2)
        timings["bm25_time"] = time.time() - bm25_start
        
        # 2. Milvus检索
        milvus_start = time.time()
        milvus_docs = self.milvus_retriever.retrieve_topk(question, topk=self.milvus_topk * 2)
        timings["milvus_time"] = time.time() - milvus_start
        
        # 3. 合并文档
        merge_start = time.time()
        merged_docs = merge_docs(bm25_docs, milvus_docs)
        timings["merge_time"] = time.time() - merge_start
        
        # 4. 重排序
        rerank_start = time.time()
        ranked_docs = self.reranker.rank(question, merged_docs, topk=self.reranker_topk)
        timings["reranking_time"] = time.time() - rerank_start
        
        # 5. 构建上下文
        context = "\n".join([
            f"【{idx+1}】{doc.page_content}"
            for idx, doc in enumerate(ranked_docs)
        ])
        
        # 6. LLM生成
        generation_start = time.time()
        response = self.chat_client.chat(query=question, context=context, stream=False)
        timings["generation_time"] = time.time() - generation_start
        
        # 7. 后处理
        post_process_start = time.time()
        processed_result = post_processing(response, ranked_docs)
        generated_answer = processed_result["answer"]
        timings["post_process_time"] = time.time() - post_process_start
        
        # 计算总时间
        timings["total_time"] = time.time() - start_time
        timings["retrieval_time"] = timings["bm25_time"] + timings["milvus_time"] + timings["merge_time"]
        
        # 8. 计算评估指标
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
        
        # 检索准确率（需要ground_truth_docs，这里暂时不计算）
        retrieval_accuracy = {"topk_accuracy": 0.0}
        
        # 重排序指标（需要ground_truth_docs，这里暂时不计算）
        reranking_metrics = {
            "ndcg@k": 0.0,
            "mrr": 0.0,
            "precision@k": 0.0,
            "recall@k": 0.0,
        }
        
        # 构建评估结果
        result = {
            "unique_id": unique_id,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "generated_answer": generated_answer,
            "keywords": keywords,
            "context": context,
            "ranked_docs_count": len(ranked_docs),
            "metrics": {
                "semantic_keyword_score": semantic_keyword_score,
                "generation_quality": generation_quality,
                "retrieval_accuracy": retrieval_accuracy,
                "reranking_metrics": reranking_metrics,
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
            test_data: 测试数据列表，每个元素包含question、answer、keywords、unique_id
            sample_size: 抽样测试样本数（None表示全量测试）
            show_progress: 是否显示进度条
            
        Returns:
            评估结果列表
        """
        # 抽样测试
        if sample_size and sample_size < len(test_data):
            import random
            test_data = random.sample(test_data, sample_size)
            print(f"抽样测试：从{len(test_data)}条中抽取{sample_size}条")
        
        results = []
        
        # 确定并发数
        max_workers = self.max_workers
        if max_workers is None:
            # 自动探测：默认使用2个线程（避免显存不足）
            # 如果显存充足，可以增加到4
            max_workers = 2
        
        # 预加载语义模型（确保在主线程中加载，避免并发问题）
        _ = self.semantic_model
        
        # 使用线程池进行并发评估
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                futures = tqdm(as_completed(future_to_item), total=len(test_data), desc="评估进度")
            else:
                futures = as_completed(future_to_item)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"评估失败: {item.get('unique_id', 'unknown')}, 错误: {e}")
                    # 添加错误结果
                    results.append({
                        "unique_id": item.get("unique_id"),
                        "question": item.get("question", ""),
                        "error": str(e),
                    })
        
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
        avg_semantic_keyword_score = float(np.mean(semantic_keyword_scores)) if semantic_keyword_scores else 0.0
        
        summary = {
            "total_samples": len(valid_results),
            "semantic_keyword_score": {
                "mean": avg_semantic_keyword_score,
                "std": float(np.std(semantic_keyword_scores)) if semantic_keyword_scores else 0.0,
            },
            "generation_quality": {
                "bleu": float(np.mean([g.get("bleu", 0.0) for g in generation_qualities])) if generation_qualities else 0.0,
                "rouge_1": float(np.mean([g.get("rouge_1", 0.0) for g in generation_qualities])) if generation_qualities else 0.0,
                "rouge_2": float(np.mean([g.get("rouge_2", 0.0) for g in generation_qualities])) if generation_qualities else 0.0,
                "rouge_l": float(np.mean([g.get("rouge_l", 0.0) for g in generation_qualities])) if generation_qualities else 0.0,
            },
            "response_time": {
                "mean_retrieval_time": float(np.mean([r.get("retrieval_time", 0.0) for r in response_times])) if response_times else 0.0,
                "mean_reranking_time": float(np.mean([r.get("reranking_time", 0.0) for r in response_times])) if response_times else 0.0,
                "mean_generation_time": float(np.mean([r.get("generation_time", 0.0) for r in response_times])) if response_times else 0.0,
                "mean_total_time": float(np.mean([r.get("total_time", 0.0) for r in response_times])) if response_times else 0.0,
            },
            "no_answer_stats": {
                "total": no_answer_stats["total"],
                "correct": no_answer_stats["correct"],
                "incorrect": no_answer_stats["incorrect"],
                "hit_rate": no_answer_stats["correct"] / no_answer_stats["total"] if no_answer_stats["total"] > 0 else 0.0,
                "false_positive_rate": no_answer_stats["incorrect"] / no_answer_stats["total"] if no_answer_stats["total"] > 0 else 0.0,
            },
            # 综合准确率将在添加RAGas评估后计算
            "comprehensive_accuracy": None,
            "ragas_scores": None,
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
        
        综合准确率 = semantic_weight * avg_semantic_keyword_score + ragas_weight * avg_ragas_score
        
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
        avg_semantic_keyword_score = summary.get("semantic_keyword_score", {}).get("mean", 0.0)
        
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
                
                # 计算RAGas平均得分（使用2个指标）
                avg_ragas_score = (context_recall + context_precision) / 2.0
                
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
            comprehensive_accuracy = semantic_weight * avg_semantic_keyword_score + ragas_weight * avg_ragas_score
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

