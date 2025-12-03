"""
评估指标计算模块

实现各种RAG评估指标的计算函数。
"""

import re
import numpy as np
from typing import List, Dict, Optional
from langchain_core.documents import Document
from text2vec import SentenceModel, semantic_search
import jieba
from collections import Counter


def calculate_semantic_keyword_score(
    pred: str,
    gold: str,
    keywords: List[str],
    semantic_model: SentenceModel,
) -> Dict[str, float]:
    """
    计算语义相似度+实体关键词加权评分

    Args:
        pred: 生成答案
        gold: 标准答案
        keywords: 实体关键词列表
        semantic_model: 语义相似度模型

    Returns:
        包含各项得分的字典：
        - semantic_score: 语义相似度得分
        - keyword_score: 关键词匹配得分（0或1）
        - final_score: 加权综合得分
    """
    # 处理无答案情况的特殊评分逻辑
    if gold == "无答案" and pred != gold:
        return {
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "final_score": 0.0,
        }
    elif gold == "无答案" and pred == gold:
        return {
            "semantic_score": 1.0,
            "keyword_score": 1.0,
            "final_score": 1.0,
        }

    # 计算语义相似度得分
    semantic_score = semantic_search(
        semantic_model.encode([gold]), semantic_model.encode([pred]), top_k=1
    )[0][0]["score"]

    # 计算关键词匹配得分
    join_keywords = [word for word in keywords if word in pred]
    keyword_score = (
        _calc_jaccard(join_keywords, keywords, threshold=0.3) if keywords else None
    )

    # 根据是否有关键词决定最终得分计算方式
    if not keywords:
        # 如果没有提供关键词，仅使用语义相似度得分
        # 这种情况通常发生在测试数据中没有提取关键词，或者答案本身不包含实体关键词
        final_score = semantic_score
    else:
        # 加权计算最终得分：20%关键词得分 + 80%语义得分
        final_score = 0.2 * keyword_score + 0.8 * semantic_score

    return {
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,  # 可能为None（当keywords为空时）
        "final_score": final_score,
        "matched_keywords": join_keywords,  # 添加匹配到的关键词列表
        "total_keywords": len(keywords),  # 添加总关键词数
    }


def _calc_jaccard(list_a: List[str], list_b: List[str], threshold: float = 0.3) -> int:
    """
    计算两个列表的Jaccard相似度并根据阈值返回二元结果

    Args:
        list_a: 第一个列表
        list_b: 第二个列表
        threshold: 判断相似度的阈值，默认为0.3

    Returns:
        如果匹配度高于阈值返回1，否则返回0
    """
    size_a, size_b = len(list_a), len(list_b)
    # 找出list_a中同时存在于list_b的元素
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    # 计算匹配度得分：共同元素数量占list_b的比例
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0


def calculate_retrieval_accuracy(
    retrieved_docs: List[Document],
    ground_truth_docs: Optional[List[Document]] = None,
    topk: int = 5,
) -> Dict[str, float]:
    """
    计算检索准确率

    注意：此函数需要ground_truth_docs作为输入。如果测试数据中没有ground_truth文档信息，
    此函数将返回0.0。可以通过检查检索到的文档中是否包含答案关键词来间接评估检索质量。

    Args:
        retrieved_docs: 检索到的文档列表
        ground_truth_docs: 标准答案文档列表（可选，如果为None则返回0.0）
        topk: Top-K值

    Returns:
        包含Top-K准确率的字典
    """
    if not retrieved_docs:
        return {"topk_accuracy": 0.0}

    if ground_truth_docs is None or not ground_truth_docs:
        # 如果没有ground_truth_docs，返回0.0并记录警告
        return {"topk_accuracy": 0.0}

    # 获取标准答案文档的unique_id集合
    gt_ids = {doc.metadata.get("unique_id") for doc in ground_truth_docs}

    # 检查Top-K检索结果中是否包含标准答案
    retrieved_ids = {doc.metadata.get("unique_id") for doc in retrieved_docs[:topk]}

    hit_count = len(gt_ids & retrieved_ids)
    accuracy = hit_count / len(gt_ids) if gt_ids else 0.0

    return {"topk_accuracy": accuracy}


def calculate_reranking_metrics(
    query: str,
    ranked_docs: List[Document],
    ground_truth_docs: Optional[List[Document]] = None,
    topk: int = 5,
) -> Dict[str, float]:
    """
    计算重排序效果指标

    注意：此函数需要ground_truth_docs作为输入。如果测试数据中没有ground_truth文档信息，
    此函数将返回0.0。可以通过检查重排序后的文档相关性来间接评估重排序效果。

    Args:
        query: 查询问题
        ranked_docs: 重排序后的文档列表
        ground_truth_docs: 标准答案文档列表（可选，如果为None则返回0.0）
        topk: Top-K值

    Returns:
        包含NDCG@K、MRR、Precision@K、Recall@K的字典
    """
    if not ranked_docs:
        return {
            "ndcg@k": 0.0,
            "mrr": 0.0,
            "precision@k": 0.0,
            "recall@k": 0.0,
        }

    if ground_truth_docs is None or not ground_truth_docs:
        # 如果没有ground_truth_docs，返回0.0
        return {
            "ndcg@k": 0.0,
            "mrr": 0.0,
            "precision@k": 0.0,
            "recall@k": 0.0,
        }

    gt_ids = {doc.metadata.get("unique_id") for doc in ground_truth_docs}

    # 计算MRR（平均倒数排名）
    mrr = 0.0
    for idx, doc in enumerate(ranked_docs[:topk], 1):
        if doc.metadata.get("unique_id") in gt_ids:
            mrr = 1.0 / idx
            break

    # 计算Precision@K和Recall@K
    retrieved_ids = {doc.metadata.get("unique_id") for doc in ranked_docs[:topk]}
    hit_count = len(gt_ids & retrieved_ids)
    precision = hit_count / topk if topk > 0 else 0.0
    recall = hit_count / len(gt_ids) if gt_ids else 0.0

    # 计算NDCG@K（简化版本）
    ndcg = _calculate_ndcg(ranked_docs[:topk], gt_ids, topk)

    return {
        "ndcg@k": ndcg,
        "mrr": mrr,
        "precision@k": precision,
        "recall@k": recall,
    }


def _calculate_ndcg(ranked_docs: List[Document], gt_ids: set, k: int) -> float:
    """计算NDCG@K（简化版本）"""
    dcg = 0.0
    for idx, doc in enumerate(ranked_docs[:k], 1):
        if doc.metadata.get("unique_id") in gt_ids:
            dcg += 1.0 / np.log2(idx + 1)

    # 理想DCG（IDCG）
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gt_ids), k) + 1))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_generation_quality(
    pred: str,
    gold: str,
    use_rouge: bool = True,
) -> Dict[str, float]:
    """
    计算生成质量指标

    Args:
        pred: 生成答案
        gold: 标准答案
        use_rouge: 是否计算ROUGE分数

    Returns:
        包含BLEU、ROUGE的字典（已移除exact_match）
    """

    # BLEU分数（使用jieba分词）
    bleu = _calculate_bleu(pred, gold)

    # ROUGE分数（使用jieba分词）
    rouge = {}
    if use_rouge:
        rouge = _calculate_rouge_chinese(pred, gold)

    # 语义相似度（需要在外部使用semantic_model计算）
    # 这里先返回0，实际使用时在rag_evaluator中计算

    result = {
        "bleu": bleu,
        **rouge,
    }

    return result


def _calculate_exact_match(pred: str, gold: str) -> float:
    """计算精确匹配（去除标点后比较）"""
    # 去除标点符号和空白字符
    pred_clean = re.sub(r"[。，、；：！？\s]+", "", pred.strip())
    gold_clean = re.sub(r"[。，、；：！？\s]+", "", gold.strip())

    return 1.0 if pred_clean == gold_clean else 0.0


def _calculate_bleu(pred: str, gold: str) -> float:
    """计算BLEU分数（使用jieba分词）"""
    pred_tokens = list(jieba.cut(pred))
    gold_tokens = list(jieba.cut(gold))

    if not gold_tokens:
        return 0.0

    # 计算1-gram精确匹配
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    matches = sum((pred_counter & gold_counter).values())
    precision = matches / len(pred_tokens) if pred_tokens else 0.0

    return precision


def _calculate_rouge_chinese(pred: str, gold: str) -> Dict[str, float]:
    """计算ROUGE分数（使用jieba分词）"""
    pred_tokens = list(jieba.cut(pred))
    gold_tokens = list(jieba.cut(gold))

    if not gold_tokens:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    # ROUGE-1
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    matches_1 = sum((pred_counter & gold_counter).values())
    rouge_1 = matches_1 / len(gold_tokens) if gold_tokens else 0.0

    # ROUGE-2
    pred_bigrams = [tuple(pred_tokens[i : i + 2]) for i in range(len(pred_tokens) - 1)]
    gold_bigrams = [tuple(gold_tokens[i : i + 2]) for i in range(len(gold_tokens) - 1)]
    pred_bigram_counter = Counter(pred_bigrams)
    gold_bigram_counter = Counter(gold_bigrams)
    matches_2 = sum((pred_bigram_counter & gold_bigram_counter).values())
    rouge_2 = matches_2 / len(gold_bigrams) if gold_bigrams else 0.0

    # ROUGE-L（最长公共子序列，简化版本）
    rouge_l = (
        _calculate_lcs(pred_tokens, gold_tokens) / len(gold_tokens)
        if gold_tokens
        else 0.0
    )

    return {
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l,
    }


def _calculate_lcs(seq1: List[str], seq2: List[str]) -> int:
    """计算最长公共子序列长度"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def calculate_response_time(
    timings: Dict[str, float],
) -> Dict[str, float]:
    """
    计算响应时间指标

    Args:
        timings: 各阶段耗时字典，包含：
            - retrieval_time: 检索时间
            - reranking_time: 重排序时间
            - generation_time: 生成时间

    Returns:
        包含各阶段耗时和总时间的字典
    """
    total_time = (
        timings.get("retrieval_time", 0.0)
        + timings.get("reranking_time", 0.0)
        + timings.get("generation_time", 0.0)
    )

    return {
        "retrieval_time": timings.get("retrieval_time", 0.0),
        "reranking_time": timings.get("reranking_time", 0.0),
        "generation_time": timings.get("generation_time", 0.0),
        "total_time": total_time,
    }
