#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reranker效果评估脚本

对比两种Reranker模型：
1. 基线Reranker (BGE-Reranker-v2-m3)
2. 微调后的Reranker (checkpoint_0, checkpoint_1)

计算指标：
- Top1 Recall: 第一个文档是否是最相关的（content[0]是否排在第一位）
- Top3 Recall: 前3个文档中是否包含最相关的文档
- MRR (Mean Reciprocal Rank): 平均倒数排名

参考: EVRAG/RAG-Retrieval/rag_retrieval/train/reranker/predict.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evrag.reranker import BGEReranker
from evrag.config import get_settings


def calculate_top1_recall(predicted_ranks: List[int], ground_truth_rank: int) -> bool:
    """计算Top1 Recall：第一个预测是否是最相关的"""
    return predicted_ranks[0] == ground_truth_rank if predicted_ranks else False


def calculate_top3_recall(predicted_ranks: List[int], ground_truth_rank: int) -> bool:
    """计算Top3 Recall：前3个预测中是否包含最相关的"""
    return ground_truth_rank in predicted_ranks[:3] if len(predicted_ranks) >= 3 else ground_truth_rank in predicted_ranks


def calculate_mrr(predicted_ranks: List[int], ground_truth_rank: int) -> float:
    """计算MRR：最相关文档的倒数排名"""
    try:
        rank = predicted_ranks.index(ground_truth_rank) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def evaluate_reranker(
    reranker: BGEReranker,
    test_data: List[Dict],
    method_name: str = "Reranker"
) -> Dict[str, float]:
    """评估Reranker模型"""
    print(f"\n评估 {method_name}...")
    
    total = 0
    top1_right = 0
    top3_right = 0
    mrr_scores = []
    
    for idx, item in enumerate(test_data):
        query = item["query"]
        contents = item["content"]  # 文档列表
        
        if not contents or len(contents) == 0:
            continue
        
        # 计算每个文档的分数
        scores = []
        try:
            # 将content列表转换为Document列表
            from langchain_core.documents import Document
            docs = [Document(page_content=content) for content in contents]
            
            # 使用reranker的rank方法计算分数
            # 我们需要获取分数，所以直接使用模型计算
            pairs = [(query, content) for content in contents]
            
            inputs = reranker.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=reranker.max_length,
                return_tensors="pt",
            )
            
            if reranker.device.startswith("cuda:") and torch.cuda.is_available():
                inputs = {k: v.to(reranker.device) for k, v in inputs.items()}
            elif reranker.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 计算相关性分数
            with torch.no_grad():
                outputs = reranker.model(**inputs)
                batch_scores = outputs.logits.squeeze(dim=-1)
            
            scores = batch_scores.detach().cpu().numpy().tolist()
            if not isinstance(scores, list):
                scores = [float(scores)]
        except Exception as e:
            print(f"  警告: 计算分数失败 (query {idx}): {e}")
            scores = [0.0] * len(contents)
        
        # 根据分数排序，得到排名
        # 分数越高，排名越靠前（rank越小）
        # ranked_indices[i] 表示第i个文档的排名位置（0表示第一位）
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # 假设第一个文档（content[0]）是最相关的（ground truth）
        # 这是根据predict.py的逻辑：content[0]应该是正样本
        ground_truth_rank = 0  # 第一个文档的索引
        
        # 计算指标
        top1_right += calculate_top1_recall(ranked_indices, ground_truth_rank)
        top3_right += calculate_top3_recall(ranked_indices, ground_truth_rank)
        mrr_scores.append(calculate_mrr(ranked_indices, ground_truth_rank))
        
        total += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  已处理 {idx + 1}/{len(test_data)} 条数据...")
    
    results = {
        "total": total,
        "top1_recall": top1_right / total if total > 0 else 0.0,
        "top3_recall": top3_right / total if total > 0 else 0.0,
        "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
    }
    
    print(f"  Total: {total}")
    print(f"  Top1 Recall: {results['top1_recall']:.4f}")
    print(f"  Top3 Recall: {results['top3_recall']:.4f}")
    print(f"  MRR: {results['mrr']:.4f}")
    
    return results


def main():
    """主函数"""
    print("=" * 80)
    print("Reranker效果评估 - 基线 vs 微调模型对比")
    print("=" * 80)
    
    # 加载测试数据
    test_data_path = project_root / "data" / "rerank_data" / "test.json"
    print(f"\n加载测试数据: {test_data_path}")
    
    if not test_data_path.exists():
        print(f"✗ 测试数据文件不存在: {test_data_path}")
        return
    
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 确保content是列表格式
                if isinstance(item.get('content'), str):
                    print(f"⚠ 警告: 数据格式不正确，content应该是列表")
                    continue
                test_data.append(item)
    
    print(f"测试数据量: {len(test_data)} 条")
    
    if len(test_data) == 0:
        print("✗ 没有有效的测试数据")
        return
    
    # 获取配置
    settings = get_settings()
    
    # 评估结果
    all_results = {}
    
    # 1. 评估基线Reranker
    print("\n" + "=" * 80)
    print("1. 基线Reranker评估")
    print("=" * 80)
    baseline_path = settings.bge_reranker_model_path
    if baseline_path and Path(baseline_path).exists():
        try:
            print(f"  加载模型: {baseline_path}")
            baseline_reranker = BGEReranker(model_path=baseline_path)
            all_results["基线Reranker"] = evaluate_reranker(
                baseline_reranker, test_data, "基线Reranker"
            )
        except Exception as e:
            print(f"✗ 基线Reranker评估失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ 基线Reranker路径不存在: {baseline_path}")
    
    # 2. 评估微调后的Reranker - checkpoint_0
    print("\n" + "=" * 80)
    print("3. 微调Reranker评估 - Checkpoint 0")
    print("=" * 80)
    checkpoint_0_path = project_root / "models" / "finetuned" / "bge_reranker" / "runs" / "checkpoints" / "checkpoint_0"
    if checkpoint_0_path.exists():
        try:
            print(f"  加载模型: {checkpoint_0_path}")
            checkpoint_0_reranker = BGEReranker(model_path=checkpoint_0_path)
            all_results["微调Reranker-Checkpoint0"] = evaluate_reranker(
                checkpoint_0_reranker, test_data, "微调Reranker-Checkpoint0"
            )
        except Exception as e:
            print(f"✗ Checkpoint 0评估失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Checkpoint 0路径不存在: {checkpoint_0_path}")
    
    # 3. 评估微调后的Reranker - checkpoint_1
    print("\n" + "=" * 80)
    print("4. 微调Reranker评估 - Checkpoint 1")
    print("=" * 80)
    checkpoint_1_path = project_root / "models" / "finetuned" / "bge_reranker" / "runs" / "checkpoints" / "checkpoint_1"
    if checkpoint_1_path.exists():
        try:
            print(f"  加载模型: {checkpoint_1_path}")
            checkpoint_1_reranker = BGEReranker(model_path=checkpoint_1_path)
            all_results["微调Reranker-Checkpoint1"] = evaluate_reranker(
                checkpoint_1_reranker, test_data, "微调Reranker-Checkpoint1"
            )
        except Exception as e:
            print(f"✗ Checkpoint 1评估失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Checkpoint 1路径不存在: {checkpoint_1_path}")
    
    # 输出对比结果
    print("\n" + "=" * 80)
    print("评估结果对比")
    print("=" * 80)
    
    if all_results:
        print(f"\n{'方法':<35} {'Top1 Recall':<15} {'Top3 Recall':<15} {'MRR':<15}")
        print("-" * 80)
        for method, results in all_results.items():
            print(f"{method:<35} {results['top1_recall']:<15.4f} {results['top3_recall']:<15.4f} {results['mrr']:<15.4f}")
        
        # 计算相对改进
        if "基线Reranker" in all_results and "微调Reranker-Checkpoint0" in all_results:
            baseline = all_results["基线Reranker"]
            checkpoint0 = all_results["微调Reranker-Checkpoint0"]
            print(f"\n{'指标':<20} {'基线':<15} {'Checkpoint0':<15} {'改进':<15}")
            print("-" * 80)
            print(f"{'Top1 Recall':<20} {baseline['top1_recall']:<15.4f} {checkpoint0['top1_recall']:<15.4f} {checkpoint0['top1_recall'] - baseline['top1_recall']:<15.4f}")
            print(f"{'Top3 Recall':<20} {baseline['top3_recall']:<15.4f} {checkpoint0['top3_recall']:<15.4f} {checkpoint0['top3_recall'] - baseline['top3_recall']:<15.4f}")
            print(f"{'MRR':<20} {baseline['mrr']:<15.4f} {checkpoint0['mrr']:<15.4f} {checkpoint0['mrr'] - baseline['mrr']:<15.4f}")
        
        # 保存结果
        output_path = project_root / "rag_test_reports" / "reranker_comparison_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 结果已保存到: {output_path}")
    else:
        print("\n✗ 没有可用的评估结果")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
