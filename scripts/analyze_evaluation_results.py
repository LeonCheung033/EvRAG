#!/usr/bin/env python3
"""
分析评估结果，诊断为什么数值偏低
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

def analyze_results(summary_path: Path) -> Dict[str, Any]:
    """分析评估结果"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    analysis = {
        "total_samples": summary.get("total_samples", 0),
        "semantic_keyword_score": summary.get("semantic_keyword_score", {}),
        "comprehensive_accuracy": summary.get("comprehensive_accuracy", {}),
        "ragas_scores": summary.get("ragas_scores"),
        "generation_quality": summary.get("generation_quality", {}),
    }
    
    return analysis

def diagnose_low_scores(summary_path: Path, results_path: Path) -> Dict[str, Any]:
    """诊断为什么分数偏低"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 过滤有效结果
    valid_results = [r for r in results if "error" not in r]
    
    diagnosis = {
        "total_samples": len(valid_results),
        "semantic_scores": [],
        "keyword_scores": [],
        "final_scores": [],
        "low_score_samples": [],
    }
    
    for result in valid_results:
        metrics = result.get("metrics", {})
        semantic_keyword = metrics.get("semantic_keyword_score", {})
        
        semantic_score = semantic_keyword.get("semantic_score", 0.0)
        keyword_score = semantic_keyword.get("keyword_score")
        final_score = semantic_keyword.get("final_score", 0.0)
        
        diagnosis["semantic_scores"].append(semantic_score)
        diagnosis["keyword_scores"].append(keyword_score if keyword_score is not None else 0.0)
        diagnosis["final_scores"].append(final_score)
        
        # 找出低分样本（final_score < 0.85）
        if final_score < 0.85:
            diagnosis["low_score_samples"].append({
                "unique_id": result.get("unique_id"),
                "question": result.get("question", "")[:50] + "...",
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "final_score": final_score,
                "ground_truth": result.get("ground_truth_answer", "")[:50] + "...",
                "generated": result.get("generated_answer", "")[:50] + "...",
            })
    
    # 计算统计信息
    if diagnosis["semantic_scores"]:
        diagnosis["semantic_stats"] = {
            "mean": float(np.mean(diagnosis["semantic_scores"])),
            "std": float(np.std(diagnosis["semantic_scores"])),
            "min": float(np.min(diagnosis["semantic_scores"])),
            "max": float(np.max(diagnosis["semantic_scores"])),
        }
    
    if diagnosis["final_scores"]:
        diagnosis["final_stats"] = {
            "mean": float(np.mean(diagnosis["final_scores"])),
            "std": float(np.std(diagnosis["final_scores"])),
            "min": float(np.min(diagnosis["final_scores"])),
            "max": float(np.max(diagnosis["final_scores"])),
        }
    
    return diagnosis

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_evaluation_results.py <evaluation_summary.json> [evaluation_results.json]")
        sys.exit(1)
    
    summary_path = Path(sys.argv[1])
    results_path = Path(sys.argv[2]) if len(sys.argv) > 2 else summary_path.parent / summary_path.name.replace("_summary.json", ".json")
    
    if not summary_path.exists():
        print(f"错误: 文件不存在: {summary_path}")
        sys.exit(1)
    
    print("="*60)
    print("评估结果分析")
    print("="*60)
    
    # 分析汇总结果
    analysis = analyze_results(summary_path)
    
    print(f"\n总样本数: {analysis['total_samples']}")
    print(f"\n语义相似度+关键词得分:")
    semantic = analysis.get("semantic_keyword_score", {})
    print(f"  平均值: {semantic.get('mean', 0.0):.4f}")
    print(f"  标准差: {semantic.get('std', 0.0):.4f}")
    
    comprehensive = analysis.get("comprehensive_accuracy", {})
    if comprehensive:
        print(f"\n综合准确率:")
        print(f"  得分: {comprehensive.get('score', 0.0):.4f} ({comprehensive.get('score', 0.0)*100:.2f}%)")
        print(f"  语义+关键词得分: {comprehensive.get('semantic_keyword_score', 0.0):.4f}")
        if comprehensive.get('ragas_score'):
            print(f"  RAGas得分: {comprehensive.get('ragas_score', 0.0):.4f}")
    
    ragas = analysis.get("ragas_scores")
    if ragas:
        print(f"\nRAGas指标:")
        print(f"  ContextRecall: {ragas.get('context_recall', 0.0):.4f}")
        print(f"  ContextPrecision: {ragas.get('context_precision', 0.0):.4f}")
        print(f"  平均得分: {ragas.get('average', 0.0):.4f}")
    
    # 诊断低分原因
    if results_path.exists():
        print(f"\n{'='*60}")
        print("低分样本诊断")
        print("="*60)
        
        diagnosis = diagnose_low_scores(summary_path, results_path)
        
        if diagnosis.get("semantic_stats"):
            print(f"\n语义相似度统计:")
            stats = diagnosis["semantic_stats"]
            print(f"  平均值: {stats['mean']:.4f}")
            print(f"  最小值: {stats['min']:.4f}")
            print(f"  最大值: {stats['max']:.4f}")
            print(f"  标准差: {stats['std']:.4f}")
        
        if diagnosis.get("final_stats"):
            print(f"\n最终得分统计:")
            stats = diagnosis["final_stats"]
            print(f"  平均值: {stats['mean']:.4f}")
            print(f"  最小值: {stats['min']:.4f}")
            print(f"  最大值: {stats['max']:.4f}")
            print(f"  标准差: {stats['std']:.4f}")
        
        low_score_count = len(diagnosis.get("low_score_samples", []))
        print(f"\n低分样本数 (final_score < 0.85): {low_score_count}/{diagnosis['total_samples']}")
        
        if low_score_count > 0 and low_score_count <= 10:
            print(f"\n低分样本详情:")
            for i, sample in enumerate(diagnosis["low_score_samples"][:10], 1):
                print(f"\n  {i}. ID: {sample['unique_id']}")
                print(f"     问题: {sample['question']}")
                print(f"     语义得分: {sample['semantic_score']:.4f}")
                print(f"     关键词得分: {sample['keyword_score']}")
                print(f"     最终得分: {sample['final_score']:.4f}")
                print(f"     标准答案: {sample['ground_truth']}")
                print(f"     生成答案: {sample['generated']}")
    
    print(f"\n{'='*60}")
    print("可能的原因分析:")
    print("="*60)
    print("1. 语义相似度偏低:")
    print("   - 生成答案可能缺少关键信息")
    print("   - 生成答案的表达方式与标准答案差异较大")
    print("   - 语义模型对细节敏感")
    print("\n2. 关键词匹配问题:")
    print("   - 测试数据中keywords字段可能为空")
    print("   - 生成答案中可能未包含所有关键词")
    print("\n3. RAGas评估:")
    print("   - ContextPrecision可能偏低（检索到的上下文相关性不够）")
    print("   - 需要检查检索和重排序的效果")
    print("\n4. 样本数量:")
    print("   - 如果样本数较少，可能不具有代表性")
    print("   - 建议使用全量测试数据（750条）进行评估")

if __name__ == "__main__":
    main()

