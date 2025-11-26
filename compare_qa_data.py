#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QA数据对比工具：对比新项目（EvRAG）和原项目（EVRAG）的QA数据文件
输出对比报告到logs目录
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter


def load_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSONL格式文件"""
    data = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
    return data


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSON格式文件"""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return [data] if data else []


def analyze_qa_pairs(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析QA对数据"""
    stats = {
        "total_count": len(qa_pairs),
        "positive_count": 0,
        "negative_count": 0,
        "with_keywords_count": 0,
        "unique_questions": set(),
        "unique_answers": set(),
        "question_lengths": [],
        "answer_lengths": [],
        "keywords_count": 0,
    }
    
    for qa in qa_pairs:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        keywords = qa.get("keywords", [])
        
        stats["unique_questions"].add(question)
        stats["unique_answers"].add(answer)
        stats["question_lengths"].append(len(question))
        stats["answer_lengths"].append(len(answer))
        
        if answer == "无答案":
            stats["negative_count"] += 1
        else:
            stats["positive_count"] += 1
        
        if keywords:
            stats["with_keywords_count"] += 1
            stats["keywords_count"] += len(keywords)
    
    stats["unique_questions_count"] = len(stats["unique_questions"])
    stats["unique_answers_count"] = len(stats["unique_answers"])
    stats["avg_question_length"] = sum(stats["question_lengths"]) / len(stats["question_lengths"]) if stats["question_lengths"] else 0
    stats["avg_answer_length"] = sum(stats["answer_lengths"]) / len(stats["answer_lengths"]) if stats["answer_lengths"] else 0
    stats["avg_keywords_count"] = stats["keywords_count"] / stats["with_keywords_count"] if stats["with_keywords_count"] > 0 else 0
    
    # 清理不需要的字段
    del stats["unique_questions"]
    del stats["unique_answers"]
    del stats["question_lengths"]
    del stats["answer_lengths"]
    
    return stats


def compare_qa_pair_files(new_file: Path, old_file: Path) -> Dict[str, Any]:
    """对比qa_pair.json文件（JSONL格式）"""
    new_data = load_jsonl_file(new_file)
    old_data = load_jsonl_file(old_file)
    
    new_unique_ids = {item.get("unique_id") for item in new_data if item.get("unique_id")}
    old_unique_ids = {item.get("unique_id") for item in old_data if item.get("unique_id")}
    common_unique_ids = new_unique_ids & old_unique_ids
    
    return {
        "new_count": len(new_data),
        "old_count": len(old_data),
        "new_unique_ids_count": len(new_unique_ids),
        "old_unique_ids_count": len(old_unique_ids),
        "common_unique_ids_count": len(common_unique_ids),
        "new_only_count": len(new_unique_ids - old_unique_ids),
        "old_only_count": len(old_unique_ids - new_unique_ids),
        "diff_ratio": abs(len(new_data) - len(old_data)) / len(old_data) if len(old_data) > 0 else 0,
    }


def compare_json_qa_files(new_file: Path, old_file: Path) -> Dict[str, Any]:
    """对比JSON格式的QA文件（train_qa_pair.json, test_qa_pair.json）"""
    new_data = load_json_file(new_file)
    old_data = load_json_file(old_file)
    
    # 分析数据
    new_stats = analyze_qa_pairs(new_data)
    old_stats = analyze_qa_pairs(old_data)
    
    # 对比unique_id
    new_unique_ids = {item.get("unique_id") for item in new_data if item.get("unique_id")}
    old_unique_ids = {item.get("unique_id") for item in old_data if item.get("unique_id")}
    common_unique_ids = new_unique_ids & old_unique_ids
    
    return {
        "new_count": len(new_data),
        "old_count": len(old_data),
        "new_stats": new_stats,
        "old_stats": old_stats,
        "common_unique_ids_count": len(common_unique_ids),
        "new_only_count": len(new_unique_ids - old_unique_ids),
        "old_only_count": len(old_unique_ids - new_unique_ids),
        "diff_ratio": abs(len(new_data) - len(old_data)) / len(old_data) if len(old_data) > 0 else 0,
        "new_sample": new_data[:3] if len(new_data) > 0 else [],
        "old_sample": old_data[:3] if len(old_data) > 0 else [],
    }


def compare_keywords_files(new_file: Path, old_file: Path) -> Dict[str, Any]:
    """对比test_keywords_pair.json文件（JSONL格式）"""
    new_data = load_jsonl_file(new_file)
    old_data = load_jsonl_file(old_file)
    
    new_answers = {item.get("unique_id") for item in new_data if item.get("unique_id")}
    old_answers = {item.get("unique_id") for item in old_data if item.get("unique_id")}
    common_answers = new_answers & old_answers
    
    # 统计关键词数量
    new_keywords_count = sum(len(item.get("raw_resp", "").split(",")) for item in new_data)
    old_keywords_count = sum(len(item.get("raw_resp", "").split(",")) for item in old_data)
    
    return {
        "new_count": len(new_data),
        "old_count": len(old_data),
        "new_answers_count": len(new_answers),
        "old_answers_count": len(old_answers),
        "common_answers_count": len(common_answers),
        "new_only_count": len(new_answers - old_answers),
        "old_only_count": len(old_answers - new_answers),
        "new_keywords_count": new_keywords_count,
        "old_keywords_count": old_keywords_count,
        "avg_new_keywords": new_keywords_count / len(new_data) if len(new_data) > 0 else 0,
        "avg_old_keywords": old_keywords_count / len(old_data) if len(old_data) > 0 else 0,
        "diff_ratio": abs(len(new_data) - len(old_data)) / len(old_data) if len(old_data) > 0 else 0,
    }


def compare_expand_files(new_file: Path, old_file: Path) -> Dict[str, Any]:
    """对比expand_qa_pair.json文件（JSONL格式）"""
    new_data = load_jsonl_file(new_file)
    old_data = load_jsonl_file(old_file)
    
    new_questions = {item.get("unique_id") for item in new_data if item.get("unique_id")}
    old_questions = {item.get("unique_id") for item in old_data if item.get("unique_id")}
    common_questions = new_questions & old_questions
    
    # 统计每个问题生成的改写问题数量
    new_expand_counts = []
    old_expand_counts = []
    for item in new_data:
        raw_resp = item.get("raw_resp", "")
        if raw_resp:
            new_expand_counts.append(len(raw_resp.split("\n")))
    for item in old_data:
        raw_resp = item.get("raw_resp", "")
        if raw_resp:
            old_expand_counts.append(len(raw_resp.split("\n")))
    
    return {
        "new_count": len(new_data),
        "old_count": len(old_data),
        "new_questions_count": len(new_questions),
        "old_questions_count": len(old_questions),
        "common_questions_count": len(common_questions),
        "new_only_count": len(new_questions - old_questions),
        "old_only_count": len(old_questions - new_questions),
        "avg_new_expand_count": sum(new_expand_counts) / len(new_expand_counts) if new_expand_counts else 0,
        "avg_old_expand_count": sum(old_expand_counts) / len(old_expand_counts) if old_expand_counts else 0,
        "diff_ratio": abs(len(new_data) - len(old_data)) / len(old_data) if len(old_data) > 0 else 0,
    }


def generate_comparison_report(
    new_project_dir: Path,
    old_project_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """生成对比报告"""
    
    # 定义文件路径
    files_to_compare = {
        "qa_pair.json": {
            "new_path": new_project_dir / "data/qa_pairs/qa_pair.json",
            "old_path": old_project_dir / "data/qa_pairs/qa_pair.json",
            "compare_func": compare_qa_pair_files,
        },
        "expand_qa_pair.json": {
            "new_path": new_project_dir / "data/qa_pairs/expand_qa_pair.json",
            "old_path": old_project_dir / "data/qa_pairs/expand_qa_pair.json",
            "compare_func": compare_expand_files,
        },
        "train_qa_pair.json": {
            "new_path": new_project_dir / "data/qa_pairs/train_qa_pair.json",
            "old_path": old_project_dir / "data/qa_pairs/train_qa_pair.json",
            "compare_func": compare_json_qa_files,
        },
        "test_qa_pair.json": {
            "new_path": new_project_dir / "data/qa_pairs/test_qa_pair.json",
            "old_path": old_project_dir / "data/qa_pairs/test_qa_pair.json",
            "compare_func": compare_json_qa_files,
        },
        "test_keywords_pair.json": {
            "new_path": new_project_dir / "data/qa_pairs/test_keywords_pair.json",
            "old_path": old_project_dir / "data/qa_pairs/test_keywords_pair.json",
            "compare_func": compare_keywords_files,
        },
    }
    
    # 执行对比
    comparison_results = {}
    for file_name, file_info in files_to_compare.items():
        new_path = file_info["new_path"]
        old_path = file_info["old_path"]
        compare_func = file_info["compare_func"]
        
        print(f"\n{'='*60}")
        print(f"Comparing: {file_name}")
        print(f"{'='*60}")
        
        if not new_path.exists():
            print(f"⚠️  New file not found: {new_path}")
            comparison_results[file_name] = {"status": "new_file_not_found"}
            continue
        
        if not old_path.exists():
            print(f"⚠️  Old file not found: {old_path}")
            comparison_results[file_name] = {"status": "old_file_not_found"}
            continue
        
        try:
            result = compare_func(new_path, old_path)
            comparison_results[file_name] = {
                "status": "compared",
                "result": result,
            }
            
            # 打印简要结果
            print(f"New count: {result.get('new_count', 0)}")
            print(f"Old count: {result.get('old_count', 0)}")
            if "diff_ratio" in result:
                print(f"Difference ratio: {result['diff_ratio']*100:.2f}%")
            if "common_unique_ids_count" in result:
                print(f"Common unique_ids: {result['common_unique_ids_count']}")
            if "common_questions_count" in result:
                print(f"Common questions: {result['common_questions_count']}")
            if "common_answers_count" in result:
                print(f"Common answers: {result['common_answers_count']}")
            
        except Exception as e:
            print(f"❌ Error comparing {file_name}: {e}")
            comparison_results[file_name] = {
                "status": "error",
                "error": str(e),
            }
    
    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "new_project": str(new_project_dir),
        "old_project": str(old_project_dir),
        "comparison_results": comparison_results,
    }
    
    # 保存报告
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"qa_data_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Comparison report saved to: {report_file}")
    print(f"{'='*60}")
    
    # 生成简要摘要
    print("\n📊 Summary:")
    print("-" * 60)
    for file_name, file_result in comparison_results.items():
        if file_result.get("status") == "compared":
            result = file_result.get("result", {})
            new_count = result.get("new_count", 0)
            old_count = result.get("old_count", 0)
            diff_ratio = result.get("diff_ratio", 0) * 100
            
            status_icon = "✅" if diff_ratio < 10 else "⚠️" if diff_ratio < 30 else "❌"
            print(f"{status_icon} {file_name}:")
            print(f"   New: {new_count}, Old: {old_count}, Diff: {diff_ratio:.2f}%")
        else:
            print(f"⚠️  {file_name}: {file_result.get('status', 'unknown')}")
    
    return report


if __name__ == "__main__":
    import sys
    
    # 默认路径
    new_project_dir = Path("/remote-home/share/liangZhang/EvRAG")
    old_project_dir = Path("/remote-home/share/liangZhang/EVRAG")
    output_dir = Path("/remote-home/share/liangZhang/EvRAG/logs")
    
    if len(sys.argv) > 1:
        new_project_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        old_project_dir = Path(sys.argv[2])
    if len(sys.argv) > 3:
        output_dir = Path(sys.argv[3])
    
    print("=" * 60)
    print("QA Data Comparison Tool")
    print("=" * 60)
    print(f"New project: {new_project_dir}")
    print(f"Old project: {old_project_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    
    generate_comparison_report(new_project_dir, old_project_dir, output_dir)

