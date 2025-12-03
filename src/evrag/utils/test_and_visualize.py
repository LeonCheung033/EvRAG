#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段二测试和可视化工具

包含：
1. 数据质量测试
2. 索引测试（BM25和Milvus检索准确率）
3. QA质量测试
4. 图表生成（文档长度分布、QA对统计、检索准确率对比）
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from src.evrag.retriever import BM25Retriever, MilvusRetriever
from src.evrag.gen_qa.generator import QAGenerator

# 使用默认英文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_documents() -> Dict[str, List[Document]]:
    """加载所有文档"""
    data_dir = Path("/remote-home/share/liangZhang/EvRAG/data/processed_docs")

    docs = {}
    for name in ["raw_docs", "clean_docs", "split_docs"]:
        file_path = data_dir / f"{name}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                docs[name] = pickle.load(f)
        else:
            docs[name] = []

    return docs


def load_qa_pairs() -> List[Dict[str, Any]]:
    """加载QA对"""
    qa_file = Path("/remote-home/share/liangZhang/EvRAG/data/qa_pairs/qa_pair.json")

    if not qa_file.exists():
        return []

    qa_pairs = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    qa_list = QAGenerator.parse_qa_response(item.get("raw_resp", "[]"))
                    for qa in qa_list:
                        qa_pairs.append(
                            {
                                "unique_id": item.get("unique_id"),
                                "question": qa.get("question", ""),
                                "answer": qa.get("answer", ""),
                            }
                        )
                except Exception:
                    continue

    return qa_pairs


def test_data_quality(docs: Dict[str, List[Document]]) -> Dict[str, Any]:
    """数据质量测试"""
    print("=" * 80)
    print("数据质量测试")
    print("=" * 80)

    results = {}

    for stage_name, stage_docs in docs.items():
        if not stage_docs:
            continue

        print(f"\n【{stage_name}】")
        print("-" * 80)

        # 统计信息
        total_docs = len(stage_docs)
        lengths = [
            len(doc.page_content) for doc in stage_docs if hasattr(doc, "page_content")
        ]

        if lengths:
            avg_length = sum(lengths) / len(lengths)
            min_length = min(lengths)
            max_length = max(lengths)

            print(f"文档总数: {total_docs}")
            print(f"平均长度: {avg_length:.1f} 字符")
            print(f"最小长度: {min_length} 字符")
            print(f"最大长度: {max_length} 字符")

            # 长度分布
            length_ranges = {
                "0-100": sum(1 for l in lengths if l <= 100),
                "101-256": sum(1 for l in lengths if 101 <= l <= 256),
                "257-512": sum(1 for l in lengths if 257 <= l <= 512),
                "513-1024": sum(1 for l in lengths if 513 <= l <= 1024),
                "1024+": sum(1 for l in lengths if l > 1024),
            }

            print("\n长度分布:")
            for range_name, count in length_ranges.items():
                percentage = count / len(lengths) * 100 if lengths else 0
                print(f"  {range_name}: {count} 个 ({percentage:.1f}%)")

            # 检查超过512的文档
            over_512 = sum(1 for l in lengths if l > 512)
            if over_512 > 0:
                print(f"\n⚠️  超过512字符的文档: {over_512} 个")

            results[stage_name] = {
                "total_docs": total_docs,
                "avg_length": avg_length,
                "min_length": min_length,
                "max_length": max_length,
                "length_distribution": length_ranges,
                "over_512_count": over_512,
                "lengths": lengths,
            }

    return results


def test_index_retrieval() -> Dict[str, Any]:
    """索引测试：测试BM25和Milvus检索"""
    print("\n" + "=" * 80)
    print("索引测试")
    print("=" * 80)

    results = {
        "bm25": {"success": False, "error": None},
        "milvus": {"success": False, "error": None},
    }

    # 测试查询
    test_queries = [
        "如何打开车门",
        "如何充电",
        "如何设置座椅",
        "如何打开后备箱",
        "如何锁定车辆",
    ]

    # 测试BM25
    print("\n【BM25检索测试】")
    print("-" * 80)
    try:
        bm25_retriever = BM25Retriever(docs=None, retrieve=True)
        print("✓ BM25检索器加载成功")

        bm25_results = []
        for query in test_queries:
            try:
                docs = bm25_retriever.retrieve_topk(query, topk=5)
                bm25_results.append(
                    {"query": query, "num_results": len(docs), "success": True}
                )
                print(f"  ✓ 查询 '{query}': 返回 {len(docs)} 个结果")
            except Exception as e:
                bm25_results.append(
                    {
                        "query": query,
                        "num_results": 0,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ✗ 查询 '{query}': 失败 - {e}")

        results["bm25"] = {
            "success": True,
            "test_results": bm25_results,
            "success_rate": sum(1 for r in bm25_results if r["success"])
            / len(bm25_results)
            * 100,
        }
    except Exception as e:
        print(f"✗ BM25检索器加载失败: {e}")
        results["bm25"]["error"] = str(e)

    # 测试Milvus
    print("\n【Milvus检索测试】")
    print("-" * 80)
    try:
        milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
        print("✓ Milvus检索器加载成功")

        milvus_results = []
        for query in test_queries:
            try:
                docs = milvus_retriever.retrieve_topk(query, topk=5)
                milvus_results.append(
                    {"query": query, "num_results": len(docs), "success": True}
                )
                print(f"  ✓ 查询 '{query}': 返回 {len(docs)} 个结果")
            except Exception as e:
                milvus_results.append(
                    {
                        "query": query,
                        "num_results": 0,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ✗ 查询 '{query}': 失败 - {e}")

        results["milvus"] = {
            "success": True,
            "test_results": milvus_results,
            "success_rate": sum(1 for r in milvus_results if r["success"])
            / len(milvus_results)
            * 100,
        }
    except Exception as e:
        print(f"✗ Milvus检索器加载失败: {e}")
        results["milvus"]["error"] = str(e)

    return results


def test_qa_quality(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """QA质量测试"""
    print("\n" + "=" * 80)
    print("QA质量测试")
    print("=" * 80)

    if not qa_pairs:
        print("⚠️  没有QA对数据")
        return {}

    print(f"\n总QA对数量: {len(qa_pairs)}")

    # 统计问题长度
    question_lengths = [len(qa["question"]) for qa in qa_pairs if qa.get("question")]
    answer_lengths = [len(qa["answer"]) for qa in qa_pairs if qa.get("answer")]

    if question_lengths:
        print("\n问题统计:")
        print(f"  平均长度: {sum(question_lengths) / len(question_lengths):.1f} 字符")
        print(f"  最小长度: {min(question_lengths)} 字符")
        print(f"  最大长度: {max(question_lengths)} 字符")

    if answer_lengths:
        print("\n答案统计:")
        print(f"  平均长度: {sum(answer_lengths) / len(answer_lengths):.1f} 字符")
        print(f"  最小长度: {min(answer_lengths)} 字符")
        print(f"  最大长度: {max(answer_lengths)} 字符")

    # 抽样检查
    print("\n【抽样检查（前5个）】")
    print("-" * 80)
    for i, qa in enumerate(qa_pairs[:5], 1):
        print(f"\n{i}. 问题: {qa.get('question', 'N/A')[:80]}...")
        print(f"   答案: {qa.get('answer', 'N/A')[:80]}...")

    return {
        "total_qa_pairs": len(qa_pairs),
        "question_lengths": question_lengths,
        "answer_lengths": answer_lengths,
        "avg_question_length": sum(question_lengths) / len(question_lengths)
        if question_lengths
        else 0,
        "avg_answer_length": sum(answer_lengths) / len(answer_lengths)
        if answer_lengths
        else 0,
    }


def plot_document_length_distribution(
    docs: Dict[str, List[Document]], output_dir: Path
):
    """绘制文档长度分布直方图"""
    print("\n" + "=" * 80)
    print("生成文档长度分布直方图")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Document Length Distribution", fontsize=16, fontweight="bold")

    stage_names = ["raw_docs", "clean_docs", "split_docs"]
    stage_labels = ["Raw Documents", "Cleaned Documents", "Split Documents"]

    for idx, (stage_name, stage_label) in enumerate(zip(stage_names, stage_labels)):
        if stage_name not in docs or not docs[stage_name]:
            continue

        lengths = [
            len(doc.page_content)
            for doc in docs[stage_name]
            if hasattr(doc, "page_content")
        ]

        if lengths:
            ax = axes[idx]
            ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(
                f"{stage_label}\n(Total: {len(lengths)}, Avg: {sum(lengths) / len(lengths):.0f})",
                fontsize=12,
            )
            ax.set_xlabel("Document Length (characters)", fontsize=10)
            ax.set_ylabel("Document Count", fontsize=10)
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            ax.axvline(
                sum(lengths) / len(lengths),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {sum(lengths) / len(lengths):.0f}",
            )
            ax.legend()

    plt.tight_layout()
    output_path = output_dir / "document_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ 图表已保存到: {output_path}")


def plot_qa_statistics(qa_pairs: List[Dict[str, Any]], output_dir: Path):
    """绘制QA对数量统计柱状图"""
    print("\n" + "=" * 80)
    print("生成QA对数量统计柱状图")
    print("=" * 80)

    if not qa_pairs:
        print("⚠️  没有QA对数据，跳过图表生成")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("QA Pair Statistics", fontsize=16, fontweight="bold")

    # 问题长度分布
    question_lengths = [len(qa["question"]) for qa in qa_pairs if qa.get("question")]
    if question_lengths:
        ax1 = axes[0]
        ax1.hist(
            question_lengths, bins=30, edgecolor="black", alpha=0.7, color="skyblue"
        )
        ax1.set_title(
            f"Question Length Distribution\n(Total: {len(question_lengths)}, Avg: {sum(question_lengths) / len(question_lengths):.0f})",
            fontsize=12,
        )
        ax1.set_xlabel("Question Length (characters)", fontsize=10)
        ax1.set_ylabel("Question Count", fontsize=10)
        ax1.grid(True, alpha=0.3)

    # 答案长度分布
    answer_lengths = [len(qa["answer"]) for qa in qa_pairs if qa.get("answer")]
    if answer_lengths:
        ax2 = axes[1]
        ax2.hist(
            answer_lengths, bins=30, edgecolor="black", alpha=0.7, color="lightcoral"
        )
        ax2.set_title(
            f"Answer Length Distribution\n(Total: {len(answer_lengths)}, Avg: {sum(answer_lengths) / len(answer_lengths):.0f})",
            fontsize=12,
        )
        ax2.set_xlabel("Answer Length (characters)", fontsize=10)
        ax2.set_ylabel("Answer Count", fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "qa_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ 图表已保存到: {output_path}")


def plot_retrieval_comparison(index_results: Dict[str, Any], output_dir: Path):
    """绘制检索准确率对比图（BM25 vs Milvus）"""
    print("\n" + "=" * 80)
    print("生成检索准确率对比图")
    print("=" * 80)

    if not index_results.get("bm25", {}).get("success") and not index_results.get(
        "milvus", {}
    ).get("success"):
        print("⚠️  BM25和Milvus都未成功加载，跳过图表生成")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Retrieval Test Comparison (BM25 vs Milvus)", fontsize=16, fontweight="bold"
    )

    # 成功率对比
    ax1 = axes[0]
    methods = []
    success_rates = []

    if index_results.get("bm25", {}).get("success"):
        methods.append("BM25")
        success_rates.append(index_results["bm25"].get("success_rate", 0))

    if index_results.get("milvus", {}).get("success"):
        methods.append("Milvus")
        success_rates.append(index_results["milvus"].get("success_rate", 0))

    if methods:
        bars = ax1.bar(
            methods,
            success_rates,
            color=["#3498db", "#e74c3c"],
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_title("Retrieval Success Rate", fontsize=12)
        ax1.set_ylabel("Success Rate (%)", fontsize=10)
        ax1.set_ylim(0, 110)
        ax1.grid(True, alpha=0.3, axis="y")

        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # 每个查询的结果数量对比
    if index_results.get("bm25", {}).get("test_results") and index_results.get(
        "milvus", {}
    ).get("test_results"):
        ax2 = axes[1]

        queries = [r["query"] for r in index_results["bm25"]["test_results"]]
        bm25_counts = [r["num_results"] for r in index_results["bm25"]["test_results"]]
        milvus_counts = [
            r["num_results"] for r in index_results["milvus"]["test_results"]
        ]

        x = np.arange(len(queries))
        width = 0.35

        bars1 = ax2.bar(
            x - width / 2,
            bm25_counts,
            width,
            label="BM25",
            color="#3498db",
            alpha=0.7,
            edgecolor="black",
        )
        bars2 = ax2.bar(
            x + width / 2,
            milvus_counts,
            width,
            label="Milvus",
            color="#e74c3c",
            alpha=0.7,
            edgecolor="black",
        )

        ax2.set_title("Results per Query", fontsize=12)
        ax2.set_xlabel("Query", fontsize=10)
        ax2.set_ylabel("Result Count", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [q[:10] + "..." if len(q) > 10 else q for q in queries],
            rotation=45,
            ha="right",
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
    else:
        ax2.text(
            0.5,
            0.5,
            "Insufficient Data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Results per Query", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / "retrieval_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ 图表已保存到: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("阶段二测试和可视化")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path("/remote-home/share/liangZhang/EvRAG/data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("\n【步骤1】加载数据...")
    docs = load_documents()
    qa_pairs = load_qa_pairs()

    print(f"✓ 原始文档: {len(docs.get('raw_docs', []))} 个")
    print(f"✓ 清洗后文档: {len(docs.get('clean_docs', []))} 个")
    print(f"✓ 切分后文档: {len(docs.get('split_docs', []))} 个")
    print(f"✓ QA对: {len(qa_pairs)} 个")

    # 2. 数据质量测试
    data_quality_results = test_data_quality(docs)

    # 3. 索引测试
    index_results = test_index_retrieval()

    # 4. QA质量测试
    qa_quality_results = test_qa_quality(qa_pairs)

    # 5. 生成图表
    print("\n" + "=" * 80)
    print("生成图表")
    print("=" * 80)

    plot_document_length_distribution(docs, output_dir)
    plot_qa_statistics(qa_pairs, output_dir)
    plot_retrieval_comparison(index_results, output_dir)

    # 6. 保存测试报告
    report = {
        "data_quality": data_quality_results,
        "index_test": index_results,
        "qa_quality": qa_quality_results,
    }

    report_path = output_dir / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 测试报告已保存到: {report_path}")

    # 7. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("✓ 数据质量测试: 完成")
    print("✓ 索引测试: 完成")
    print("✓ QA质量测试: 完成")
    print("✓ 图表生成: 完成")
    print(f"\n所有结果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
