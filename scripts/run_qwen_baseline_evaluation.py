#!/usr/bin/env python3
"""
Qwen基线系统评估脚本

使用Qwen3-32B + Qwen3-Embedding-8B进行端到端RAG评估。
从原始PDF文件加载并分块，模拟真实chatbot的粗糙处理方式。
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evrag.evaluation import QwenBaselineEvaluator
from src.evrag.config import get_settings


def main(
    test_data_path: str = "data/qa_pairs/test_qa_pair_verify.json",
    output_dir: Optional[str] = None,
    pdf_path: Optional[str] = None,
    topk: int = 10,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    sample_size: Optional[int] = None,
    full_test: bool = False,
    use_ragas: bool = True,
    max_workers: Optional[int] = None,
    llm_model_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
):
    """
    运行Qwen基线系统评估
    
    Args:
        test_data_path: 测试数据文件路径
        output_dir: 输出目录（如果为None，自动生成带时间戳的目录）
        pdf_path: PDF文件路径（如果为None，从配置读取）
        topk: 检索数量
        chunk_size: 文本分块大小
        chunk_overlap: 文本分块重叠大小
        sample_size: 抽样测试样本数（None表示全量测试，仅在full_test=False时生效）
        full_test: 是否执行全量测试
        use_ragas: 是否使用RAGas评估
        max_workers: 最大并发工作线程数
        llm_model_name: LLM模型名称
        embedding_model_name: Embedding模型名称
        temperature: LLM温度参数
        max_tokens: LLM最大生成token数
    """
    print("=" * 60)
    print("Qwen基线系统评估")
    print("=" * 60)
    
    # 获取配置
    settings = get_settings()
    
    # 检查测试数据文件
    test_data_file = Path(test_data_path)
    if not test_data_file.exists():
        print(f"❌ 错误: 测试数据文件不存在: {test_data_file}")
        print(f"   请确保文件路径正确")
        return 1
    
    # 加载测试数据
    print(f"\n📂 加载测试数据: {test_data_file}")
    with open(test_data_file, "r", encoding="utf-8") as f:
        test_data_list = json.load(f)
    
    print(f"   总测试样本数: {len(test_data_list)}")
    
    # 确定是否抽样
    if not full_test and sample_size and sample_size < len(test_data_list):
        import random
        random.seed(42)
        test_data_list = random.sample(test_data_list, sample_size)
        print(f"   抽样测试样本数: {sample_size}")
    else:
        print(f"   全量测试样本数: {len(test_data_list)}")
    
    # 确定输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"rag_test_reports/qwen_baseline_{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 输出目录: {output_path}")
    
    # 初始化评估器
    print("\n🔧 初始化Qwen基线评估器...")
    try:
        evaluator = QwenBaselineEvaluator(
            topk=topk,
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            max_workers=max_workers,
            pdf_path=Path(pdf_path) if pdf_path else None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print("   ✓ Qwen基线评估器初始化成功")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 执行批量评估
    print("\n🚀 执行批量评估...")
    try:
        results = evaluator.evaluate_batch(
            test_data_list,
            sample_size=None,  # 已经抽样了
            show_progress=True,
        )
        
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]
        
        print(f"\n   ✓ 评估完成")
        print(f"     有效结果数: {len(valid_results)}")
        if error_results:
            print(f"     错误结果数: {len(error_results)}")
    except Exception as e:
        print(f"   ❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 计算汇总指标和综合准确率
    print("\n📊 计算汇总指标...")
    try:
        summary = evaluator.calculate_comprehensive_accuracy(
            use_ragas=use_ragas,
            semantic_weight=0.7,
            ragas_weight=0.3,
            ragas_sample_size=None,  # 全量RAGas评估
        )
        
        # 打印关键指标
        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)
        print(f"总样本数: {summary.get('total_samples', 0)}")
        
        # 语义相似度+关键词加权得分
        semantic_score = summary.get("semantic_keyword_score", {})
        if semantic_score:
            print(f"\n语义相似度+关键词加权得分:")
            print(f"  平均分: {semantic_score.get('mean', 0.0):.4f}")
            print(f"  标准差: {semantic_score.get('std', 0.0):.4f}")
        
        # 生成质量指标
        gen_quality = summary.get("generation_quality", {})
        if gen_quality:
            print(f"\n生成质量指标:")
            print(f"  BLEU: {gen_quality.get('bleu', 0.0):.4f}")
            print(f"  ROUGE-1: {gen_quality.get('rouge_1', 0.0):.4f}")
            print(f"  ROUGE-2: {gen_quality.get('rouge_2', 0.0):.4f}")
            print(f"  ROUGE-L: {gen_quality.get('rouge_l', 0.0):.4f}")
        
        # RAGas得分
        if use_ragas and "ragas_scores" in summary:
            ragas_scores = summary["ragas_scores"]
            print(f"\nRAGas得分:")
            print(f"  Context Recall: {ragas_scores.get('context_recall', 0.0):.4f}")
            print(f"  Context Precision: {ragas_scores.get('context_precision', 0.0):.4f}")
            print(f"  平均分: {ragas_scores.get('average', 0.0):.4f}")
        
        # 综合准确率
        comp_accuracy = summary.get("comprehensive_accuracy", {})
        if comp_accuracy:
            print(f"\n综合准确率:")
            print(f"  得分: {comp_accuracy.get('score', 0.0):.4f}")
            print(f"  语义得分: {comp_accuracy.get('semantic_keyword_score', 0.0):.4f}")
            if comp_accuracy.get('ragas_score'):
                print(f"  RAGas得分: {comp_accuracy.get('ragas_score', 0.0):.4f}")
        
        # 响应时间
        resp_time = summary.get("response_time", {})
        if resp_time:
            print(f"\n响应时间:")
            print(f"  平均检索时间: {resp_time.get('mean_retrieval_time', 0.0):.2f}秒")
            print(f"  平均生成时间: {resp_time.get('mean_generation_time', 0.0):.2f}秒")
            print(f"  平均总时间: {resp_time.get('mean_total_time', 0.0):.2f}秒")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"   ❌ 计算汇总指标失败: {e}")
        import traceback
        traceback.print_exc()
        summary = None
    
    # 保存结果
    print("\n💾 保存评估结果...")
    try:
        output_file = output_path / "qwen_baseline_evaluation_results.json"
        evaluator.save_results(output_file, summary=summary)
        print(f"   ✓ 结果已保存到: {output_file}")
    except Exception as e:
        print(f"   ❌ 保存结果失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("✓ 评估完成！")
    print("=" * 60)
    print(f"结果文件: {output_path}")
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Qwen基线系统评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数运行（抽样30条）
  python scripts/run_qwen_baseline_evaluation.py
  
  # 全量测试
  python scripts/run_qwen_baseline_evaluation.py --full-test
  
  # 指定测试数据和输出目录
  python scripts/run_qwen_baseline_evaluation.py \\
      --test-data data/qa_pairs/test_qa_pair_handmade_verify01.json \\
      --output-dir rag_test_reports/my_test \\
      --sample-size 10
  
  # 自定义PDF路径和分块参数
  python scripts/run_qwen_baseline_evaluation.py \\
      --pdf-path data/Tesla_Manual.pdf \\
      --chunk-size 500 \\
      --chunk-overlap 50
        """
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/qa_pairs/test_qa_pair_handmade_verify01.json",
        help="测试数据文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（如果为None，自动生成带时间戳的目录）"
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="PDF文件路径（如果为None，从配置读取）"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="检索数量（默认：10）"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="文本分块大小（默认：500）"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="文本分块重叠大小（默认：50）"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="抽样测试样本数（None表示全量测试，仅在--full-test未指定时生效）"
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="是否执行全量测试"
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="不使用RAGas评估"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="最大并发工作线程数（None表示自动探测）"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM模型名称（默认：Qwen/Qwen3-32B）"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding模型名称（默认：Qwen/Qwen3-Embedding-8B）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM温度参数（默认：0.1）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="LLM最大生成token数（默认：512）"
    )
    
    args = parser.parse_args()
    
    exit_code = main(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        pdf_path=args.pdf_path,
        topk=args.topk,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sample_size=args.sample_size,
        full_test=args.full_test,
        use_ragas=not args.no_ragas,
        max_workers=args.max_workers,
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    sys.exit(exit_code)

