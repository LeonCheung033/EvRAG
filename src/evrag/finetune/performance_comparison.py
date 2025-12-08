"""性能对比模块

对比微调前后模型性能
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .model_evaluator import ModelEvaluator


class PerformanceComparison:
    """性能对比器"""

    def __init__(self):
        """初始化性能对比器"""
        pass

    def compare_llm_performance(
        self,
        baseline_results: Dict[str, Any],
        finetuned_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        对比LLM性能

        Args:
            baseline_results: 基线模型评估结果
            finetuned_results: 微调后模型评估结果

        Returns:
            对比结果字典
        """
        comparison = {
            "baseline": {
                "total_samples": baseline_results.get("total_samples", 0),
                "metrics": baseline_results.get("metrics", {}),
            },
            "finetuned": {
                "total_samples": finetuned_results.get("total_samples", 0),
                "metrics": finetuned_results.get("metrics", {}),
            },
            "improvement": {},
        }

        # 计算改进幅度
        baseline_metrics = baseline_results.get("metrics", {})
        finetuned_metrics = finetuned_results.get("metrics", {})

        for metric in finetuned_metrics:
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                finetuned_value = finetuned_metrics[metric]
                if baseline_value != 0:
                    improvement = (
                        (finetuned_value - baseline_value) / baseline_value
                    ) * 100
                    comparison["improvement"][metric] = {
                        "absolute": finetuned_value - baseline_value,
                        "relative": improvement,
                    }
                else:
                    comparison["improvement"][metric] = {
                        "absolute": finetuned_value - baseline_value,
                        "relative": float("inf") if finetuned_value > 0 else 0.0,
                    }

        return comparison

    def compare_reranker_performance(
        self,
        baseline_results: Dict[str, Any],
        finetuned_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        对比Reranker性能

        Args:
            baseline_results: 基线模型评估结果
            finetuned_results: 微调后模型评估结果

        Returns:
            对比结果字典
        """
        return self.compare_llm_performance(baseline_results, finetuned_results)

    def generate_comparison_report(
        self,
        comparison: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """
        生成对比报告

        Args:
            comparison: 对比结果字典
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 模型性能对比报告\n\n")
            f.write("## 基线模型\n\n")
            f.write(f"- 样本数: {comparison['baseline']['total_samples']}\n")
            f.write("- 指标:\n")
            for metric, value in comparison["baseline"]["metrics"].items():
                f.write(f"  - {metric}: {value:.4f}\n")
            f.write("\n")

            f.write("## 微调后模型\n\n")
            f.write(f"- 样本数: {comparison['finetuned']['total_samples']}\n")
            f.write("- 指标:\n")
            for metric, value in comparison["finetuned"]["metrics"].items():
                f.write(f"  - {metric}: {value:.4f}\n")
            f.write("\n")

            f.write("## 性能改进\n\n")
            for metric, improvement in comparison.get("improvement", {}).items():
                f.write(f"### {metric}\n")
                f.write(f"- 绝对改进: {improvement['absolute']:.4f}\n")
                if improvement["relative"] != float("inf"):
                    f.write(f"- 相对改进: {improvement['relative']:.2f}%\n")
                else:
                    f.write(f"- 相对改进: 从0提升到{improvement['absolute']:.4f}\n")
                f.write("\n")

        print(f"✓ 对比报告已保存: {output_path}")
        return output_path

    def compare_models(
        self,
        baseline_model_path: Path,
        finetuned_model_path: Path,
        test_data_path: Path,
        model_type: str = "llm",
        baseline_base_model_path: Optional[Path] = None,
        finetuned_base_model_path: Optional[Path] = None,
        finetuned_is_lora: bool = False,
        output_dir: Optional[Path] = None,
        # vLLM相关参数
        use_vllm: bool = False,
        baseline_vllm_url: Optional[str] = None,
        baseline_vllm_model: Optional[str] = None,
        finetuned_vllm_url: Optional[str] = None,
        finetuned_vllm_model: Optional[str] = None,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        对比两个模型的性能

        Args:
            baseline_model_path: 基线模型路径
            finetuned_model_path: 微调后模型路径
            test_data_path: 测试数据路径
            model_type: 模型类型（"llm" 或 "reranker"）
            baseline_base_model_path: 基线模型的基础模型路径（仅LLM需要）
            finetuned_base_model_path: 微调后模型的基础模型路径（仅LoRA需要）
            finetuned_is_lora: 微调后模型是否为LoRA
            output_dir: 输出目录

        Returns:
            对比结果字典
        """
        print("=" * 70)
        print("开始模型性能对比评估")
        print("=" * 70)
        print()

        # 仅支持vLLM模式（LLM类型）
        if not use_vllm or model_type != "llm":
            raise ValueError("当前仅支持使用vLLM进行LLM模型评估")

        if not baseline_vllm_model or not finetuned_vllm_model:
            raise ValueError(
                "使用vLLM时必须显式指定 --baseline-vllm-model 和 --finetuned-vllm-model"
            )

        # 准备评估参数
        baseline_evaluator = ModelEvaluator(
            model_path=baseline_model_path,
            model_type=model_type,
            base_model_path=baseline_base_model_path,
            is_lora=False,
        )
        baseline_eval_kwargs = {
            "use_vllm": True,
            "vllm_base_url": baseline_vllm_url,
            "vllm_model": baseline_vllm_model,
            "batch_size": batch_size,
        }

        finetuned_evaluator = ModelEvaluator(
            model_path=finetuned_model_path,
            model_type=model_type,
            base_model_path=finetuned_base_model_path,
            is_lora=finetuned_is_lora,
        )
        finetuned_eval_kwargs = {
            "use_vllm": True,
            "vllm_base_url": finetuned_vllm_url,
            "vllm_model": finetuned_vllm_model,
            "batch_size": batch_size,
        }

        # 并发评估两个模型
        print("🚀 使用并发评估（同时评估基线模型和微调模型）...")
        print()

        def evaluate_baseline():
            print("评估基线模型...")
            result = baseline_evaluator.evaluate(test_data_path, **baseline_eval_kwargs)
            print("✓ 基线模型评估完成")
            return "baseline", result

        def evaluate_finetuned():
            print("评估微调后模型...")
            result = finetuned_evaluator.evaluate(
                test_data_path, **finetuned_eval_kwargs
            )
            print("✓ 微调模型评估完成")
            return "finetuned", result

        # 使用线程池并发执行
        baseline_results = None
        finetuned_results = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_baseline = executor.submit(evaluate_baseline)
            future_finetuned = executor.submit(evaluate_finetuned)

            # 等待两个任务完成
            for future in as_completed([future_baseline, future_finetuned]):
                try:
                    model_type_name, result = future.result()
                    if model_type_name == "baseline":
                        baseline_results = result
                    else:
                        finetuned_results = result
                except Exception as e:
                    print(f"✗ 评估失败: {e}")
                    raise

        print()

        # 对比结果
        if model_type == "llm":
            comparison = self.compare_llm_performance(
                baseline_results, finetuned_results
            )
        else:
            comparison = self.compare_reranker_performance(
                baseline_results, finetuned_results
            )

        # 保存对比报告
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / f"{model_type}_comparison_report.md"
            self.generate_comparison_report(comparison, report_path)

            # 保存详细结果JSON
            json_path = output_dir / f"{model_type}_comparison_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "baseline_results": baseline_results,
                        "finetuned_results": finetuned_results,
                        "comparison": comparison,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"✓ 详细结果已保存: {json_path}")

        # 打印对比摘要
        print("=" * 70)
        print("性能对比摘要")
        print("=" * 70)
        print()
        print("基线模型指标:")
        for metric, value in comparison["baseline"]["metrics"].items():
            print(f"  {metric}: {value:.4f}")
        print()
        print("微调后模型指标:")
        for metric, value in comparison["finetuned"]["metrics"].items():
            print(f"  {metric}: {value:.4f}")
        print()
        print("性能改进:")
        for metric, improvement in comparison.get("improvement", {}).items():
            abs_imp = improvement["absolute"]
            rel_imp = improvement["relative"]
            if rel_imp != float("inf"):
                print(f"  {metric}: {abs_imp:+.4f} ({rel_imp:+.2f}%)")
            else:
                print(f"  {metric}: {abs_imp:+.4f} (从0提升)")
        print()

        return comparison
