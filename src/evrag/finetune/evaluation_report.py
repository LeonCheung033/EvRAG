"""评估报告生成模块

生成详细的评估报告
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class EvaluationReport:
    """评估报告生成器"""

    def __init__(self, output_dir: Path):
        """
        初始化评估报告生成器

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        model_type: str,
        model_path: Path,
        test_data_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        生成评估报告

        Args:
            evaluation_results: 评估结果字典
            model_type: 模型类型
            model_path: 模型路径
            test_data_path: 测试数据路径
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{model_type}_evaluation_{timestamp}.md"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {model_type.upper()}模型评估报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**模型路径**: {model_path}\n\n")
            f.write(f"**测试数据**: {test_data_path}\n\n")

            f.write("## 评估概览\n\n")
            f.write(f"- **总样本数**: {evaluation_results.get('total_samples', 0)}\n\n")

            f.write("## 评估指标\n\n")
            metrics = evaluation_results.get("metrics", {})
            if metrics:
                for metric, value in metrics.items():
                    f.write(f"- **{metric}**: {value:.4f}\n")
            else:
                f.write("暂无评估指标\n")
            f.write("\n")

            f.write("## 详细结果\n\n")
            predictions = evaluation_results.get("predictions", [])
            if predictions:
                f.write(f"共 {len(predictions)} 个预测结果\n\n")
                # 可以选择性地展示前几个样本
                for i, pred in enumerate(predictions[:5]):
                    f.write(f"### 样本 {i+1}\n\n")
                    if "expected" in pred:
                        f.write(f"**期望输出**: {pred['expected']}\n\n")
                    if "predicted" in pred:
                        f.write(f"**预测输出**: {pred['predicted']}\n\n")
                    f.write("---\n\n")

        print(f"✓ 评估报告已保存: {output_path}")
        return output_path

