"""训练可视化模块

生成训练过程的各种图表
"""

import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


class TrainingVisualizer:
    """训练可视化器"""

    def __init__(self, output_dir: Path):
        """
        初始化可视化器

        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_loss(
        self,
        log_dir: Path,
        output_path: Optional[Path] = None,
        smooth: bool = True,
    ) -> Path:
        """
        绘制训练loss曲线

        Args:
            log_dir: TensorBoard日志目录
            output_path: 输出文件路径
            smooth: 是否平滑曲线

        Returns:
            输出文件路径
        """
        if output_path is None:
            output_path = self.output_dir / "training_loss.png"

        # 从TensorBoard读取数据（支持多种标签格式）
        # LLaMA-Factory格式: train/loss, eval/loss
        # RAG-Retrieval格式: avg_loss, cur_loss, val_loss
        data = self._read_tensorboard_scalars(
            log_dir, ["train/loss", "eval/loss", "avg_loss", "cur_loss", "val_loss"]
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # 优先使用LLaMA-Factory格式
        if "train/loss" in data:
            train_steps = data["train/loss"]["steps"]
            train_values = data["train/loss"]["values"]
            if smooth:
                train_values = self._smooth_curve(train_values)
            ax.plot(
                train_steps, train_values, label="Training Loss", linewidth=2, alpha=0.8
            )
        # 使用RAG-Retrieval格式
        elif "avg_loss" in data:
            train_steps = data["avg_loss"]["steps"]
            train_values = data["avg_loss"]["values"]
            if smooth:
                train_values = self._smooth_curve(train_values)
            ax.plot(
                train_steps, train_values, label="Average Loss", linewidth=2, alpha=0.8
            )

        # 绘制当前loss（可选）
        if "cur_loss" in data and "avg_loss" in data:
            cur_steps = data["cur_loss"]["steps"]
            cur_values = data["cur_loss"]["values"]
            if smooth:
                cur_values = self._smooth_curve(cur_values)
            ax.plot(
                cur_steps,
                cur_values,
                label="Current Loss",
                linewidth=1,
                alpha=0.5,
                linestyle=":",
            )

        # 验证loss
        if "eval/loss" in data:
            eval_steps = data["eval/loss"]["steps"]
            eval_values = data["eval/loss"]["values"]
            if smooth:
                eval_values = self._smooth_curve(eval_values)
            ax.plot(
                eval_steps,
                eval_values,
                label="Validation Loss",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
            )
        elif "val_loss" in data:
            eval_steps = data["val_loss"]["steps"]
            eval_values = data["val_loss"]["values"]
            if smooth:
                eval_values = self._smooth_curve(eval_values)
            ax.plot(
                eval_steps,
                eval_values,
                label="Validation Loss",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
            )

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
        if ax.get_legend_handles_labels()[0]:  # 只有当有图例时才显示
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ 训练Loss曲线已保存: {output_path}")
        return output_path

    def plot_learning_rate(
        self,
        log_dir: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        绘制学习率曲线

        Args:
            log_dir: TensorBoard日志目录
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        if output_path is None:
            output_path = self.output_dir / "learning_rate.png"

        # 从TensorBoard读取数据（支持多种标签格式）
        # LLaMA-Factory格式: train/learning_rate
        # RAG-Retrieval格式: lr
        data = self._read_tensorboard_scalars(log_dir, ["train/learning_rate", "lr"])

        fig, ax = plt.subplots(figsize=(12, 6))

        if "train/learning_rate" in data:
            steps = data["train/learning_rate"]["steps"]
            values = data["train/learning_rate"]["values"]
            ax.plot(
                steps,
                values,
                label="Learning Rate",
                linewidth=2,
                color="green",
                alpha=0.8,
            )
        elif "lr" in data:
            steps = data["lr"]["steps"]
            values = data["lr"]["values"]
            ax.plot(
                steps,
                values,
                label="Learning Rate",
                linewidth=2,
                color="green",
                alpha=0.8,
            )

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        if ax.get_legend_handles_labels()[0]:  # 只有当有图例时才显示
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        # 只有当学习率值大于0时才使用对数刻度
        if (
            "train/learning_rate" in data
            and min(data["train/learning_rate"]["values"]) > 0
        ):
            ax.set_yscale("log")
        elif "lr" in data and min(data["lr"]["values"]) > 0:
            ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ 学习率曲线已保存: {output_path}")
        return output_path

    def plot_evaluation_metrics(
        self,
        log_dir: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        绘制评估指标曲线

        Args:
            log_dir: TensorBoard日志目录
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        if output_path is None:
            output_path = self.output_dir / "evaluation_metrics.png"

        # 从TensorBoard读取评估指标
        metrics = ["eval/accuracy", "eval/f1", "eval/precision", "eval/recall"]
        data = self._read_tensorboard_scalars(log_dir, metrics)

        fig, ax = plt.subplots(figsize=(12, 6))

        for metric in metrics:
            if metric in data:
                steps = data[metric]["steps"]
                values = data[metric]["values"]
                metric_name = metric.split("/")[-1]
                ax.plot(steps, values, label=metric_name, linewidth=2, alpha=0.8)

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title("Evaluation Metrics", fontsize=14, fontweight="bold")
        if ax.get_legend_handles_labels()[0]:  # 只有当有图例时才显示
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ 评估指标曲线已保存: {output_path}")
        return output_path

    def plot_gpu_utilization(
        self,
        gpu_log_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        绘制GPU利用率图

        Args:
            gpu_log_path: GPU监控日志文件路径
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        if output_path is None:
            output_path = self.output_dir / "gpu_utilization.png"

        if not gpu_log_path.exists():
            print(f"⚠ GPU日志文件不存在: {gpu_log_path}")
            return output_path

        # 读取GPU日志
        import csv

        data = []
        with open(gpu_log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 8:
                    try:
                        data.append(
                            {
                                "timestamp": pd.to_datetime(row[0]),
                                "gpu_index": int(row[1]),
                                "utilization": float(row[3].replace("%", "")),
                                "memory_used": int(row[4].replace("MiB", "").strip()),
                                "memory_total": int(row[5].replace("MiB", "").strip()),
                            }
                        )
                    except Exception:
                        continue

        if not data:
            print("⚠ GPU日志数据为空")
            return output_path

        df = pd.DataFrame(data)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # GPU利用率
        for gpu_id in df["gpu_index"].unique():
            gpu_data = df[df["gpu_index"] == gpu_id]
            axes[0].plot(
                gpu_data["timestamp"],
                gpu_data["utilization"],
                label=f"GPU {gpu_id}",
                linewidth=2,
                alpha=0.7,
            )

        axes[0].set_xlabel("时间", fontsize=12)
        axes[0].set_ylabel("GPU利用率 (%)", fontsize=12)
        axes[0].set_title("GPU利用率变化", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)

        # GPU内存使用
        for gpu_id in df["gpu_index"].unique():
            gpu_data = df[df["gpu_index"] == gpu_id]
            memory_usage = (gpu_data["memory_used"] / gpu_data["memory_total"]) * 100
            axes[1].plot(
                gpu_data["timestamp"],
                memory_usage,
                label=f"GPU {gpu_id}",
                linewidth=2,
                alpha=0.7,
            )

        axes[1].set_xlabel("时间", fontsize=12)
        axes[1].set_ylabel("显存使用率 (%)", fontsize=12)
        axes[1].set_title("显存使用率变化", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ GPU利用率图已保存: {output_path}")
        return output_path

    def _read_tensorboard_scalars(
        self,
        log_dir: Path,
        scalar_tags: List[str],
    ) -> Dict:
        """
        从TensorBoard日志读取标量数据

        Args:
            log_dir: TensorBoard日志目录
            scalar_tags: 要读取的标量标签列表

        Returns:
            标量数据字典
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return {}

        # 查找TensorBoard事件文件
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            return {}

        # 使用最新的事件文件
        event_file = max(event_files, key=lambda x: x.stat().st_mtime)

        try:
            ea = EventAccumulator(str(event_file.parent))
            ea.Reload()

            data = {}
            for tag in scalar_tags:
                if tag in ea.Tags()["scalars"]:
                    scalar_events = ea.Scalars(tag)
                    steps = [e.step for e in scalar_events]
                    values = [e.value for e in scalar_events]
                    data[tag] = {"steps": steps, "values": values}

            return data
        except Exception as e:
            print(f"读取TensorBoard日志错误: {e}")
            return {}

    def _smooth_curve(self, values: List[float], window_size: int = 10) -> List[float]:
        """
        平滑曲线（移动平均）

        Args:
            values: 原始值列表
            window_size: 窗口大小

        Returns:
            平滑后的值列表
        """
        if len(values) < window_size:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start:end]))

        return smoothed
