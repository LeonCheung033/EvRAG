"""
SFT数据分析与验证模块

提供以下功能：
1. 数据统计：SFT数据和Reranker数据的统计信息
2. 可视化：生成数据分布图表
3. 数据验证：验证数据格式、完整性和平衡性
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import statistics

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class SFTDataAnalyzer:
    """SFT数据分析器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化分析器
        
        Args:
            output_dir: 数据输出目录（包含qa_pairs、summary_data、rerank_data等）
        """
        self.output_dir = Path(output_dir)
        self.qa_pairs_dir = self.output_dir / "qa_pairs"
        self.summary_dir = self.output_dir / "summary_data"
        self.rerank_dir = self.output_dir / "rerank_data"
    
    def analyze_summary_data(self) -> Dict[str, Any]:
        """
        分析summary_data（SFT训练数据）
        
        Returns:
            统计信息字典
        """
        stats = {
            "train": {},
            "test": {},
        }
        
        # 分析训练集
        train_path = self.summary_dir / "train.json"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            stats["train"] = self._analyze_summary_items(train_data, "train")
        
        # 分析测试集
        test_path = self.summary_dir / "test.json"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            stats["test"] = self._analyze_summary_items(test_data, "test")
        
        return stats
    
    def _analyze_summary_items(self, items: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
        """分析summary数据项"""
        if not items:
            return {"count": 0}
        
        context_lengths = []
        output_lengths = []
        has_citations = 0
        no_answer_count = 0
        
        for item in items:
            # 统计context长度
            context = item.get("context", "")
            context_lengths.append(len(context))
            
            # 统计output长度
            output = item.get("output", "")
            output_lengths.append(len(output))
            
            # 统计引用标记
            if "【" in output and "】" in output:
                has_citations += 1
            
            # 统计无答案数量
            if output.strip() == "无答案":
                no_answer_count += 1
        
        return {
            "count": len(items),
            "avg_context_length": statistics.mean(context_lengths) if context_lengths else 0,
            "min_context_length": min(context_lengths) if context_lengths else 0,
            "max_context_length": max(context_lengths) if context_lengths else 0,
            "avg_output_length": statistics.mean(output_lengths) if output_lengths else 0,
            "min_output_length": min(output_lengths) if output_lengths else 0,
            "max_output_length": max(output_lengths) if output_lengths else 0,
            "has_citations": has_citations,
            "citation_rate": has_citations / len(items) if items else 0,
            "no_answer_count": no_answer_count,
            "no_answer_rate": no_answer_count / len(items) if items else 0,
        }
    
    def analyze_rerank_data(self) -> Dict[str, Any]:
        """
        分析rerank_data（Reranker训练数据）
        
        Returns:
            统计信息字典
        """
        stats = {
            "train": {},
            "dev": {},
            "test": {},
        }
        
        # 分析训练集
        train_path = self.rerank_dir / "train.json"
        if train_path.exists():
            stats["train"] = self._analyze_rerank_file(train_path, "train")
        
        # 分析开发集
        dev_path = self.rerank_dir / "dev.json"
        if dev_path.exists():
            stats["dev"] = self._analyze_rerank_file(dev_path, "dev")
        
        # 分析测试集
        test_path = self.rerank_dir / "test.json"
        if test_path.exists():
            stats["test"] = self._analyze_rerank_file(test_path, "test")
        
        return stats
    
    def _analyze_rerank_file(self, file_path: Path, split_name: str) -> Dict[str, Any]:
        """分析rerank数据文件（JSONL格式）"""
        items = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        if not items:
            return {"count": 0}
        
        # 统计标签分布
        labels = [item.get("label", -1) for item in items]
        label_counts = Counter(labels)
        
        # 统计内容长度
        content_lengths = [len(item.get("content", "")) for item in items]
        
        total = len(items)
        label_distribution = {
            "0": label_counts.get(0, 0),
            "1": label_counts.get(1, 0),
            "2": label_counts.get(2, 0),
        }
        label_rates = {
            "0": label_distribution["0"] / total if total > 0 else 0,
            "1": label_distribution["1"] / total if total > 0 else 0,
            "2": label_distribution["2"] / total if total > 0 else 0,
        }
        
        return {
            "count": total,
            "label_distribution": label_distribution,
            "label_rates": label_rates,
            "avg_content_length": statistics.mean(content_lengths) if content_lengths else 0,
            "min_content_length": min(content_lengths) if content_lengths else 0,
            "max_content_length": max(content_lengths) if content_lengths else 0,
        }
    
    def print_summary_statistics(self, stats: Dict[str, Any]):
        """打印summary数据统计信息"""
        console.print("\n[bold cyan]SFT数据统计[/bold cyan]")
        
        table = Table(title="Summary Data Statistics", box=box.ROUNDED)
        table.add_column("指标", style="cyan")
        table.add_column("训练集", style="green")
        table.add_column("测试集", style="yellow")
        
        train_stats = stats.get("train", {})
        test_stats = stats.get("test", {})
        
        table.add_row("样本数量", 
                     f"{train_stats.get('count', 0):,}", 
                     f"{test_stats.get('count', 0):,}")
        table.add_row("平均Context长度", 
                     f"{train_stats.get('avg_context_length', 0):.1f}", 
                     f"{test_stats.get('avg_context_length', 0):.1f}")
        table.add_row("平均Output长度", 
                     f"{train_stats.get('avg_output_length', 0):.1f}", 
                     f"{test_stats.get('avg_output_length', 0):.1f}")
        table.add_row("引用标记数量", 
                     f"{train_stats.get('has_citations', 0):,}", 
                     f"{test_stats.get('has_citations', 0):,}")
        table.add_row("引用率", 
                     f"{train_stats.get('citation_rate', 0)*100:.1f}%", 
                     f"{test_stats.get('citation_rate', 0)*100:.1f}%")
        table.add_row("无答案数量", 
                     f"{train_stats.get('no_answer_count', 0):,}", 
                     f"{test_stats.get('no_answer_count', 0):,}")
        table.add_row("无答案率", 
                     f"{train_stats.get('no_answer_rate', 0)*100:.1f}%", 
                     f"{test_stats.get('no_answer_rate', 0)*100:.1f}%")
        
        console.print(table)
    
    def print_rerank_statistics(self, stats: Dict[str, Any]):
        """打印rerank数据统计信息"""
        console.print("\n[bold cyan]Reranker数据统计[/bold cyan]")
        
        table = Table(title="Rerank Data Statistics", box=box.ROUNDED)
        table.add_column("指标", style="cyan")
        table.add_column("训练集", style="green")
        table.add_column("开发集", style="blue")
        table.add_column("测试集", style="yellow")
        
        train_stats = stats.get("train", {})
        dev_stats = stats.get("dev", {})
        test_stats = stats.get("test", {})
        
        table.add_row("样本数量", 
                     f"{train_stats.get('count', 0):,}", 
                     f"{dev_stats.get('count', 0):,}",
                     f"{test_stats.get('count', 0):,}")
        
        # 标签分布
        for label in ["0", "1", "2"]:
            label_name = {"0": "负样本", "1": "中等样本", "2": "正样本"}[label]
            train_count = train_stats.get("label_distribution", {}).get(label, 0)
            dev_count = dev_stats.get("label_distribution", {}).get(label, 0)
            test_count = test_stats.get("label_distribution", {}).get(label, 0)
            table.add_row(f"{label_name} (label={label})", 
                         f"{train_count:,}", 
                         f"{dev_count:,}",
                         f"{test_count:,}")
        
        # 标签比例
        for label in ["0", "1", "2"]:
            label_name = {"0": "负样本率", "1": "中等样本率", "2": "正样本率"}[label]
            train_rate = train_stats.get("label_rates", {}).get(label, 0) * 100
            dev_rate = dev_stats.get("label_rates", {}).get(label, 0) * 100
            test_rate = test_stats.get("label_rates", {}).get(label, 0) * 100
            table.add_row(f"{label_name}", 
                         f"{train_rate:.1f}%", 
                         f"{dev_rate:.1f}%",
                         f"{test_rate:.1f}%")
        
        console.print(table)
    
    def generate_visualizations(self, stats: Dict[str, Any], output_dir: Path):
        """
        生成可视化图表
        
        Args:
            stats: 统计数据字典
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        plots_dir = output_dir / "logs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. SFT数据样本分布图
        self._plot_summary_distribution(stats.get("summary_data", {}), plots_dir)
        
        # 2. Reranker数据标签分布图
        self._plot_rerank_label_distribution(stats.get("rerank_data", {}), plots_dir)
        
        console.print(f"\n[bold green]✓[/bold green] 图表已保存到: {plots_dir}")
    
    def _plot_summary_distribution(self, summary_stats: Dict[str, Any], output_dir: Path):
        """绘制SFT数据样本分布图"""
        if not summary_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SFT Data Distribution', fontsize=16, fontweight='bold')
        
        train_stats = summary_stats.get("train", {})
        test_stats = summary_stats.get("test", {})
        
        # 1. 样本数量对比
        ax1 = axes[0, 0]
        splits = ["Train", "Test"]
        counts = [train_stats.get("count", 0), test_stats.get("count", 0)]
        colors = ['#4CAF50', '#FF9800']
        bars = ax1.bar(splits, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Sample Count Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # 2. Context长度分布
        ax2 = axes[0, 1]
        train_avg = train_stats.get("avg_context_length", 0)
        test_avg = test_stats.get("avg_context_length", 0)
        ax2.bar(["Train", "Test"], [train_avg, test_avg], color=['#2196F3', '#FF5722'], alpha=0.7, edgecolor='black')
        ax2.set_title('Average Context Length', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Length (characters)', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.text(0, train_avg, f'{train_avg:.1f}', ha='center', va='bottom', fontsize=10)
        ax2.text(1, test_avg, f'{test_avg:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Output长度分布
        ax3 = axes[1, 0]
        train_avg = train_stats.get("avg_output_length", 0)
        test_avg = test_stats.get("avg_output_length", 0)
        ax3.bar(["Train", "Test"], [train_avg, test_avg], color=['#9C27B0', '#F44336'], alpha=0.7, edgecolor='black')
        ax3.set_title('Average Output Length', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Length (characters)', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.text(0, train_avg, f'{train_avg:.1f}', ha='center', va='bottom', fontsize=10)
        ax3.text(1, test_avg, f'{test_avg:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 引用率和无答案率
        ax4 = axes[1, 1]
        train_citation = train_stats.get("citation_rate", 0) * 100
        test_citation = test_stats.get("citation_rate", 0) * 100
        train_no_answer = train_stats.get("no_answer_rate", 0) * 100
        test_no_answer = test_stats.get("no_answer_rate", 0) * 100
        
        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, [train_citation, test_citation], width, label='Citation Rate', 
               color='#4CAF50', alpha=0.7, edgecolor='black')
        ax4.bar(x + width/2, [train_no_answer, test_no_answer], width, label='No Answer Rate', 
               color='#F44336', alpha=0.7, edgecolor='black')
        ax4.set_title('Citation Rate & No Answer Rate', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Rate (%)', fontsize=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(["Train", "Test"])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / 'sft_data_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"  - SFT数据分布图: {output_path}")
    
    def _plot_rerank_label_distribution(self, rerank_stats: Dict[str, Any], output_dir: Path):
        """绘制Reranker数据标签分布图"""
        if not rerank_stats:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Reranker Data Label Distribution', fontsize=16, fontweight='bold')
        
        splits = ["train", "dev", "test"]
        split_names = ["Train", "Dev", "Test"]
        colors = {'0': '#F44336', '1': '#FF9800', '2': '#4CAF50'}
        labels_map = {'0': 'Negative', '1': 'Medium', '2': 'Positive'}
        
        for idx, (split, split_name) in enumerate(zip(splits, split_names)):
            split_stats = rerank_stats.get(split, {})
            if not split_stats or split_stats.get("count", 0) == 0:
                axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                              transform=axes[idx].transAxes, fontsize=14)
                axes[idx].set_title(f'{split_name} Set', fontsize=12, fontweight='bold')
                continue
            
            label_dist = split_stats.get("label_distribution", {})
            label_rates = split_stats.get("label_rates", {})
            
            # 饼图
            labels = []
            sizes = []
            colors_list = []
            for label_key in ['0', '1', '2']:
                count = label_dist.get(label_key, 0)
                if count > 0:
                    labels.append(f"{labels_map[label_key]}\n({count})")
                    sizes.append(count)
                    colors_list.append(colors[label_key])
            
            if sizes:
                axes[idx].pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%',
                             startangle=90, textprops={'fontsize': 10})
                axes[idx].set_title(f'{split_name} Set\n(Total: {split_stats.get("count", 0):,})', 
                                   fontsize=12, fontweight='bold')
            else:
                axes[idx].text(0.5, 0.5, 'No Valid Data', ha='center', va='center', 
                              transform=axes[idx].transAxes, fontsize=12)
                axes[idx].set_title(f'{split_name} Set', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = output_dir / 'rerank_label_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"  - Reranker标签分布图: {output_path}")


class SFTDataValidator:
    """SFT数据验证器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化验证器
        
        Args:
            output_dir: 数据输出目录
        """
        self.output_dir = Path(output_dir)
        self.qa_pairs_dir = self.output_dir / "qa_pairs"
        self.summary_dir = self.output_dir / "summary_data"
        self.rerank_dir = self.output_dir / "rerank_data"
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> Dict[str, Any]:
        """
        验证所有数据
        
        Returns:
            验证结果字典
        """
        results = {
            "train_data": self.validate_train_data(),
            "summary_data": self.validate_summary_data(),
            "rerank_data": self.validate_rerank_data(),
            "data_balance": self.validate_data_balance(),
            "data_completeness": self.validate_data_completeness(),
        }
        
        return results
    
    def validate_train_data(self) -> Dict[str, Any]:
        """验证train_data.json格式"""
        train_data_path = self.qa_pairs_dir / "train_data.json"
        
        if not train_data_path.exists():
            return {"valid": False, "error": "文件不存在"}
        
        required_fields = ["query", "context", "response", "merged_docs"]
        errors = []
        valid_count = 0
        total_count = 0
        
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                total_count += 1
                try:
                    item = json.loads(line)
                    
                    # 检查必需字段
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        errors.append(f"行 {line_num}: 缺少字段 {missing_fields}")
                        continue
                    
                    # 检查字段类型
                    if not isinstance(item["query"], str):
                        errors.append(f"行 {line_num}: query应为字符串")
                        continue
                    if not isinstance(item["context"], list):
                        errors.append(f"行 {line_num}: context应为列表")
                        continue
                    if not isinstance(item["response"], str):
                        errors.append(f"行 {line_num}: response应为字符串")
                        continue
                    if not isinstance(item["merged_docs"], list):
                        errors.append(f"行 {line_num}: merged_docs应为列表")
                        continue
                    
                    valid_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"行 {line_num}: JSON解析错误 - {e}")
        
        return {
            "valid": len(errors) == 0,
            "total_count": total_count,
            "valid_count": valid_count,
            "errors": errors[:10],  # 只显示前10个错误
            "error_count": len(errors),
        }
    
    def validate_summary_data(self) -> Dict[str, Any]:
        """验证summary_data格式"""
        results = {}
        
        for split in ["train", "test"]:
            file_path = self.summary_dir / f"{split}.json"
            
            if not file_path.exists():
                results[split] = {"valid": False, "error": "文件不存在"}
                continue
            
            required_fields = ["query", "context", "instruction", "input", "output"]
            errors = []
            valid_count = 0
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    results[split] = {"valid": False, "error": "应为JSON数组格式"}
                    continue
                
                for idx, item in enumerate(data):
                    # 检查必需字段
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        errors.append(f"索引 {idx}: 缺少字段 {missing_fields}")
                        continue
                    
                    # 检查字段类型
                    if not isinstance(item["query"], str):
                        errors.append(f"索引 {idx}: query应为字符串")
                        continue
                    if not isinstance(item["context"], str):
                        errors.append(f"索引 {idx}: context应为字符串")
                        continue
                    if not isinstance(item["instruction"], str):
                        errors.append(f"索引 {idx}: instruction应为字符串")
                        continue
                    if not isinstance(item["input"], str):
                        errors.append(f"索引 {idx}: input应为字符串")
                        continue
                    if not isinstance(item["output"], str):
                        errors.append(f"索引 {idx}: output应为字符串")
                        continue
                    
                    valid_count += 1
                
                results[split] = {
                    "valid": len(errors) == 0,
                    "total_count": len(data),
                    "valid_count": valid_count,
                    "errors": errors[:10],
                    "error_count": len(errors),
                }
            except json.JSONDecodeError as e:
                results[split] = {"valid": False, "error": f"JSON解析错误: {e}"}
        
        return results
    
    def validate_rerank_data(self) -> Dict[str, Any]:
        """验证rerank_data格式"""
        results = {}
        
        for split in ["train", "dev", "test"]:
            file_path = self.rerank_dir / f"{split}.json"
            
            if not file_path.exists():
                results[split] = {"valid": False, "error": "文件不存在"}
                continue
            
            required_fields = ["query", "content", "label"]
            errors = []
            valid_count = 0
            total_count = 0
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    total_count += 1
                    try:
                        item = json.loads(line)
                        
                        # 检查必需字段
                        missing_fields = [field for field in required_fields if field not in item]
                        if missing_fields:
                            errors.append(f"行 {line_num}: 缺少字段 {missing_fields}")
                            continue
                        
                        # 检查字段类型
                        if not isinstance(item["query"], str):
                            errors.append(f"行 {line_num}: query应为字符串")
                            continue
                        if not isinstance(item["content"], str):
                            errors.append(f"行 {line_num}: content应为字符串")
                            continue
                        if not isinstance(item["label"], int) or item["label"] not in [0, 1, 2]:
                            errors.append(f"行 {line_num}: label应为0、1或2")
                            continue
                        
                        valid_count += 1
                    except json.JSONDecodeError as e:
                        errors.append(f"行 {line_num}: JSON解析错误 - {e}")
            
            results[split] = {
                "valid": len(errors) == 0,
                "total_count": total_count,
                "valid_count": valid_count,
                "errors": errors[:10],
                "error_count": len(errors),
            }
        
        return results
    
    def validate_data_balance(self) -> Dict[str, Any]:
        """验证数据平衡性（正负样本比例）"""
        results = {}
        
        # 检查rerank数据的标签分布
        for split in ["train", "dev", "test"]:
            file_path = self.rerank_dir / f"{split}.json"
            
            if not file_path.exists():
                continue
            
            labels = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            labels.append(item.get("label", -1))
                        except:
                            pass
            
            if not labels:
                continue
            
            label_counts = Counter(labels)
            total = len(labels)
            
            # 计算比例
            rates = {
                label: count / total 
                for label, count in label_counts.items()
            }
            
            # 检查是否平衡（正样本应该最多，负样本和中等样本应该相对平衡）
            warnings = []
            if rates.get(2, 0) < 0.2:  # 正样本少于20%
                warnings.append("正样本比例过低（<20%）")
            if rates.get(0, 0) > 0.6:  # 负样本超过60%
                warnings.append("负样本比例过高（>60%）")
            if rates.get(1, 0) < 0.1:  # 中等样本少于10%
                warnings.append("中等样本比例过低（<10%）")
            
            results[split] = {
                "total": total,
                "label_counts": dict(label_counts),
                "label_rates": rates,
                "warnings": warnings,
                "balanced": len(warnings) == 0,
            }
        
        return results
    
    def validate_data_completeness(self) -> Dict[str, Any]:
        """验证数据完整性（无缺失字段）"""
        results = {
            "train_data": {},
            "summary_data": {},
            "rerank_data": {},
        }
        
        # 检查train_data.json
        train_data_path = self.qa_pairs_dir / "train_data.json"
        if train_data_path.exists():
            missing_fields_count = 0
            total_count = 0
            
            with open(train_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    total_count += 1
                    try:
                        item = json.loads(line)
                        if not all(key in item for key in ["query", "context", "response", "merged_docs"]):
                            missing_fields_count += 1
                    except:
                        pass
            
            results["train_data"] = {
                "total": total_count,
                "missing_fields": missing_fields_count,
                "complete": missing_fields_count == 0,
            }
        
        # 检查summary_data
        for split in ["train", "test"]:
            file_path = self.summary_dir / f"{split}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    missing_fields_count = 0
                    for item in data:
                        if not all(key in item for key in ["query", "context", "instruction", "input", "output"]):
                            missing_fields_count += 1
                    
                    results["summary_data"][split] = {
                        "total": len(data),
                        "missing_fields": missing_fields_count,
                        "complete": missing_fields_count == 0,
                    }
                except:
                    pass
        
        # 检查rerank_data
        for split in ["train", "dev", "test"]:
            file_path = self.rerank_dir / f"{split}.json"
            if file_path.exists():
                missing_fields_count = 0
                total_count = 0
                
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        total_count += 1
                        try:
                            item = json.loads(line)
                            if not all(key in item for key in ["query", "content", "label"]):
                                missing_fields_count += 1
                        except:
                            pass
                
                results["rerank_data"][split] = {
                    "total": total_count,
                    "missing_fields": missing_fields_count,
                    "complete": missing_fields_count == 0,
                }
        
        return results
    
    def print_validation_results(self, results: Dict[str, Any]):
        """打印验证结果"""
        console.print("\n[bold cyan]数据验证结果[/bold cyan]")
        
        # 格式验证
        console.print("\n[bold yellow]1. 格式验证[/bold yellow]")
        
        # train_data
        train_data_result = results.get("train_data", {})
        if train_data_result.get("valid"):
            console.print(f"[green]✓[/green] train_data.json: 格式正确 ({train_data_result.get('valid_count', 0)}/{train_data_result.get('total_count', 0)} 条有效)")
        else:
            console.print(f"[red]✗[/red] train_data.json: 格式错误")
            if train_data_result.get("errors"):
                for error in train_data_result["errors"][:5]:
                    console.print(f"  - {error}")
        
        # summary_data
        summary_result = results.get("summary_data", {})
        for split in ["train", "test"]:
            split_result = summary_result.get(split, {})
            if split_result.get("valid"):
                console.print(f"[green]✓[/green] summary_data/{split}.json: 格式正确 ({split_result.get('valid_count', 0)}/{split_result.get('total_count', 0)} 条有效)")
            else:
                console.print(f"[red]✗[/red] summary_data/{split}.json: 格式错误")
                if split_result.get("errors"):
                    for error in split_result["errors"][:5]:
                        console.print(f"  - {error}")
        
        # rerank_data
        rerank_result = results.get("rerank_data", {})
        for split in ["train", "dev", "test"]:
            split_result = rerank_result.get(split, {})
            if split_result.get("valid"):
                console.print(f"[green]✓[/green] rerank_data/{split}.json: 格式正确 ({split_result.get('valid_count', 0)}/{split_result.get('total_count', 0)} 条有效)")
            else:
                console.print(f"[red]✗[/red] rerank_data/{split}.json: 格式错误")
                if split_result.get("errors"):
                    for error in split_result["errors"][:5]:
                        console.print(f"  - {error}")
        
        # 数据平衡性
        console.print("\n[bold yellow]2. 数据平衡性[/bold yellow]")
        balance_result = results.get("data_balance", {})
        for split in ["train", "dev", "test"]:
            split_result = balance_result.get(split, {})
            if split_result:
                if split_result.get("balanced"):
                    console.print(f"[green]✓[/green] rerank_data/{split}.json: 数据平衡")
                else:
                    console.print(f"[yellow]⚠[/yellow] rerank_data/{split}.json: 数据不平衡")
                    for warning in split_result.get("warnings", []):
                        console.print(f"  - {warning}")
                label_rates = split_result.get("label_rates", {})
                console.print(f"  标签分布: 正样本={label_rates.get(2, 0)*100:.1f}%, 中等样本={label_rates.get(1, 0)*100:.1f}%, 负样本={label_rates.get(0, 0)*100:.1f}%")
        
        # 数据完整性
        console.print("\n[bold yellow]3. 数据完整性[/bold yellow]")
        completeness_result = results.get("data_completeness", {})
        
        train_data_complete = completeness_result.get("train_data", {})
        if train_data_complete.get("complete"):
            console.print(f"[green]✓[/green] train_data.json: 数据完整")
        else:
            console.print(f"[red]✗[/red] train_data.json: {train_data_complete.get('missing_fields', 0)} 条数据缺少字段")
        
        for split in ["train", "test"]:
            summary_complete = completeness_result.get("summary_data", {}).get(split, {})
            if summary_complete.get("complete"):
                console.print(f"[green]✓[/green] summary_data/{split}.json: 数据完整")
            else:
                console.print(f"[red]✗[/red] summary_data/{split}.json: {summary_complete.get('missing_fields', 0)} 条数据缺少字段")
        
        for split in ["train", "dev", "test"]:
            rerank_complete = completeness_result.get("rerank_data", {}).get(split, {})
            if rerank_complete.get("complete"):
                console.print(f"[green]✓[/green] rerank_data/{split}.json: 数据完整")
            else:
                console.print(f"[red]✗[/red] rerank_data/{split}.json: {rerank_complete.get('missing_fields', 0)} 条数据缺少字段")

