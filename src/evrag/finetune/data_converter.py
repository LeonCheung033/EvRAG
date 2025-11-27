"""数据格式转换模块

将SFT数据转换为LLaMA-Factory所需的格式
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import yaml


class DataConverter:
    """数据格式转换器"""

    def __init__(self, output_dir: Path):
        """
        初始化数据转换器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_sft_data_to_llamafactory(
        self,
        input_path: Path,
        output_path: Path,
    ) -> Path:
        """
        将SFT数据转换为LLaMA-Factory格式

        Args:
            input_path: 输入的SFT数据文件路径（summary_data/train.json格式）
            output_path: 输出的LLaMA-Factory格式数据文件路径

        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 读取原始数据
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 转换为LLaMA-Factory格式
        llamafactory_data = []
        for item in data:
            llamafactory_item = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
            }
            llamafactory_data.append(llamafactory_item)

        # 保存转换后的数据
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(llamafactory_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 数据转换完成: {len(llamafactory_data)} 条数据")
        print(f"  输入: {input_path}")
        print(f"  输出: {output_path}")

        return output_path

    def create_dataset_config(
        self,
        dataset_name: str,
        train_file: Path,
        test_file: Path = None,
        output_path: Path = None,
    ) -> Path:
        """
        创建LLaMA-Factory数据集配置文件

        Args:
            dataset_name: 数据集名称
            train_file: 训练数据文件路径
            test_file: 测试数据文件路径（可选）
            output_path: 输出配置文件路径

        Returns:
            配置文件路径
        """
        if output_path is None:
            output_path = self.output_dir / f"{dataset_name}.yaml"

        # 构建数据集配置
        # 确保使用绝对路径或相对于项目根目录的路径
        train_file_str = str(train_file.absolute()) if train_file.is_absolute() else str(train_file)
        dataset_config = {
            dataset_name: {
                "file_name": train_file_str,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
        }

        # 如果有测试集，添加测试集配置
        if test_file and test_file.exists():
            test_file_str = str(test_file.absolute()) if test_file.is_absolute() else str(test_file)
            dataset_config[f"{dataset_name}_eval"] = {
                "file_name": test_file_str,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }

        # 保存YAML配置文件（用于参考）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False)

        # 同时更新data/dataset_info.json（LLaMA-Factory需要这个文件）
        dataset_info_path = Path("data/dataset_info.json")
        dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取现有的dataset_info.json（如果存在）
        if dataset_info_path.exists():
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        # 更新数据集信息
        # 使用相对于data目录的路径
        train_file_relative = str(train_file.relative_to(Path("data"))) if Path("data") in train_file.parents else str(train_file)
        dataset_info[dataset_name] = {
            "file_name": train_file_relative,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
        
        if test_file and test_file.exists():
            test_file_relative = str(test_file.relative_to(Path("data"))) if Path("data") in test_file.parents else str(test_file)
            dataset_info[f"{dataset_name}_eval"] = {
                "file_name": test_file_relative,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
        
        # 保存dataset_info.json
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"✓ 数据集配置已创建: {output_path}")
        print(f"✓ 数据集信息已更新: {dataset_info_path}")

        return output_path

