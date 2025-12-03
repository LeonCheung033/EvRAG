"""Reranker微调模块

使用RAG-Retrieval进行Reranker微调
"""

import yaml
import subprocess
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


class RerankerFineTuner:
    """Reranker微调器"""

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        rag_retrieval_path: Optional[Path] = None,
    ):
        """
        初始化Reranker微调器

        Args:
            config_path: 训练配置文件路径
            output_dir: 模型输出目录
            rag_retrieval_path: RAG-Retrieval项目路径（可选，默认使用项目目录下的RAG-Retrieval）
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 确定RAG-Retrieval路径
        if rag_retrieval_path:
            self.rag_retrieval_path = Path(rag_retrieval_path)
        else:
            # 默认使用项目目录下的RAG-Retrieval
            # __file__: src/evrag/finetune/reranker_finetuner.py
            # 向上4级到项目根目录: finetune -> evrag -> src -> EvRAG
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            self.rag_retrieval_path = project_root / "RAG-Retrieval"

        if not self.rag_retrieval_path.exists():
            raise FileNotFoundError(
                f"RAG-Retrieval路径不存在: {self.rag_retrieval_path}\n"
                "请提供rag_retrieval_path参数或确保RAG-Retrieval在项目根目录下"
            )

    def prepare_config(
        self,
        model_path: str,
        train_file: Path,
        val_file: Path,
        **kwargs,
    ) -> Path:
        """
        准备训练配置文件

        Args:
            model_path: 基础模型路径
            train_file: 训练数据文件路径
            val_file: 验证数据文件路径
            **kwargs: 其他训练参数

        Returns:
            配置文件路径
        """
        # 读取基础配置
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # 更新配置
        config.update(
            {
                "model_name_or_path": model_path,
                "train_dataset": str(train_file),
                "val_dataset": str(val_file),
                "output_dir": str(self.output_dir),
                **kwargs,
            }
        )

        # 保存配置文件
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        print(f"✓ 训练配置已准备: {self.config_path}")

        return self.config_path

    def train(
        self,
        cuda_visible_devices: Optional[str] = None,
    ) -> None:
        """
        执行训练

        Args:
            cuda_visible_devices: 可见的GPU设备（如"0"）
        """

        # 设置环境变量
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        print("开始训练...")
        print(f"  配置文件: {self.config_path}")
        print(f"  输出目录: {self.output_dir}")
        if cuda_visible_devices:
            print(f"  GPU设备: {cuda_visible_devices}")

        # RAG-Retrieval的训练需要在特定目录下运行
        train_dir = self.rag_retrieval_path / "rag_retrieval" / "train" / "reranker"

        if not train_dir.exists():
            raise FileNotFoundError(f"训练目录不存在: {train_dir}")

        # 创建日志目录和日志文件
        log_dir = Path(__file__).parent.parent.parent.parent / "logs" / "finetune"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"reranker_training_{timestamp}.log"

        print(f"  训练日志: {log_file}")

        # 读取配置文件，将相对路径转换为绝对路径
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # 将相对路径转换为绝对路径（相对于项目根目录）
        project_root = Path(__file__).parent.parent.parent.parent
        if (
            "train_dataset" in config
            and not Path(config["train_dataset"]).is_absolute()
        ):
            config["train_dataset"] = str(
                (project_root / config["train_dataset"]).absolute()
            )
        if "val_dataset" in config and not Path(config["val_dataset"]).is_absolute():
            config["val_dataset"] = str(
                (project_root / config["val_dataset"]).absolute()
            )
        if "output_dir" in config and not Path(config["output_dir"]).is_absolute():
            config["output_dir"] = str((project_root / config["output_dir"]).absolute())

        # 创建临时配置文件（使用绝对路径）
        temp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(config, temp_config, allow_unicode=True, default_flow_style=False)
        temp_config.close()
        temp_config_path = temp_config.name

        # 使用subprocess在正确的目录下执行训练
        # 因为RAG-Retrieval使用相对导入，需要在特定目录下运行
        cmd = [
            "python",
            "train_reranker.py",
            "--config",
            temp_config_path,
        ]

        # 执行训练命令，同时保存日志
        with open(log_file, "w", encoding="utf-8") as log_file_handle:
            try:
                # 写入训练开始信息
                log_file_handle.write("=" * 70 + "\n")
                log_file_handle.write("Reranker Fine-tuning Log\n")
                log_file_handle.write(
                    f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                log_file_handle.write(f"配置文件: {self.config_path}\n")
                log_file_handle.write(f"输出目录: {self.output_dir}\n")
                if cuda_visible_devices:
                    log_file_handle.write(f"GPU设备: {cuda_visible_devices}\n")
                log_file_handle.write("=" * 70 + "\n\n")
                log_file_handle.flush()

                # 使用Popen实时读取输出
                process = subprocess.Popen(
                    cmd,
                    cwd=str(train_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # 行缓冲
                )

                # 实时读取输出并同时写入日志和控制台
                return_code = None
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        # 同时写入日志文件和控制台
                        log_file_handle.write(output)
                        log_file_handle.flush()
                        print(output, end="")

                return_code = process.poll()

                # 写入训练结束信息
                log_file_handle.write("\n" + "=" * 70 + "\n")
                log_file_handle.write(
                    f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                log_file_handle.write(f"退出码: {return_code}\n")
                log_file_handle.write("=" * 70 + "\n")

                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)

                # 删除临时配置文件
                try:
                    os.unlink(temp_config_path)
                except:
                    pass

                print(f"✓ 训练完成: {self.output_dir}")
                print(f"✓ 训练日志已保存: {log_file}")
            except subprocess.CalledProcessError as e:
                # 如果训练失败，也要保存错误日志
                log_file_handle.write("\n" + "=" * 70 + "\n")
                log_file_handle.write("训练失败\n")
                log_file_handle.write(f"退出码: {e.returncode}\n")
                log_file_handle.write("=" * 70 + "\n")
                # 删除临时配置文件
                try:
                    if "temp_config_path" in locals():
                        os.unlink(temp_config_path)
                except:
                    pass
                print(f"✗ 训练失败，返回码: {e.returncode}")
                print(f"✗ 错误日志已保存: {log_file}")
                raise
            except Exception as e:
                # 保存异常信息
                log_file_handle.write("\n" + "=" * 70 + "\n")
                log_file_handle.write(f"训练异常: {str(e)}\n")
                log_file_handle.write("=" * 70 + "\n")
                # 删除临时配置文件
                try:
                    if "temp_config_path" in locals():
                        os.unlink(temp_config_path)
                except:
                    pass
                print(f"✗ 训练失败: {e}")
                print(f"✗ 错误日志已保存: {log_file}")
                raise

    def get_checkpoints(self) -> list[Path]:
        """
        获取所有checkpoint路径

        Returns:
            checkpoint路径列表
        """
        checkpoints = []
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    checkpoints.append(item)
        return sorted(
            checkpoints, key=lambda x: int(x.name.split("-")[1]) if "-" in x.name else 0
        )
