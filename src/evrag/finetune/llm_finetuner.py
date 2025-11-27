"""LLM微调模块

使用LLaMA-Factory进行LLM微调
"""

import yaml
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class LLMFineTuner:
    """LLM微调器"""

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
    ):
        """
        初始化LLM微调器

        Args:
            config_path: 训练配置文件路径
            output_dir: 模型输出目录
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_config(
        self,
        model_path: str,
        dataset_name: str,
        train_file: Path,
        test_file: Optional[Path] = None,
        **kwargs,
    ) -> Path:
        """
        准备训练配置文件

        Args:
            model_path: 基础模型路径
            dataset_name: 数据集名称
            train_file: 训练数据文件路径
            test_file: 测试数据文件路径（可选）
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
        config.update({
            "model_name_or_path": model_path,
            "dataset": dataset_name,
            "output_dir": str(self.output_dir),
            **kwargs,
        })

        # 如果有测试集，添加评估配置
        if test_file and test_file.exists():
            config["do_eval"] = True
            config["eval_strategy"] = "steps"
            config["eval_steps"] = config.get("eval_steps", 100)

        # 保存配置文件
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        print(f"✓ 训练配置已准备: {self.config_path}")

        return self.config_path

    def train(
        self,
        cuda_visible_devices: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """
        执行训练

        Args:
            cuda_visible_devices: 可见的GPU设备（如"0,1,2,3"）
            resume_from_checkpoint: 从checkpoint恢复训练
        """
        import os

        # 设置环境变量
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        # 如果从checkpoint恢复，更新配置
        if resume_from_checkpoint:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            config["resume_from_checkpoint"] = resume_from_checkpoint
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        print(f"开始训练...")
        print(f"  配置文件: {self.config_path}")
        print(f"  输出目录: {self.output_dir}")
        if cuda_visible_devices:
            print(f"  GPU设备: {cuda_visible_devices}")

        # 使用llamafactory-cli命令启动训练（LLaMA-Factory要求通过CLI启动）
        # 构建命令: llamafactory-cli train <config_file>
        config_file = str(self.config_path.absolute())
        # 使用shutil.which查找llamafactory-cli命令
        import shutil
        llamafactory_cli = shutil.which("llamafactory-cli")
        if not llamafactory_cli:
            # 如果找不到，使用python -m方式
            cmd = [sys.executable, "-m", "llamafactory.cli", "train", config_file]
        else:
            cmd = [llamafactory_cli, "train", config_file]

        # 准备环境变量
        env = dict(os.environ)
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        # 添加NCCL环境变量以解决分布式训练hang问题
        # 单机多卡训练优化配置
        if cuda_visible_devices and "," in cuda_visible_devices:
            # 多GPU训练时的NCCL配置
            env.setdefault("NCCL_P2P_DISABLE", "1")  # 禁用P2P，使用更稳定的通信方式
            env.setdefault("NCCL_SHM_DISABLE", "0")  # 启用共享内存
            env.setdefault("NCCL_SOCKET_IFNAME", "lo")  # 使用loopback接口（单机多卡）
            env.setdefault("NCCL_IB_DISABLE", "1")  # 禁用InfiniBand
            env.setdefault("NCCL_DEBUG", "INFO")  # 设置INFO级别以便调试
            env.setdefault("NCCL_TIMEOUT", "1800")  # 增加超时时间到30分钟
            env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")  # 使用阻塞等待，更稳定（使用新变量名）
            env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")  # 启用异步错误处理

        # 创建日志目录和日志文件
        log_dir = Path(__file__).parent.parent.parent.parent / "logs" / "finetune"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"llm_training_{timestamp}.log"
        
        print(f"  训练日志: {log_file}")

        # 执行训练命令，同时保存日志
        with open(log_file, "w", encoding="utf-8") as log_file_handle:
            try:
                # 写入训练开始信息
                log_file_handle.write("=" * 70 + "\n")
                log_file_handle.write("LLM Fine-tuning Log\n")
                log_file_handle.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file_handle.write(f"配置文件: {self.config_path}\n")
                log_file_handle.write(f"输出目录: {self.output_dir}\n")
                if cuda_visible_devices:
                    log_file_handle.write(f"GPU设备: {cuda_visible_devices}\n")
                log_file_handle.write("=" * 70 + "\n\n")
                log_file_handle.flush()
                
                # 使用Popen实时读取输出
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=Path(__file__).parent.parent.parent.parent,  # 项目根目录
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # 行缓冲
                )
                
                # 实时读取输出并同时写入日志和控制台
                return_code = None
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # 同时写入日志文件和控制台
                        log_file_handle.write(output)
                        log_file_handle.flush()
                        print(output, end="")
                
                return_code = process.poll()
                
                # 写入训练结束信息
                log_file_handle.write("\n" + "=" * 70 + "\n")
                log_file_handle.write(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file_handle.write(f"退出码: {return_code}\n")
                log_file_handle.write("=" * 70 + "\n")
                
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
                
                print(f"✓ 训练完成: {self.output_dir}")
                print(f"✓ 训练日志已保存: {log_file}")
            except subprocess.CalledProcessError as e:
                # 如果训练失败，也要保存错误日志
                log_file_handle.write("\n" + "=" * 70 + "\n")
                log_file_handle.write("训练失败\n")
                log_file_handle.write(f"退出码: {e.returncode}\n")
                log_file_handle.write("=" * 70 + "\n")
                print(f"✗ 训练失败: 退出码 {e.returncode}")
                print(f"✗ 错误日志已保存: {log_file}")
                raise
            except Exception as e:
                # 保存异常信息
                log_file_handle.write("\n" + "=" * 70 + "\n")
                log_file_handle.write(f"训练异常: {str(e)}\n")
                log_file_handle.write("=" * 70 + "\n")
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
        return sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        获取最佳checkpoint（通常是最后一个）

        Returns:
            最佳checkpoint路径
        """
        checkpoints = self.get_checkpoints()
        return checkpoints[-1] if checkpoints else None

