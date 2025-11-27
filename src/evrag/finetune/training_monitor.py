"""训练监控模块

实时监控训练指标和GPU使用情况
"""

import time
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import threading


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, log_dir: Path, gpu_log_path: Optional[Path] = None):
        """
        初始化训练监控器

        Args:
            log_dir: 日志目录
            gpu_log_path: GPU监控日志文件路径
        """
        self.log_dir = Path(log_dir)
        self.gpu_log_path = Path(gpu_log_path) if gpu_log_path else self.log_dir / "gpu_monitor.log"
        self.gpu_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_gpu_monitoring(self, interval: int = 5):
        """
        启动GPU监控

        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring:
            print("GPU监控已在运行")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_gpu_loop,
            args=(interval,),
            daemon=True,
        )
        self.monitor_thread.start()
        print(f"✓ GPU监控已启动，日志保存到: {self.gpu_log_path}")

    def stop_gpu_monitoring(self):
        """停止GPU监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("✓ GPU监控已停止")

    def _monitor_gpu_loop(self, interval: int):
        """GPU监控循环"""
        with open(self.gpu_log_path, "a", encoding="utf-8") as f:
            while self.monitoring:
                try:
                    # 获取GPU信息
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for line in result.stdout.strip().split("\n"):
                            if line.strip():
                                f.write(f"{timestamp},{line}\n")
                                f.flush()
                except Exception as e:
                    print(f"GPU监控错误: {e}")

                time.sleep(interval)

    def parse_gpu_log(self) -> List[Dict]:
        """
        解析GPU监控日志

        Returns:
            GPU监控数据列表
        """
        if not self.gpu_log_path.exists():
            return []

        data = []
        with open(self.gpu_log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 8:
                    data.append({
                        "timestamp": row[0],
                        "gpu_index": int(row[1]),
                        "gpu_name": row[2],
                        "utilization": float(row[3].replace("%", "")),
                        "memory_used": int(row[4].replace("MiB", "").strip()),
                        "memory_total": int(row[5].replace("MiB", "").strip()),
                        "temperature": int(row[6].replace("C", "").strip()),
                    })

        return data

    def get_tensorboard_logs(self, log_dir: Path) -> Dict:
        """
        获取TensorBoard日志信息

        Args:
            log_dir: TensorBoard日志目录

        Returns:
            日志信息字典
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return {}

        # 查找TensorBoard事件文件
        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            return {}

        return {
            "log_dir": str(log_dir),
            "event_files": [str(f) for f in event_files],
            "latest_event_file": str(max(event_files, key=lambda x: x.stat().st_mtime)),
        }

