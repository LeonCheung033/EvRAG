"""
性能监控工具

用于记录任务执行时间、CPU、内存、GPU等资源使用情况。
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, task_name: str, output_dir: Optional[Path] = None):
        """
        初始化性能监控器
        
        Args:
            task_name: 任务名称
            output_dir: 输出目录（可选）
        """
        self.task_name = task_name
        self.output_dir = output_dir or Path("logs/performance")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.steps: List[Dict[str, Any]] = []
        self.resource_samples: List[Dict[str, Any]] = []
        
        # 初始化GPU监控（如果可用）
        self.gpu_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                self.gpu_available = False
        
        # 记录初始资源状态
        self._record_resource_sample("start")
    
    def _get_cpu_memory_info(self) -> Dict[str, Any]:
        """获取CPU和内存信息"""
        if not PSUTIL_AVAILABLE:
            return {}
        
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_rss_mb": memory_info.rss / 1024 / 1024,  # MB
            "memory_vms_mb": memory_info.vms / 1024 / 1024,  # MB
            "system_cpu_percent": psutil.cpu_percent(interval=0.1),
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        if not self.gpu_available:
            return {}
        
        gpu_info = {}
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 获取GPU利用率
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # 获取显存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # 获取GPU名称
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                # 处理不同版本的pynvml返回类型
                if isinstance(name_bytes, bytes):
                    name = name_bytes.decode('utf-8')
                else:
                    name = str(name_bytes)
                
                # 获取温度
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                gpu_info[f"gpu_{i}"] = {
                    "name": name,
                    "utilization_gpu_percent": util.gpu,
                    "utilization_memory_percent": util.memory,
                    "memory_used_mb": mem_info.used / 1024 / 1024,  # MB
                    "memory_total_mb": mem_info.total / 1024 / 1024,  # MB
                    "memory_free_mb": mem_info.free / 1024 / 1024,  # MB
                    "memory_used_percent": (mem_info.used / mem_info.total) * 100,
                    "temperature_c": temp,
                }
        except Exception as e:
            gpu_info["error"] = str(e)
        
        return gpu_info
    
    def _record_resource_sample(self, label: str):
        """记录资源使用样本"""
        sample = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "elapsed_seconds": time.time() - self.start_time if self.start_time else 0,
        }
        
        # CPU和内存
        sample.update(self._get_cpu_memory_info())
        
        # GPU
        if self.gpu_available:
            sample.update(self._get_gpu_info())
        
        self.resource_samples.append(sample)
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self._record_resource_sample("start")
    
    def end(self):
        """结束监控"""
        self.end_time = time.time()
        self._record_resource_sample("end")
    
    @contextmanager
    def step(self, step_name: str):
        """
        步骤上下文管理器
        
        Usage:
            with monitor.step("PDF解析"):
                # 执行任务
                pass
        """
        step_start = time.time()
        self._record_resource_sample(f"step_{step_name}_start")
        
        try:
            yield
        finally:
            step_end = time.time()
            step_duration = step_end - step_start
            self._record_resource_sample(f"step_{step_name}_end")
            
            self.steps.append({
                "name": step_name,
                "start_time": step_start,
                "end_time": step_end,
                "duration_seconds": step_duration,
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        total_duration = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0
        
        # 计算平均资源使用
        if self.resource_samples:
            avg_cpu = sum(s.get("cpu_percent", 0) for s in self.resource_samples) / len(self.resource_samples)
            avg_memory_mb = sum(s.get("memory_rss_mb", 0) for s in self.resource_samples) / len(self.resource_samples)
            max_memory_mb = max(s.get("memory_rss_mb", 0) for s in self.resource_samples)
        else:
            avg_cpu = 0
            avg_memory_mb = 0
            max_memory_mb = 0
        
        summary = {
            "task_name": self.task_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "total_duration_seconds": total_duration,
            "total_duration_formatted": self._format_duration(total_duration),
            "steps": self.steps,
            "step_summary": {
                step["name"]: {
                    "duration_seconds": step["duration_seconds"],
                    "duration_formatted": self._format_duration(step["duration_seconds"]),
                }
                for step in self.steps
            },
            "resource_usage": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_mb": avg_memory_mb,
                "max_memory_mb": max_memory_mb,
            },
        }
        
        # 添加GPU摘要
        if self.gpu_available and self.resource_samples:
            gpu_summary = {}
            for i in range(self.gpu_count):
                gpu_key = f"gpu_{i}"
                samples = [s for s in self.resource_samples if gpu_key in s]
                if samples:
                    gpu_summary[gpu_key] = {
                        "name": samples[0][gpu_key].get("name", "Unknown"),
                        "avg_utilization_percent": sum(s[gpu_key].get("utilization_gpu_percent", 0) for s in samples) / len(samples),
                        "max_utilization_percent": max(s[gpu_key].get("utilization_gpu_percent", 0) for s in samples),
                        "avg_memory_used_mb": sum(s[gpu_key].get("memory_used_mb", 0) for s in samples) / len(samples),
                        "max_memory_used_mb": max(s[gpu_key].get("memory_used_mb", 0) for s in samples),
                    }
            summary["resource_usage"]["gpu"] = gpu_summary
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}分{secs:.2f}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}小时{minutes}分{secs:.2f}秒"
    
    def save_report(self, filename: Optional[str] = None):
        """
        保存性能报告
        
        Args:
            filename: 文件名（可选，默认使用任务名称和时间戳）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.task_name}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        report = {
            "summary": self.get_summary(),
            "detailed_samples": self.resource_samples,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print(f"性能监控报告: {summary['task_name']}")
        print(f"{'='*60}")
        print(f"总耗时: {summary['total_duration_formatted']}")
        print(f"\n步骤耗时:")
        for step_name, step_info in summary['step_summary'].items():
            print(f"  - {step_name}: {step_info['duration_formatted']}")
        
        print(f"\n资源使用:")
        print(f"  - 平均CPU使用率: {summary['resource_usage']['avg_cpu_percent']:.2f}%")
        print(f"  - 平均内存使用: {summary['resource_usage']['avg_memory_mb']:.2f} MB")
        print(f"  - 峰值内存使用: {summary['resource_usage']['max_memory_mb']:.2f} MB")
        
        if 'gpu' in summary['resource_usage']:
            print(f"\nGPU使用:")
            for gpu_key, gpu_info in summary['resource_usage']['gpu'].items():
                print(f"  - {gpu_info['name']} ({gpu_key}):")
                print(f"    - 平均利用率: {gpu_info['avg_utilization_percent']:.2f}%")
                print(f"    - 峰值利用率: {gpu_info['max_utilization_percent']:.2f}%")
                print(f"    - 平均显存使用: {gpu_info['avg_memory_used_mb']:.2f} MB")
                print(f"    - 峰值显存使用: {gpu_info['max_memory_used_mb']:.2f} MB")
        
        print(f"{'='*60}\n")

