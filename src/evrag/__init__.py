"""
EvRAG - Enhanced Vector Retrieval-Augmented Generation

一个增强的检索增强生成系统，支持多种检索器和重排序方法。
"""

__version__ = "0.1.0"
__author__ = "Leon Cheung"

from .config import Settings, get_settings, reload_settings
from .tool_func import merge_docs, post_processing

__all__ = [
    "Settings",
    "get_settings",
    "reload_settings",
    "__version__",
    "__author__",
    "merge_docs",
    "post_processing",
]
