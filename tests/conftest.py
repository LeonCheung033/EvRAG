"""
pytest配置文件

这个文件会被pytest自动加载，用于定义：
- 共享的fixtures（测试夹具）
- 测试配置
- 钩子函数
"""

import pytest
from pathlib import Path

# 定义项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def project_root():
    """返回项目根目录路径"""
    return PROJECT_ROOT


# 可以在这里添加更多共享的fixtures
# 例如：测试用的临时目录、mock对象等