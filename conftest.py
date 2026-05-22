"""pytest 全局配置：将项目根目录加入 sys.path。

这样所有测试中 from src.xxx import xxx 都能正常工作，
无需每次手动设置 PYTHONPATH。
"""
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))