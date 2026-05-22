"""配置入口门面 — 加载环境变量 + 导出 Settings 单例。

本模块是 RAG 系统的配置入口，职责：
1. 调用 load_dotenv(override=True) 确保 .env 文件中的环境变量可用
2. 创建并导出 settings 单例供全局使用

使用方式：
    from src.core.config import settings

    api_key = settings.deepseek_api_key
    persist_dir = settings.chroma_persist_directory
"""


from dotenv import load_dotenv

# override=True 确保 .env 文件中的值覆盖已存在的环境变量
# 为什么在模块级调用：Settings() 实例化需要读取环境变量，
# load_dotenv 必须先于 Settings() 执行
load_dotenv(override=True)

from src.core.settings import Settings

# Settings 单例 — 全局唯一配置对象
# 为什么在此实例化：load_dotenv 必须先于 Settings() 执行，
# 而 config.py 是最先被导入的核心模块，天然保证顺序
settings = Settings()

__all__ = [
    "settings",
]
