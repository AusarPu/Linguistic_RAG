#!/usr/bin/env python3
"""
数据集下载器包
包含各种数据集的独立下载器，支持看门狗测速功能
"""

from .base_downloader import BaseDownloader
from .download_natural_questions import NaturalQuestionsDownloader
from .download_hotpotqa import HotpotQADownloader
from .download_triviaqa import TriviaQADownloader
from .download_msmarco import MSMarcoDownloader

__version__ = "1.0.0"
__author__ = "RAG System"

__all__ = [
    'BaseDownloader',
    'NaturalQuestionsDownloader',
    'HotpotQADownloader', 
    'TriviaQADownloader',
    'MSMarcoDownloader'
]