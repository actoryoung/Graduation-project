"""
特征提取器抽象基类模块

本模块定义了所有特征提取器的统一接口规范，确保不同模态的特征提取器
（文本、音频、视频）遵循相同的接口契约。

典型的特征提取流程：
    1. 加载原始数据（文本/音频/视频）
    2. 预处理（preprocess）- 标准化输入格式
    3. 特征提取（extract）- 生成数值特征向量
    4. 返回numpy数组用于下游任务

示例:
    >>> from src.features.base import FeatureExtractor
    >>>
    >>> class MyExtractor(FeatureExtractor):
    ...     def preprocess(self, raw_input):
    ...         # 实现预处理逻辑
    ...         return processed_data
    ...
    ...     def extract(self, data):
    ...         # 实现特征提取逻辑
    ...         return features
"""

# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np

from config import Config


class FeatureExtractor(ABC):
    """
    特征提取器抽象基类

    定义了所有模态特征提取器必须实现的接口。本类遵循接口隔离原则，
    仅定义核心方法，子类可以根据需要扩展额外功能。

    类属性:
        device (str): 计算设备 ('cuda' 或 'cpu')
        text_dim (int): 文本特征维度
        audio_dim (int): 音频特征维度
        video_dim (int): 视频特征维度

    示例:
        >>> class TextExtractor(FeatureExtractor):
        ...     def preprocess(self, text: str) -> dict:
        ...         return {'tokens': text.split()}
        ...
        ...     def extract(self, data: dict) -> np.ndarray:
        ...         return np.zeros(self.text_dim)
    """

    # 从配置类继承特征维度
    device: str = Config.device
    text_dim: int = Config.text_dim
    audio_dim: int = Config.audio_dim
    video_dim: int = Config.video_dim

    def __init__(self, device: str = None):
        """
        初始化特征提取器

        Args:
            device: 计算设备，默认为None时使用Config.device

        Raises:
            ValueError: 当device不是'cuda'或'cpu'时
        """
        if device is not None:
            if device not in ['cuda', 'cpu']:
                raise ValueError(
                    f"设备必须是 'cuda' 或 'cpu'，得到的是: {device}"
                )
            self.device = device
        else:
            self.device = Config.device

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """
        预处理原始输入数据

        本方法负责将原始输入（文本字符串、音频文件路径、视频文件路径等）
        转换为适合特征提取的格式。预处理步骤可能包括：
        - 文本：分词、截断、padding
        - 音频：重采样、降噪、归一化
        - 视频：帧提取、裁剪、缩放

        Args:
            raw_input: 原始输入数据，类型取决于具体模态：
                - 文本: str
                - 音频: str (文件路径) 或 np.ndarray (波形数据)
                - 视频: str (文件路径) 或 np.ndarray (帧数组)

        Returns:
            预处理后的数据，具体格式由子类决定

        Raises:
            ValueError: 当输入格式不正确时
            TypeError: 当输入类型不匹配时

        示例:
            >>> extractor = TextExtractor()
            >>> processed = extractor.preprocess("Hello world")
            >>> print(type(processed))
            <class 'dict'>
        """
        pass

    @abstractmethod
    def extract(self, data: Any) -> np.ndarray:
        """
        从预处理的的数据中提取特征

        本方法接收预处理后的数据，使用相应的预训练模型或算法
        提取固定维度的特征向量。

        Args:
            data: 预处理后的数据，格式应与preprocess的输出一致

        Returns:
            np.ndarray: 特征向量，形状取决于模态：
                - 文本: (text_dim,) 或 (sequence_length, text_dim)
                - 音频: (audio_dim,) 或 (time_steps, audio_dim)
                - 视频: (video_dim,) 或 (num_frames, video_dim)

        Raises:
            RuntimeError: 当特征提取过程出现错误时
            ValueError: 当输入数据维度不匹配时

        示例:
            >>> extractor = TextExtractor()
            >>> processed = extractor.preprocess("Hello")
            >>> features = extractor.extract(processed)
            >>> print(features.shape)
            (768,)
        """
        pass

    def extract_from_raw(self, raw_input: Any) -> np.ndarray:
        """
        便捷方法：直接从原始输入提取特征

        本方法组合了preprocess和extract两个步骤，提供一键式特征提取。

        Args:
            raw_input: 原始输入数据

        Returns:
            np.ndarray: 特征向量

        示例:
            >>> extractor = TextExtractor()
            >>> features = extractor.extract_from_raw("Hello world")
            >>> print(features.shape)
            (768,)
        """
        processed_data = self.preprocess(raw_input)
        return self.extract(processed_data)

    def __repr__(self) -> str:
        """
        返回特征提取器的字符串表示

        Returns:
            str: 类名和设备信息的字符串
        """
        return f"{self.__class__.__name__}(device='{self.device}')"

    def __str__(self) -> str:
        """
        返回特征提取器的可读字符串

        Returns:
            str: 包含类名、设备和特征维度的字符串
        """
        return (
            f"{self.__class__.__name__} - "
            f"设备: {self.device}"
        )
