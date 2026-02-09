# -*- coding: utf-8 -*-
"""
特征提取模块

本模块提供多模态情感分析系统中各模态的特征提取器。
所有特征提取器都继承自 FeatureExtractor 抽象基类。

可用提取器:
    - FeatureExtractor: 抽象基类
    - BERTFeatureExtractor: 文本特征提取器（BERT）
    - Wav2VecFeatureExtractor: 音频特征提取器（Wav2Vec 2.0）
    - OpenFaceFeatureExtractor: 视频特征提取器（OpenFace）

示例:
    >>> from src.features import BERTFeatureExtractor, Wav2VecFeatureExtractor
    >>> text_extractor = BERTFeatureExtractor()
    >>> audio_extractor = Wav2VecFeatureExtractor()
"""

from src.features.base import FeatureExtractor
from src.features.text_features import BERTFeatureExtractor
from src.features.audio_features import Wav2VecFeatureExtractor
from src.features.video_features import OpenFaceFeatureExtractor

__all__ = [
    'FeatureExtractor',
    'BERTFeatureExtractor',
    'Wav2VecFeatureExtractor',
    'OpenFaceFeatureExtractor',
]
