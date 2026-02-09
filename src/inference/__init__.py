# -*- coding: utf-8 -*-
"""
推理模块

本模块提供多模态情感分析的推理接口，包括：
- Predictor: 统一推理接口，协调特征提取和模型预测
"""

from .predictor import Predictor

__all__ = ['Predictor']
