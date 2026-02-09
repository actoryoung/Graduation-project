# -*- coding: utf-8 -*-
"""
多模态情感分析模型模块

本模块包含所有模型架构：
- MultimodalFusionModule: 基础多模态融合模型
- BERTHybridModel: BERT混合模型（69.97%准确率）
- ImprovedModel: 改进的临时模型
"""

from .fusion_module import MultimodalFusionModule
from .bert_hybrid_model import BERTHybridModel

__all__ = ['MultimodalFusionModule', 'BERTHybridModel']
