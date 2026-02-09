# -*- coding: utf-8 -*-
"""
BERT混合多模态融合模型

该模型在CMU-MOSEI数据集上训练，使用以下特征：
- BERT文本特征 (768维)
- COVAREP音频特征 (74维)
- OpenFace视频特征 (710维)

测试准确率: 69.97%
验证准确率: 70.67% (Epoch 11)
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTHybridModel(nn.Module):
    """
    BERT混合多模态融合模型

    模型架构：
        1. 文本编码器：768 -> 256
        2. 音频编码器：74 -> 128
        3. 视频编码器：710 -> 256
        4. 融合层：600 -> 512 -> 256
        5. 分类器：256 -> 7

    Attributes:
        text_encoder: 文本特征编码器
        audio_encoder: 音频特征编码器
        video_encoder: 视频特征编码器
        fusion: 融合层
        classifier: 情感分类器
        device: 计算设备（动态获取）
    """

    def __init__(
        self,
        text_dim=768,      # BERT特征
        audio_dim=74,      # COVAREP特征
        video_dim=710,     # OpenFace特征
        fusion_dim=512,
        num_classes=7,
        dropout_rate=0.3
    ):
        super(BERTHybridModel, self).__init__()

        # 模态特定编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 融合层
        total_dim = 256 + 128 + 256  # 600
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.65)
        )

        # 分类器
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    @property
    def device(self):
        """动态获取模型当前所在的设备"""
        try:
            # 获取第一个参数的设备
            return next(self.parameters()).device
        except StopIteration:
            # 如果模型没有参数，返回CPU
            return torch.device('cpu')

    def forward(self, text, audio=None, video=None) -> torch.Tensor:
        """前向传播

        Args:
            text: 文本特征 [batch, 768]
            audio: 音频特征 [batch, 74] 或 None
            video: 视频特征 [batch, 710] 或 None

        Returns:
            logits: 分类logits [batch, 7]
        """
        batch_size = text.shape[0]
        device = text.device

        # 处理缺失模态
        if audio is None:
            audio = torch.zeros(batch_size, 74, device=device)
        if video is None:
            video = torch.zeros(batch_size, 710, device=device)

        # 模态编码
        t_feat = self.text_encoder(text)
        a_feat = self.audio_encoder(audio)
        v_feat = self.video_encoder(video)

        # 拼接并融合
        fused = torch.cat([t_feat, a_feat, v_feat], dim=1)
        features = self.fusion(fused)
        logits = self.classifier(features)

        return logits

    def predict(
        self,
        text_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        video_feat: Optional[torch.Tensor] = None
    ) -> Dict:
        """预测情感，返回完整结果

        Args:
            text_feat: 文本特征 [batch, 768] 或 None
            audio_feat: 音频特征 [batch, 74] 或 None
            video_feat: 视频特征 [batch, 710] 或 None

        Returns:
            包含以下键的字典:
                - emotion (str): 预测的情感类别
                - confidence (float): 预测置信度
                - probabilities (dict): 各情感类别的概率
        """
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from config import EMOTIONS

        # 前向传播
        self.eval()
        with torch.no_grad():
            logits = self.forward(text_feat, audio_feat, video_feat)

        # 计算概率
        probabilities = F.softmax(logits, dim=1)[0]

        # 获取预测结果
        pred_idx = probabilities.argmax().item()
        confidence = probabilities[pred_idx].item()

        # 构建概率字典
        prob_dict = {
            emotion: probabilities[i].item()
            for i, emotion in enumerate(EMOTIONS)
        }

        return {
            'emotion': EMOTIONS[pred_idx],
            'confidence': confidence,
            'probabilities': prob_dict
        }
