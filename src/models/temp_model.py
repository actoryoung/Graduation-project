# -*- coding: utf-8 -*-
"""
临时训练模型 - 用于加载best_model_temp.pth

这是临时训练时使用的模型架构，包含独立的模态编码器。
仅用于加载临时模型，最终版本应使用MultimodalFusionModule。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ImprovedModel(nn.Module):
    """
    临时模型架构（用于加载best_model_temp.pth）

    结构：
        - 独立的文本/音频/视频编码器
        - 特征融合层
        - 分类层

    注意：此模型仅用于临时演示，完整训练后应使用MultimodalFusionModule
    """

    def __init__(
        self,
        text_dim: int = 300,
        audio_dim: int = 74,
        video_dim: int = 25,
        num_classes: int = 7,
        device: torch.device = None
    ):
        super().__init__()

        # 设置设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # 视频编码器
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # 将模型移动到设备
        self.to(self.device)

    def forward(self, text, audio, video):
        """前向传播

        Args:
            text: 文本特征 [batch, 300]
            audio: 音频特征 [batch, 74] 或 None
            video: 视频特征 [batch, 25] 或 None

        Returns:
            logits: 分类输出 [batch, 7]
        """
        batch_size = text.shape[0]

        # 处理缺失模态 - 使用零向量
        if audio is None:
            audio = torch.zeros(batch_size, 74, device=self.device)
        else:
            audio = audio.to(self.device)

        if video is None:
            video = torch.zeros(batch_size, 25, device=self.device)
        else:
            video = video.to(self.device)

        text = text.to(self.device)

        t = self.text_encoder(text)
        a = self.audio_encoder(audio)
        v = self.video_encoder(video)
        x = torch.cat([t, a, v], dim=1)
        return self.fusion(x)

    def predict(
        self,
        text_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        video_feat: Optional[torch.Tensor] = None
    ) -> Dict:
        """预测情感，返回完整结果

        Args:
            text_feat: 文本特征 [batch_size, 300] 或 None
            audio_feat: 音频特征 [batch_size, 74] 或 None
            video_feat: 视频特征 [batch_size, 25] 或 None

        Returns:
            结果字典，包含：
                - emotion: 预测的情感类别
                - confidence: 预测置信度
                - probabilities: 各情感类别的概率字典
        """
        self.eval()

        with torch.no_grad():
            # 前向传播获取logits
            logits = self.forward(text_feat, audio_feat, video_feat)

            # 7类情感
            emotions = [
                'strong_negative', 'negative', 'weak_negative',
                'neutral', 'weak_positive', 'positive', 'strong_positive'
            ]

            # 计算概率分布
            probabilities = F.softmax(logits, dim=1)

            # 获取预测结果
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_emotion = emotions[predicted_idx.item()]

            # 构建概率字典
            prob_dict = {
                emotions[i]: probabilities[0, i].item()
                for i in range(len(emotions))
            }

            # 构建结果字典
            result = {
                'emotion': predicted_emotion,
                'confidence': confidence.item(),
                'probabilities': prob_dict
            }

        return result


def load_temp_model(model_path: str, device: torch.device):
    """加载临时模型

    Args:
        model_path: 模型文件路径
        device: 运行设备

    Returns:
        加载好的模型
    """
    model = ImprovedModel().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model
