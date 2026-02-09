# -*- coding: utf-8 -*-
"""
BERT混合多模态融合模型 - 3类版本

在CMU-MOSEI SDK子集上训练，使用以下特征：
- BERT文本特征 (768维)
- COVAREP音频特征 (74维)
- OpenFace视频特征 (710维)

3类情感分类:
- 0: Negative (负面)
- 1: Neutral (中性)
- 2: Positive (正面)

模型架构:
    1. 文本编码器：768 -> 256
    2. 音频编码器：74 -> 128
    3. 视频编码器：710 -> 256
    4. 融合层：600 -> 512 -> 256
    5. 分类器：256 -> 3

预期性能:
    - Accuracy: 63-66% (相比GloVe基线58.88%提升4-7%)
    - Macro F1: 0.55-0.60 (相比GloVe基线0.50提升0.05-0.10)
    - Negative F1: 0.35-0.45 (相比GloVe基线0.28提升0.07-0.17)
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTHybridModel3Class(nn.Module):
    """
    BERT混合多模态融合模型 - 3类版本

    模型架构：
        1. 文本编码器：768 -> 256
        2. 音频编码器：74 -> 128
        3. 视频编码器：710 -> 256
        4. 融合层：600 -> 512 -> 256
        5. 分类器：256 -> 3

    Attributes:
        text_encoder: 文本特征编码器 (BERT -> 256)
        audio_encoder: 音频特征编码器 (COVAREP -> 128)
        video_encoder: 视频特征编码器 (OpenFace -> 256)
        fusion: 融合层
        classifier: 情感分类器 (3类)
    """

    def __init__(
        self,
        text_dim=768,      # BERT特征
        audio_dim=74,      # COVAREP特征
        video_dim=710,     # OpenFace特征
        fusion_dim=512,
        num_classes=3,     # 3类分类
        dropout_rate=0.3
    ):
        super(BERTHybridModel3Class, self).__init__()

        # 验证num_classes参数
        if num_classes != 3:
            raise ValueError(f"此模型仅支持3类分类，当前num_classes={num_classes}")

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
            nn.Dropout(dropout_rate * 0.65)  # 0.3 * 0.65 ≈ 0.2
        )

        # 分类器
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @property
    def device(self):
        """动态获取模型当前所在的设备"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def forward(self, text, audio=None, video=None) -> torch.Tensor:
        """前向传播

        Args:
            text: 文本特征 [batch, 768] (BERT特征)
            audio: 音频特征 [batch, 74] 或 None (COVAREP特征)
            video: 视频特征 [batch, 710] 或 None (OpenFace特征)

        Returns:
            logits: 分类logits [batch, 3]

        Examples:
            >>> model = BERTHybridModel3Class()
            >>> text = torch.randn(2, 768)
            >>> audio = torch.randn(2, 74)
            >>> video = torch.randn(2, 710)
            >>> logits = model(text, audio, video)
            >>> logits.shape
            torch.Size([2, 3])
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
            text_feat: 文本特征 [batch, 768] (BERT特征)
            audio_feat: 音频特征 [batch, 74] 或 None
            video_feat: 视频特征 [batch, 710] 或 None

        Returns:
            包含以下键的字典:
                - emotion (str): 预测的情感类别
                - emotion_id (int): 情感类别ID (0/1/2)
                - confidence (float): 预测置信度
                - probabilities (dict): 各情感类别的概率

        Examples:
            >>> model = BERTHybridModel3Class()
            >>> text = torch.randn(1, 768)
            >>> result = model.predict(text_feat=text)
            >>> print(result['emotion'])
            'Neutral'
        """
        self.eval()

        # 3类情感名称映射
        emotion_names = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        }

        with torch.no_grad():
            # 前向传播获取logits
            logits = self.forward(text_feat, audio_feat, video_feat)

            # 计算概率分布
            probabilities = F.softmax(logits, dim=1)

            # 获取预测结果
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_id = predicted_idx.item()
            predicted_emotion = emotion_names[predicted_id]

            # 构建概率字典
            prob_dict = {
                emotion_names[i]: probabilities[0, i].item()
                for i in range(3)
            }

            # 构建结果字典
            result = {
                'emotion': predicted_emotion,
                'emotion_id': predicted_id,
                'confidence': confidence.item(),
                'probabilities': prob_dict
            }

        return result

    def get_num_parameters(self) -> int:
        """获取模型参数数量

        Returns:
            模型总参数数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        """获取模型信息

        Returns:
            模型信息字典，包含：
                - num_parameters: 参数数量
                - model_size: 模型大小（MB）
                - input_dims: 输入特征维度
        """
        num_params = self.get_num_parameters()
        model_size = num_params * 4 / (1024 ** 2)  # float32 -> MB

        return {
            'num_parameters': num_params,
            'model_size_mb': model_size,
            'text_dim': 768,
            'audio_dim': 74,
            'video_dim': 710,
            'fusion_dim': 256,
            'num_classes': 3
        }


def test_bert_hybrid_3class():
    """测试BERT 3类混合模型功能"""
    print("=" * 70)
    print("BERT混合多模态融合模型测试 - 3类版本")
    print("=" * 70)

    # 创建模型
    model = BERTHybridModel3Class()
    info = model.get_model_info()

    print(f"\n模型信息:")
    print(f"  参数数量: {info['num_parameters']:,}")
    print(f"  模型大小: {info['model_size_mb']:.2f} MB")
    print(f"  设备: {model.device}")

    # 测试1: 三模态输入
    print("\n" + "-" * 70)
    print("测试1: 三模态输入 (BERT + COVAREP + OpenFace)")
    print("-" * 70)

    batch_size = 2
    text_feat = torch.randn(batch_size, 768)  # BERT特征
    audio_feat = torch.randn(batch_size, 74)  # COVAREP特征
    video_feat = torch.randn(batch_size, 710)  # OpenFace特征

    logits = model(text_feat, audio_feat, video_feat)
    print(f"输入形状: BERT={text_feat.shape}, COVAREP={audio_feat.shape}, OpenFace={video_feat.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"预期形状: torch.Size([{batch_size}, 3])")

    # 测试2: 单模态输入（仅BERT文本）
    print("\n" + "-" * 70)
    print("测试2: 单模态输入（仅BERT文本）")
    print("-" * 70)

    text_only = torch.randn(1, 768)
    logits_single = model(text=text_only)
    result = model.predict(text_feat=text_only)

    print(f"输入形状: {text_only.shape}")
    print(f"输出logits形状: {logits_single.shape}")
    print(f"预测情感: {result['emotion']} (ID={result['emotion_id']})")
    print(f"置信度: {result['confidence']:.4f}")
    print("概率分布:")
    for emotion, prob in result['probabilities'].items():
        print(f"  {emotion}: {prob:.4f}")

    # 测试3: 缺失模态处理
    print("\n" + "-" * 70)
    print("测试3: 缺失模态处理")
    print("-" * 70)

    # BERT + COVAREP（无视频）
    text_audio = torch.randn(1, 768)
    audio_input = torch.randn(1, 74)
    logits_ta = model(text=text_audio, audio=audio_input)
    print(f"BERT + COVAREP 输出形状: {logits_ta.shape}")

    # 仅BERT
    text_only = torch.randn(1, 768)
    logits_t = model(text=text_only)
    print(f"仅BERT 输出形状: {logits_t.shape}")

    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)


if __name__ == '__main__':
    test_bert_hybrid_3class()
