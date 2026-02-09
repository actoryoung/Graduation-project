# -*- coding: utf-8 -*-
"""
多模态融合模块 v2 - 支持预训练编码器

支持使用预训练编码器（BERT、wav2vec、ResNet）的多模态融合模块。
特征维度从1084（300+74+710）升级到3584（768+768+2048）。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

# 导入配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config, NUM_CLASSES, EMOTIONS, DEVICE


class MultimodalFusionModuleV2(nn.Module):
    """
    多模态情感分析融合模块 v2（预训练编码器版本）

    支持使用预训练编码器进行多模态情感分析。
    特征维度：文本768维（BERT）+ 音频768维（wav2vec）+ 视频2048维（ResNet）= 3584维。

    Attributes:
        text_dim: 文本特征维度
        audio_dim: 音频特征维度
        video_dim: 视频特征维度
        fusion_dim: 融合层隐藏维度
        num_classes: 情感类别数
        fusion_layers: 融合层神经网络
        classifier: 情感分类器
        device: 运行设备
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 2048,
        fusion_dim: int = 512,
        num_classes: int = 7,
        dropout_rate: float = 0.3
    ):
        """
        初始化多模态融合模块 v2

        Args:
            text_dim: 文本特征维度（BERT: 768）
            audio_dim: 音频特征维度（wav2vec: 768）
            video_dim: 视频特征维度（ResNet: 2048）
            fusion_dim: 融合层隐藏维度
            num_classes: 情感类别数（默认7）
            dropout_rate: Dropout比率（默认0.3）
        """
        super(MultimodalFusionModuleV2, self).__init__()

        # 计算输入特征总维度
        self.input_dim = text_dim + audio_dim + video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        print(f"融合模块初始化:")
        print(f"  文本维度: {text_dim}")
        print(f"  音频维度: {audio_dim}")
        print(f"  视频维度: {video_dim}")
        print(f"  总输入维度: {self.input_dim}")
        print(f"  融合维度: {fusion_dim}")

        # 定义融合层网络结构（更深的网络以处理更高维度的特征）
        self.fusion_layers = nn.Sequential(
            # 第一层：输入层 -> 隐藏层
            nn.Linear(self.input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 第二层：隐藏层 -> 更小隐藏层
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.65)  # 0.3 * 0.65 ≈ 0.2
        )

        # 定义分类层
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

        # 初始化权重
        self._init_weights()

        # 设置设备
        self.device = DEVICE
        self.to(self.device)

        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  模型参数量: {total_params:,}")

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _handle_missing_modality(
        self,
        text_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        video_feat: Optional[torch.Tensor] = None,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        处理缺失模态，使用零填充策略

        Args:
            text_feat: 文本特征 [batch_size, text_dim] 或 None
            audio_feat: 音频特征 [batch_size, audio_dim] 或 None
            video_feat: 视频特征 [batch_size, video_dim] 或 None
            batch_size: 批次大小

        Returns:
            拼接后的特征 [batch_size, input_dim]
        """
        device = self.device

        # 处理文本特征
        if text_feat is None:
            text_feat = torch.zeros(batch_size, self.text_dim, device=device)
        else:
            text_feat = text_feat.to(device)

        # 处理音频特征
        if audio_feat is None:
            audio_feat = torch.zeros(batch_size, self.audio_dim, device=device)
        else:
            audio_feat = audio_feat.to(device)

        # 处理视频特征
        if video_feat is None:
            video_feat = torch.zeros(batch_size, self.video_dim, device=device)
        else:
            video_feat = video_feat.to(device)

        # 拼接所有模态特征
        fused_feat = torch.cat([text_feat, audio_feat, video_feat], dim=1)

        return fused_feat

    def forward(
        self,
        text_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        video_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            text_feat: 文本特征 [batch_size, 768] 或 None
            audio_feat: 音频特征 [batch_size, 768] 或 None
            video_feat: 视频特征 [batch_size, 2048] 或 None

        Returns:
            logits: 情感分类logits [batch_size, 7]
        """
        # 确定批次大小
        batch_size = 1
        for feat in [text_feat, audio_feat, video_feat]:
            if feat is not None:
                batch_size = feat.shape[0]
                break

        # 处理缺失模态并拼接特征
        fused_input = self._handle_missing_modality(
            text_feat, audio_feat, video_feat, batch_size
        )

        # 通过融合层
        fused_features = self.fusion_layers(fused_input)

        # 通过分类层得到logits
        logits = self.classifier(fused_features)

        return logits

    def predict(
        self,
        text_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        video_feat: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        预测情感，返回完整结果

        Args:
            text_feat: 文本特征 [batch_size, 768] 或 None
            audio_feat: 音频特征 [batch_size, 768] 或 None
            video_feat: 视频特征 [batch_size, 2048] 或 None

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

            # 计算概率分布
            probabilities = torch.softmax(logits, dim=1)

            # 获取预测结果
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_emotion = EMOTIONS[predicted_idx.item()]

            # 构建概率字典
            prob_dict = {
                EMOTIONS[i]: probabilities[0, i].item()
                for i in range(len(EMOTIONS))
            }

            # 构建结果字典
            result = {
                'emotion': predicted_emotion,
                'confidence': confidence.item(),
                'probabilities': prob_dict
            }

        return result

    def get_num_parameters(self) -> int:
        """
        获取模型参数数量

        Returns:
            模型总参数数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_fusion_features(
        self,
        text_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        video_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取融合特征（分类前的特征）

        Args:
            text_feat: 文本特征
            audio_feat: 音频特征
            video_feat: 视频特征

        Returns:
            融合特征 [batch_size, fusion_dim//2]
        """
        self.eval()

        with torch.no_grad():
            batch_size = 1
            for feat in [text_feat, audio_feat, video_feat]:
                if feat is not None:
                    batch_size = feat.shape[0]
                    break

            fused_input = self._handle_missing_modality(
                text_feat, audio_feat, video_feat, batch_size
            )
            fused_features = self.fusion_layers(fused_input)

        return fused_features


def test_fusion_module_v2():
    """测试融合模块 v2"""
    print("=" * 70)
    print("多模态融合模块 v2 测试（预训练编码器版本）")
    print("=" * 70)

    # 创建模型
    print("\n创建模型...")
    model = MultimodalFusionModuleV2()
    print(f"设备: {model.device}")
    print(f"参数数量: {model.get_num_parameters():,}")

    # 测试1: 三模态输入
    print("\n" + "-" * 70)
    print("测试1: 三模态输入（预训练编码器）")
    print("-" * 70)

    batch_size = 2
    text_feat = torch.randn(batch_size, 768)
    audio_feat = torch.randn(batch_size, 768)
    video_feat = torch.randn(batch_size, 2048)

    logits = model(text_feat, audio_feat, video_feat)
    print(f"输入形状: 文本={text_feat.shape}, 音频={audio_feat.shape}, 视频={video_feat.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"预期形状: torch.Size([{batch_size}, {NUM_CLASSES}])")

    # 测试2: 单模态输入（仅文本）
    print("\n" + "-" * 70)
    print("测试2: 单模态输入（仅文本）")
    print("-" * 70)

    text_only = torch.randn(1, 768)
    logits_single = model(text_feat=text_only)
    result = model.predict(text_feat=text_only)

    print(f"输入形状: {text_only.shape}")
    print(f"输出logits形状: {logits_single.shape}")
    print(f"预测情感: {result['emotion']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("概率分布:")
    for emotion, prob in result['probabilities'].items():
        print(f"  {emotion}: {prob:.4f}")

    # 测试3: 缺失模态处理
    print("\n" + "-" * 70)
    print("测试3: 缺失模态处理")
    print("-" * 70)

    # 文本+音频（无视频）
    text_audio = torch.randn(1, 768)
    audio_input = torch.randn(1, 768)
    logits_ta = model(text_feat=text_audio, audio_feat=audio_input)
    print(f"文本+音频 输出形状: {logits_ta.shape}")

    # 音频+视频（无文本）
    audio_video = torch.randn(1, 768)
    video_input = torch.randn(1, 2048)
    logits_av = model(audio_feat=audio_video, video_feat=video_input)
    print(f"音频+视频 输出形状: {logits_av.shape}")

    # 仅视频
    video_only = torch.randn(1, 2048)
    logits_v = model(video_feat=video_only)
    print(f"仅视频 输出形状: {logits_v.shape}")

    # 测试4: 批量预测
    print("\n" + "-" * 70)
    print("测试4: 批量预测")
    print("-" * 70)

    batch_text = torch.randn(3, 768)
    batch_audio = torch.randn(3, 768)
    batch_video = torch.randn(3, 2048)

    batch_logits = model(batch_text, batch_audio, batch_video)
    print(f"批次大小=3, 输出形状: {batch_logits.shape}")

    # 获取融合特征
    fusion_features = model.get_fusion_features(batch_text, batch_audio, batch_video)
    print(f"融合特征形状: {fusion_features.shape}")

    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    test_fusion_module_v2()
