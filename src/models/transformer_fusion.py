# -*- coding: utf-8 -*-
"""
Transformer融合的多模态情感分析模型

使用Transformer架构融合文本(BERT)、音频(COVAREP)、视频(OpenFace)特征

架构设计:
    1. 模态编码器: 将各模态特征编码到统一维度
    2. 位置编码: 为不同模态添加位置信息
    3. Transformer编码器: 通过自注意力机制进行跨模态交互
    4. 融合层: 聚合多模态表示
    5. 分类器: 输出3类情感预测

预期性能:
    - Accuracy: 68-75%
    - Macro F1: 0.62-0.70
    - Negative F1: 0.45-0.55
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=3):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度 (3个模态)
        """
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度
        pe = pe.unsqueeze(0)

        # 注册为buffer (不参与梯度更新)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            添加位置编码后的特征
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerFusionModel(nn.Module):
    """Transformer融合的多模态情感分析模型"""

    def __init__(
        self,
        text_dim=768,      # BERT特征维度
        audio_dim=74,      # COVAREP特征维度
        video_dim=710,     # OpenFace特征维度
        hidden_dim=256,    # 隐藏层维度
        num_heads=4,       # 注意力头数
        num_layers=2,      # Transformer层数
        num_classes=3,     # 类别数
        dropout_rate=0.3
    ):
        super(TransformerFusionModel, self).__init__()

        # 模态编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # 音频特征投影层 (从 hidden_dim//2 到 hidden_dim)
        self.audio_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=3)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, text, audio, video):
        """
        前向传播

        Args:
            text: [batch, text_dim] BERT特征
            audio: [batch, audio_dim] COVAREP特征
            video: [batch, video_dim] OpenFace特征

        Returns:
            logits: [batch, num_classes] 分类logits
        """
        # 编码各模态特征
        t_feat = self.text_encoder(text)      # [batch, hidden_dim]
        a_feat = self.audio_encoder(audio)    # [batch, hidden_dim//2]
        v_feat = self.video_encoder(video)    # [batch, hidden_dim]

        # 将音频特征投影到相同维度
        a_feat = self.audio_projection(a_feat)  # [batch, hidden_dim]

        modalities = torch.stack([t_feat, a_feat, v_feat], dim=1)

        # 添加位置编码
        modalities = self.pos_encoding(modalities)

        # Transformer处理
        fused = self.transformer(modalities)  # [batch, 3, hidden_dim]

        # 聚合 (平均池化)
        fused = fused.mean(dim=1)  # [batch, hidden_dim]

        # 融合层
        output = self.fusion(fused)

        # 分类
        logits = self.classifier(output)

        return logits

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 估算模型大小 (MB)
        model_size_mb = total_params * 4 / (1024 * 1024)

        return {
            'num_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }


class TransformerFusionModelWithClassWeights(nn.Module):
    """带类别权重的Transformer融合模型"""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def forward(self, text, audio, video):
        return super().forward(text, audio, video)


def test_transformer_model():
    """测试Transformer模型"""
    print("=" * 70)
    print("测试Transformer融合模型")
    print("=" * 70)

    # 创建模型
    model = TransformerFusionModel(
        text_dim=768,
        audio_dim=74,
        video_dim=710,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=3,
        dropout_rate=0.3
    )

    # 获取模型信息
    info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  参数量: {info['num_parameters']:,}")
    print(f"  可训练参数: {info['trainable_parameters']:,}")
    print(f"  模型大小: {info['model_size_mb']:.2f} MB")

    # 测试前向传播
    batch_size = 4
    text = torch.randn(batch_size, 768)
    audio = torch.randn(batch_size, 74)
    video = torch.randn(batch_size, 710)

    print(f"\n测试输入:")
    print(f"  Text: {text.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Video: {video.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(text, audio, video)

    print(f"\n输出:")
    print(f"  Logits: {logits.shape}")
    print(f"  预测: {logits.argmax(dim=1)}")

    print("\n模型测试成功!")


if __name__ == '__main__':
    test_transformer_model()
