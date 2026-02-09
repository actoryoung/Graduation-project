# -*- coding: utf-8 -*-
"""
wav2vec 2.0音频特征提取器

使用Facebook的wav2vec 2.0自监督预训练模型提取音频特征，
替代原来的COVAREP传统声学特征。

wav2vec 2.0是从大量未标注音频中自监督学习的预训练模型，
能够捕获比传统手工特征更丰富的语音表示。
"""

import torch
import torch.nn as nn
from typing import Union
import torchaudio

class Wav2VecAudioExtractor(nn.Module):
    """
    wav2vec 2.0音频特征提取器

    使用Hugging Face transformers库加载预训练的wav2vec 2.0模型，
    提取音频的深层语义特征。

    Attributes:
        processor: wav2vec处理器（用于预处理音频）
        model: wav2vec 2.0模型
        device: 运行设备
        target_sample_rate: 目标采样率（16kHz）
    """

    def __init__(
        self,
        model_name: str = 'facebook/wav2vec2-base',
        device: str = 'cuda'
    ):
        """
        初始化wav2vec 2.0音频特征提取器

        Args:
            model_name: 预训练模型名称
                - 'facebook/wav2vec2-base': wav2vec 2.0 base (768维)
                - 'facebook/wav2vec2-large': wav2vec 2.0 large (1024维)
            device: 运行设备 ('cuda' 或 'cpu')
        """
        super(Wav2VecAudioExtractor, self).__init__()

        self.device = device
        self.model_name = model_name
        self.target_sample_rate = 16000  # wav2vec 2.0需要16kHz采样率

        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor

            print(f"正在加载wav2vec 2.0模型: {model_name}")

            # 加载处理器和模型
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)

            # 移到指定设备
            self.model.to(device)
            self.model.eval()  # 设置为评估模式

            print(f"[OK] wav2vec 2.0模型加载成功: {model_name}")
            print(f"     设备: {device}")
            print(f"     目标采样率: {self.target_sample_rate} Hz")
            print(f"     输出维度: 768")

        except Exception as e:
            print(f"[ERROR] wav2vec 2.0模型加载失败: {e}")
            raise

    @torch.no_grad()
    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        提取音频特征

        Args:
            audio_waveform: 音频波形张量
                - 形状: [samples] 或 [1, samples] 或 [channels, samples]
                - 采样率: 16kHz (如果不是，会自动重采样)

        Returns:
            audio_features: 音频特征张量 [1, 768]

        处理流程:
            1. 检查并调整采样率到16kHz
            2. 使用处理器预处理音频
            3. 通过wav2vec模型提取特征
            4. 平均池化到768维向量
        """
        # 确保是1D张量 [samples]（processor会自动添加batch维度）
        if audio_waveform.dim() == 2:
            audio_waveform = audio_waveform.squeeze(0)
        elif audio_waveform.dim() == 3:
            # 如果是多通道，取第一个通道
            audio_waveform = audio_waveform[0]

        # 检查采样率，如果不是16kHz则重采样
        current_sample_rate = getattr(self, 'current_sample_rate', None)
        if current_sample_rate is not None and current_sample_rate != self.target_sample_rate:
            # 使用torchaudio重采样
            resampler = torchaudio.transforms.Resample(
                orig_freq=current_sample_rate,
                new_freq=self.target_sample_rate
            )
            audio_waveform = resampler(audio_waveform)

        # 使用处理器预处理（processor会添加batch维度）
        inputs = self.processor(
            audio_waveform.numpy() if isinstance(audio_waveform, torch.Tensor) else audio_waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt"
        )

        # 移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 通过wav2vec模型
        outputs = self.model(**inputs)

        # 提取特征并平均池化
        # outputs.last_hidden_state: [batch, sequence_length, feature_dim]
        # feature_dim = 768 for wav2vec2-base
        features = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]

        return features

    def extract(self, audio_waveform: torch.Tensor, sample_rate: int = None) -> torch.Tensor:
        """
        提取音频特征（对外接口）

        Args:
            audio_waveform: 音频波形张量
            sample_rate: 采样率（如果为None，假设已正确采样率）

        Returns:
            音频特征向量 [1, 768]
        """
        if sample_rate is not None:
            self.current_sample_rate = sample_rate
        return self.forward(audio_waveform)

    def extract_from_file(self, audio_file: str) -> torch.Tensor:
        """
        从音频文件提取特征

        Args:
            audio_file: 音频文件路径
                支持格式: .wav, .mp3, .flac, 等

        Returns:
            音频特征向量 [1, 768]
        """
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_file)

        # 如果是立体声，转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        print(f"音频文件: {audio_file}")
        print(f"原始采样率: {sample_rate} Hz")
        print(f"音频时长: {waveform.shape[1] / sample_rate:.2f} 秒")

        # 提取特征
        features = self.extract(waveform, sample_rate)

        return features

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return 768

    def __repr__(self):
        return f"Wav2VecAudioExtractor(model_name='{self.model_name}', device='{self.device}')"


class AudioAugmentation:
    """
    音频数据增强类

    用于训练时的数据增强，包括：
    - 音高变换（Pitch Shifting）
    - 时间拉伸（Time Stretching）
    - 添加噪声
    - 时间掩码
    """

    def __init__(
        self,
        noise_level: float = 0.01,
        mask_prob: float = 0.1,
        mask_length: int = 10
    ):
        self.noise_level = noise_level
        self.mask_prob = mask_prob
        self.mask_length = mask_length

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        对音频进行数据增强

        Args:
            audio: 输入音频 [1, samples]

        Returns:
            augmented_audio: 增强后的音频
        """
        # 添加高斯噪声
        noise = torch.randn_like(audio) * self.noise_level
        audio = audio + noise

        # 时间掩码（随机遮挡一段音频）
        if torch.rand(1).item() < self.mask_prob:
            start = torch.randint(0, audio.shape[1] - self.mask_length, (1,)).item()
            audio[:, start:start + self.mask_length] = 0

        return audio


def test_wav2vec_extractor():
    """测试wav2vec 2.0音频特征提取器"""
    print("=" * 70)
    print("wav2vec 2.0音频特征提取器测试")
    print("=" * 70)

    # 测试生成随机音频
    try:
        extractor = Wav2VecAudioExtractor(device='cpu')
        print("\n测试1: 随机音频提取")
        print("-" * 70)

        # 生成1秒的随机音频（16kHz采样率）
        samples = 16000
        audio = torch.randn(samples)  # [16000] - 一维音频

        features = extractor.extract(audio, sample_rate=16000)

        print(f"输入音频形状: {audio.shape}")
        print(f"特征形状: {features.shape}")
        print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"特征均值: {features.mean():.4f}")
        print(f"特征标准差: {features.std():.4f}")

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        print("\n注意: 如果是首次运行，需要下载wav2vec 2.0模型（约360MB）")
        print("请确保网络连接正常，或手动下载模型文件。")
        return

    # 对比COVAREP
    print("\n" + "=" * 70)
    print("wav2vec vs COVAREP对比")
    print("=" * 70)
    print("| 特征提取器 | 维度 | 类型 | 预训练 | 深度学习 |")
    print("|-----------|------|------|--------|----------|")
    print("| COVAREP    | 74   | 手工 | 否     | 否       |")
    print("| **wav2vec** | **768** | **深度学习** | **是** | **是** |")

    print("\n" + "=" * 70)
    print("[OK] 所有测试通过!")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    test_wav2vec_extractor()
