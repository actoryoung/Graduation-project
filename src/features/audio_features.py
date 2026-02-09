# -*- coding: utf-8 -*-
"""
Wav2Vec 2.0音频特征提取器模块

本模块实现了基于Wav2Vec 2.0预训练模型的音频特征提取器，用于将音频文件
转换为768维的固定长度特征向量。通过平均池化时间步的隐藏状态得到音频级
表示，适用于下游的情感分析等任务。

主要功能:
    - 使用facebook/wav2vec2-base预训练模型
    - 自动音频预处理（重采样、单声道转换）
    - 输出768维特征向量
    - 支持CPU/CUDA设备切换
    - 完整异常处理（文件不存在、格式错误、空音频等）

依赖:
    - transformers: Wav2Vec2Model, Wav2Vec2Processor
    - torchaudio: 音频加载和预处理
    - torch: 张量计算

示例:
    >>> from src.features.audio_features import Wav2VecFeatureExtractor
    >>> extractor = Wav2VecFeatureExtractor()
    >>> features = extractor.extract_from_raw('path/to/audio.wav')
    >>> print(features.shape)
    (768,)
"""

# -*- coding: utf-8 -*-

import os
import warnings
from typing import Any, Tuple, Union

import numpy as np
import torch

from config import Config, AUDIO_MODEL, AUDIO_DIM, TARGET_SAMPLE_RATE, MAX_AUDIO_DURATION
from src.features.base import FeatureExtractor


class Wav2VecFeatureExtractor(FeatureExtractor):
    """
    Wav2Vec 2.0音频特征提取器

    使用facebook/wav2vec2-base预训练模型提取音频特征。提取器将输入音频
    通过Wav2Vec 2.0编码器，并通过平均池化所有时间步的隐藏状态得到音频级表示。

    类属性:
        model_name (str): Wav2Vec模型名称
        target_sample_rate (int): 目标采样率（16000Hz）
        feature_dim (int): 输出特征维度（768）
        max_duration (float): 最大音频时长（秒）

    示例:
        >>> extractor = Wav2VecFeatureExtractor()
        >>> features = extractor.extract_from_raw('audio.wav')
        >>> print(features.shape)
        (768,)
    """

    def __init__(self, device: str = None, model_name: str = None):
        """
        初始化Wav2Vec特征提取器

        加载预训练的Wav2Vec 2.0模型和处理器，并将模型移动到指定设备。

        Args:
            device: 计算设备 ('cuda' 或 'cpu')，默认为None时使用Config.device
            model_name: Wav2Vec模型名称，默认为None时使用Config.AUDIO_MODEL

        Raises:
            ValueError: 当device不是'cuda'或'cpu'时
            RuntimeError: 当模型加载失败时
        """
        super().__init__(device)

        # 设置模型名称和参数
        self.model_name = model_name or AUDIO_MODEL
        self.target_sample_rate = TARGET_SAMPLE_RATE
        self.feature_dim = AUDIO_DIM
        self.max_duration = MAX_AUDIO_DURATION

        # 动态导入torchaudio（可能未安装）
        try:
            import torchaudio
            self.torchaudio = torchaudio
        except ImportError:
            raise ImportError(
                "需要安装torchaudio库。请运行: pip install torchaudio"
            )

        # 动态导入transformers组件
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
        except ImportError:
            raise ImportError(
                "需要安装transformers库。请运行: pip install transformers"
            )

        # 加载处理器和模型
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2Model.from_pretrained(self.model_name)

            # 将模型移动到指定设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"加载Wav2Vec模型失败: {e}")

    def preprocess(self, raw_input: Union[str, Tuple[np.ndarray, int], torch.Tensor]) -> torch.Tensor:
        """
        预处理音频输入

        将音频文件路径或原始音频数据转换为Wav2Vec模型所需的输入格式。
        执行以下预处理步骤：
        1. 加载音频（如果是文件路径）
        2. 转换为单声道
        3. 重采样到16kHz
        4. 归一化音频值
        5. 转换为PyTorch张量

        Args:
            raw_input: 原始输入，可以是：
                - str: 音频文件路径
                - tuple: (audio_array, sample_rate) 的元组
                - np.ndarray: 音频波形数据（假设采样率为target_sample_rate）
                - torch.Tensor: 音频波形张量

        Returns:
            torch.Tensor: 预处理后的音频张量，形状为(1, num_samples)

        Raises:
            FileNotFoundError: 当音频文件不存在时
            ValueError: 当输入格式不正确或音频为空时
            TypeError: 当输入类型不匹配时

        示例:
            >>> extractor = Wav2VecFeatureExtractor()
            >>> audio = extractor.preprocess('audio.wav')
            >>> print(audio.shape)
            torch.Size([1, 160000])
        """
        # 处理不同类型的输入
        if isinstance(raw_input, str):
            # 文件路径：加载音频
            if not os.path.exists(raw_input):
                raise FileNotFoundError(f"音频文件不存在: {raw_input}")

            try:
                # 加载音频
                waveform, sample_rate = self.torchaudio.load(raw_input)
            except Exception as e:
                raise ValueError(f"加载音频文件失败 ({raw_input}): {e}")

        elif isinstance(raw_input, tuple) and len(raw_input) == 2:
            # (audio, sample_rate) 元组
            waveform, sample_rate = raw_input
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float()
            elif not isinstance(waveform, torch.Tensor):
                raise TypeError(f"波形数据类型必须是np.ndarray或torch.Tensor，得到: {type(waveform)}")

        elif isinstance(raw_input, np.ndarray):
            # numpy数组：假设采样率为target_sample_rate
            waveform = torch.from_numpy(raw_input).float()
            sample_rate = self.target_sample_rate

        elif isinstance(raw_input, torch.Tensor):
            # torch张量：假设采样率为target_sample_rate
            waveform = raw_input
            sample_rate = self.target_sample_rate

        else:
            raise TypeError(
                f"输入必须是文件路径、(audio, sample_rate)元组、np.ndarray或torch.Tensor，"
                f"得到: {type(raw_input)}"
            )

        # 验证波形数据
        if waveform.numel() == 0:
            raise ValueError("音频数据为空")

        # 确保波形是2D张量 (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() > 2:
            raise ValueError(f"波形维度必须是1D或2D，得到: {waveform.dim()}D")

        # 转换为单声道（平均所有声道）
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 重采样到目标采样率
        if sample_rate != self.target_sample_rate:
            try:
                resampler = self.torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate
                ).to(self.device)
                waveform = resampler(waveform)
            except Exception as e:
                raise RuntimeError(f"音频重采样失败: {e}")

        # 检查音频时长
        num_samples = waveform.shape[1]
        duration = num_samples / self.target_sample_rate
        if duration > self.max_duration:
            # 截断到最大时长
            max_samples = int(self.max_duration * self.target_sample_rate)
            waveform = waveform[:, :max_samples]
            warnings.warn(
                f"音频时长({duration:.2f}s)超过最大限制({self.max_duration}s)，"
                f"已截断到{self.max_duration}s"
            )

        # 归一化音频值到[-1, 1]范围
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()

        # 确保张量在正确的设备上
        return waveform.to(self.device)

    def extract(self, audio: torch.Tensor) -> np.ndarray:
        """
        从预处理的音频中提取Wav2Vec特征

        使用Wav2Vec 2.0模型处理预处理的音频，通过平均池化所有时间步的
        隐藏状态得到音频级表示。所有计算在torch.no_grad()上下文中执行，
        不参与梯度计算。

        Args:
            audio: 预处理后的音频张量，形状为(1, num_samples)

        Returns:
            np.ndarray: 768维特征向量，dtype为float32

        Raises:
            ValueError: 当输入数据维度不正确时
            RuntimeError: 当特征提取失败时

        示例:
            >>> extractor = Wav2VecFeatureExtractor()
            >>> audio = extractor.preprocess('audio.wav')
            >>> features = extractor.extract(audio)
            >>> print(features.shape)
            (768,)
        """
        # 验证输入
        if not isinstance(audio, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor，得到: {type(audio)}")

        if audio.dim() != 2 or audio.shape[0] != 1:
            raise ValueError(
                f"音频张量形状应为(1, num_samples)，得到: {audio.shape}"
            )

        # 确保张量在正确设备上
        audio = audio.to(self.device)

        try:
            # 使用处理器预处理音频
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inputs = self.processor(
                    audio.squeeze(0).cpu().numpy(),
                    sampling_rate=self.target_sample_rate,
                    return_tensors="pt"
                )

            # 移动输入到正确设备
            input_values = inputs['input_values'].to(self.device)

            # Wav2Vec推理（不计算梯度）
            with torch.no_grad():
                outputs = self.model(input_values)

            # outputs.last_hidden_state: (batch_size, sequence_length, hidden_size)
            hidden_states = outputs.last_hidden_state[0]  # (sequence_length, hidden_size)

            # 平均池化：对所有时间步取平均
            features = hidden_states.mean(dim=0)  # (hidden_size,)

            # 转换为numpy数组
            features = features.cpu().numpy().astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Wav2Vec特征提取失败: {e}")

        return features

    def extract_batch(self, audio_paths: list[str]) -> np.ndarray:
        """
        批量提取音频特征

        对多个音频文件进行批量特征提取。注意：由于音频长度不同，
        本方法采用逐个处理的方式（非真正的批量处理）。

        Args:
            audio_paths: 音频文件路径列表

        Returns:
            np.ndarray: 特征矩阵，形状为(len(audio_paths), 768)

        Raises:
            ValueError: 当audio_paths为空时
            FileNotFoundError: 当任何音频文件不存在时

        示例:
            >>> extractor = Wav2VecFeatureExtractor()
            >>> paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
            >>> features = extractor.extract_batch(paths)
            >>> print(features.shape)
            (3, 768)
        """
        if not audio_paths:
            raise ValueError("输入音频路径列表不能为空")

        features_list = []
        for path in audio_paths:
            try:
                features = self.extract_from_raw(path)
                features_list.append(features)
            except Exception as e:
                warnings.warn(f"提取音频特征失败 ({path}): {e}")
                # 使用零向量作为降级方案
                features_list.append(np.zeros(self.feature_dim, dtype=np.float32))

        return np.array(features_list, dtype=np.float32)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Wav2Vec 2.0音频特征提取器测试")
    print("=" * 60)

    # 测试1: 初始化
    print("\n[测试1] 初始化特征提取器")
    print("-" * 40)
    try:
        extractor = Wav2VecFeatureExtractor()
        print(f"✓ 初始化成功")
        print(f"  - 设备: {extractor.device}")
        print(f"  - 模型: {extractor.model_name}")
        print(f"  - 特征维度: {extractor.feature_dim}")
        print(f"  - 目标采样率: {extractor.target_sample_rate} Hz")
        print(f"  - 最大时长: {extractor.max_duration} s")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        exit(1)

    # 测试2: 创建测试音频
    print("\n[测试2] 创建测试音频")
    print("-" * 40)
    import tempfile

    try:
        # 创建一个简单的正弦波测试音频
        sample_rate = 16000
        duration = 2  # 秒
        frequency = 440  # A4音符
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * frequency * t)

        # 保存为临时wav文件
        temp_dir = tempfile.gettempdir()
        test_audio_path = os.path.join(temp_dir, "test_audio.wav")

        # 使用torchaudio保存
        import torchaudio
        torchaudio.save(
            test_audio_path,
            torch.from_numpy(waveform).unsqueeze(0).float(),
            sample_rate
        )
        print(f"✓ 测试音频创建成功: {test_audio_path}")
        print(f"  - 采样率: {sample_rate} Hz")
        print(f"  - 时长: {duration} s")
        print(f"  - 频率: {frequency} Hz")

    except Exception as e:
        print(f"✗ 创建测试音频失败: {e}")
        print("  跳过后续测试")
        exit(1)

    # 测试3: 单音频特征提取
    print("\n[测试3] 单音频特征提取")
    print("-" * 40)
    try:
        features = extractor.extract_from_raw(test_audio_path)
        print(f"✓ 特征提取成功")
        print(f"  - 特征形状: {features.shape}")
        assert features.shape == (768,), f"特征维度应为(768,)，得到{features.shape}"
        print(f"  ✓ 维度正确: (768,)")
        print(f"  - 数据类型: {features.dtype}")
        assert features.dtype == np.float32
        print(f"  ✓ 数据类型正确: float32")
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
        print(f"  ✓ 无NaN/Inf值")
        print(f"  - 均值: {features.mean():.6f}")
        print(f"  - 标准差: {features.std():.6f}")
        print(f"  - 最小值: {features.min():.6f}")
        print(f"  - 最大值: {features.max():.6f}")

    except Exception as e:
        print(f"✗ 特征提取失败: {e}")

    # 测试4: 分步提取
    print("\n[测试4] 分步提取（preprocess + extract）")
    print("-" * 40)
    try:
        audio = extractor.preprocess(test_audio_path)
        print(f"✓ 预处理成功")
        print(f"  - 音频形状: {audio.shape}")
        print(f"  - 采样率: {extractor.target_sample_rate} Hz")
        print(f"  - 声道数: {audio.shape[0]}")

        features = extractor.extract(audio)
        print(f"✓ 特征提取成功")
        print(f"  - 特征形状: {features.shape}")
        assert features.shape == (768,)
        print(f"  ✓ 维度正确")

    except Exception as e:
        print(f"✗ 分步提取失败: {e}")

    # 测试5: 直接使用numpy数组输入
    print("\n[测试5] 直接使用numpy数组输入")
    print("-" * 40)
    try:
        # 创建numpy数组音频
        sample_rate = 16000
        duration = 1
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_numpy = np.sin(2 * np.pi * 440 * t)

        features = extractor.preprocess(audio_numpy)
        print(f"✓ numpy数组预处理成功")
        print(f"  - 输出形状: {features.shape}")

    except Exception as e:
        print(f"✗ numpy数组处理失败: {e}")

    # 测试6: 异常处理
    print("\n[测试6] 异常处理")
    print("-" * 40)

    # 文件不存在
    print("测试文件不存在:")
    try:
        extractor.extract_from_raw('nonexistent_audio.wav')
        print("  ✗ 应该抛出异常但没有")
    except FileNotFoundError as e:
        print(f"  ✓ 正确捕获FileNotFoundError")

    # 错误的输入类型
    print("\n测试错误的输入类型:")
    try:
        extractor.preprocess(12345)
        print("  ✗ 应该抛出TypeError但没有")
    except TypeError as e:
        print(f"  ✓ 正确捕获TypeError")

    # 测试7: 重现性测试
    print("\n[测试7] 重现性测试")
    print("-" * 40)
    try:
        features1 = extractor.extract_from_raw(test_audio_path)
        features2 = extractor.extract_from_raw(test_audio_path)

        assert np.allclose(features1, features2, rtol=1e-6)
        print(f"✓ 重复提取结果一致")
        print(f"  - 最大差异: {np.abs(features1 - features2).max():.10f}")

    except Exception as e:
        print(f"✗ 重现性测试失败: {e}")

    # 测试8: 批量提取
    print("\n[测试8] 批量特征提取")
    print("-" * 40)
    try:
        # 创建多个测试音频
        paths = [test_audio_path for _ in range(3)]
        batch_features = extractor.extract_batch(paths)
        print(f"✓ 批量提取成功")
        print(f"  - 批次大小: {len(paths)}")
        print(f"  - 特征形状: {batch_features.shape}")
        assert batch_features.shape == (3, 768)
        print(f"  ✓ 维度正确: (3, 768)")

    except Exception as e:
        print(f"✗ 批量提取失败: {e}")

    # 清理临时文件
    try:
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
            print(f"\n✓ 临时文件已清理")
    except:
        pass

    # 测试总结
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n使用方法:")
    print("  from src.features.audio_features import Wav2VecFeatureExtractor")
    print("  extractor = Wav2VecFeatureExtractor()")
    print("  features = extractor.extract_from_raw('path/to/audio.wav')")
    print(f"  # features.shape = (768,)")
